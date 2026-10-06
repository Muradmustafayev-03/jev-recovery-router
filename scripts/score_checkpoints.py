"""Read-only v1 checkpoint learning curve on the full tuning split; no baseline replacement."""
from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recovery_router.calibration import mixture_weights
from recovery_router.io import read_json, read_jsonl, sha256_file, write_json
from recovery_router.model import (NONE_INDEX, bundle_fingerprint, collate_views,
                                   load_model_bundle, mask_logits)
from recovery_router.train import prepare_views


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Checkpoint-scoring time cap reached; remaining checkpoints were not scored")


def score_views(model, tokenizer, views, rows, target_oos_prior, *, device="cpu", batch_size=32,
                max_length=512, deadline=None):
    """FP32 per-view raw scores, followed by float64 weighted aggregation."""
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or not rows:
        raise ValueError("Validation must contain unique nonempty query IDs")
    expected = {(identifier, stage) for identifier in by_id for stage in ("first", "expanded")}
    if len(views) != len(expected) or {(v["id"], v["stage"]) for v in views} != expected:
        raise ValueError("Validation views must cover every query exactly once at both stages")
    weights = dict(zip(by_id, mixture_weights(rows, target_oos_prior)))
    scored = {stage: [] for stage in ("first", "expanded")}
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(views), batch_size):
            check_deadline(deadline)
            chunk = views[offset:offset + batch_size]
            batch = {key: value.to(device) for key, value in collate_views(tokenizer, chunk, max_length).items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                           candidate_counts=batch["candidate_counts"])["logits"]
            logits = mask_logits(logits.float(), batch["candidate_counts"])
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            labels = batch["labels"]
            nll = -log_probs.gather(1, labels[:, None]).squeeze(1)
            one_hot = torch.nn.functional.one_hot(labels, probs.shape[-1]).float()
            brier = (probs - one_hot).square().sum(dim=-1)
            accuracy = logits.argmax(dim=-1).eq(labels).float()
            if not torch.isfinite(nll).all() or not torch.isfinite(brier).all() or not torch.isfinite(probs).all():
                raise ValueError("Nonfinite validation scores; refusing a partial population")
            values = torch.stack((nll, brier, accuracy), dim=-1).cpu().tolist()
            for view, value in zip(chunk, values):
                scored[view["stage"]].append((view, value))
            check_deadline(deadline)
    stages = {}
    for stage, entries in scored.items():
        values = np.asarray([value for _, value in entries], dtype=np.float64)
        stage_weights = np.asarray([weights[view["id"]] for view, _ in entries])
        metrics = {"n": len(entries), "n_oos": sum(bool(by_id[v["id"]]["is_oos"]) for v, _ in entries),
                   "n_none_targets": sum(v["target"] == NONE_INDEX for v, _ in entries)}
        metrics["n_shown_targets"] = metrics["n"] - metrics["n_none_targets"]
        for column, name in enumerate(("nll", "brier", "accuracy")):
            metrics[name] = float(values[:, column].mean())
            metrics["weighted_" + name] = float(np.average(values[:, column], weights=stage_weights))
        stages[stage] = metrics
    return {"selector_weighted_nll": .5 * sum(m["weighted_nll"] for m in stages.values()),
            "selector_unweighted_nll": .5 * sum(m["nll"] for m in stages.values()), "stages": stages}


def checkpoint_paths(model_root):
    model_root = Path(model_root).resolve()
    final = read_json(model_root / "metadata.json")
    paths = sorted((path for path in (model_root / "checkpoints").glob("step-*") if path.is_dir()),
                   key=lambda path: int(read_json(path / "metadata.json")["step"]))
    paths += [model_root]
    for path in paths:
        metadata = read_json(path / "metadata.json")
        if metadata["model"] != final["model"]:
            raise ValueError("Checkpoint backbone/config differs from the final v1 bundle")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seconds", type=float, default=600)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0 or args.batch_size <= 0:
        parser.error("Time cap and batch size must be positive and finite")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    started = time.monotonic()
    deadline = started + args.max_seconds
    root, output = Path(args.model_root).resolve(), Path(args.output).resolve()
    if output.is_relative_to(root) or output in (Path(args.validation).resolve(), Path(args.config).resolve()):
        raise ValueError("Diagnostic output must be outside the original bundles and input files")
    if output.exists():
        raise FileExistsError("Diagnostic output already exists; use a fresh path")
    config = read_json(args.config)
    prior = float(config["policy"]["target_oos_prior"])
    rows = read_jsonl(args.validation)
    if not rows or any(row.get("split") != "tuning" or row.get("group") not in ("stable", "oos") for row in rows):
        raise ValueError("Only the complete declared tuning split is eligible; test/novel/transfer rows are forbidden")
    paths = checkpoint_paths(root)
    report = {"status": "in_progress", "diagnostic_only": True,
              "baseline_model": str(root), "baseline_model_fingerprint": bundle_fingerprint(root),
              "validation_path": str(Path(args.validation).resolve()), "validation_sha256": sha256_file(args.validation),
              "config_sha256": sha256_file(args.config), "scorer_sha256": sha256_file(__file__),
              "target_oos_prior": prior, "query_count": len(rows), "view_count": 2 * len(rows),
              "selector": "0.5*(first weighted raw FP32 NLL + expanded weighted raw FP32 NLL)",
              "tie_tolerance": 1e-4, "tie_rule": "earlier optimizer step",
              "max_seconds": args.max_seconds, "batch_size": args.batch_size, "device": args.device,
              "planned_checkpoints": [str(path) for path in paths], "checkpoints": [],
              "note": "Read-only exploratory learning curve. The final v1 model remains the baseline; no calibration or training occurs."}
    write_json(output, report)
    alarm_handler = None
    if hasattr(signal, "setitimer"):
        def timed_out(signum, frame):
            raise TimeoutError("Checkpoint-scoring time cap reached")
        alarm_handler = signal.signal(signal.SIGALRM, timed_out)
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic()))
    try:
        for path in paths:
            check_deadline(deadline)
            model, tokenizer, metadata = load_model_bundle(path, args.device)
            try:
                check_deadline(deadline)
                max_length = int(metadata["model"].get("max_length", 512))
                views, rejected = prepare_views(args.validation, tokenizer, max_length, shuffle_candidates=False)
                if rejected:
                    raise ValueError("Rejected validation views; refusing a reduced validation population")
                if not {view["candidate_count"] for view in views} <= set(metadata.get("trained_candidate_counts", [])):
                    raise ValueError("Checkpoint lacks trained candidate counts required by validation")
                result = score_views(model, tokenizer, views, rows, prior, device=args.device,
                                     batch_size=args.batch_size, max_length=max_length, deadline=deadline)
                result.update(path=str(path), step=int(metadata["step"]), is_final_baseline=path == root,
                              model_fingerprint=bundle_fingerprint(path), metadata_sha256=sha256_file(path / "metadata.json"))
                report["checkpoints"].append(result)
                write_json(output, report)
                print(f"Scored step {result['step']}: weighted NLL={result['selector_weighted_nll']:.6f}", flush=True)
            finally:
                del model, tokenizer
                gc.collect()
                if args.device == "cuda":
                    torch.cuda.empty_cache()
        best = None
        for result in sorted(report["checkpoints"], key=lambda item: item["step"]):
            if best is None or result["selector_weighted_nll"] < best["selector_weighted_nll"] - report["tie_tolerance"]:
                best = result
        report.update(status="complete", best_checkpoint_diagnostic_only={key: best[key] for key in ("path", "step", "selector_weighted_nll")})
    except BaseException as error:
        report.update(status="incomplete", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if alarm_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, alarm_handler)
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(output, report)


if __name__ == "__main__":
    main()
