"""Read-only, post-selection train/tuning diagnostics for the selected model.

Run with PYTHONPATH pointing to the frozen extended experiment's src directory.
This script does not add the main checkout's src to sys.path, fit any parameter,
choose a checkpoint, or read test outcomes. It intentionally reuses the exact
frozen trainer's prepare_views and evaluate_views implementations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import signal
import time


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Post-selection training-gap audit time cap reached")


def _runtime():
    try:
        import torch
        from recovery_router import model as model_module
        from recovery_router import train as original_trainer
        from recovery_router import train_extended as extended_trainer
    except ImportError as error:
        raise RuntimeError("Set PYTHONPATH to the frozen extended experiment's src directory") from error
    return torch, model_module, original_trainer, extended_trainer


def validate_provenance(model_path, summary_path, train_path, validation_path, config_path, fingerprint):
    metadata = read_json(model_path / "metadata.json")
    summary = read_json(summary_path)
    config = read_json(config_path)
    training_report = read_json(model_path / "training_report.json")
    step = metadata.get("step")
    if (not isinstance(step, int) or isinstance(step, bool) or step <= 0
            or step != summary.get("selected_step") or metadata.get("checkpoint_selection_eligible") is not True
            or training_report.get("stop_reason") != "validation_selected"
            or training_report.get("optimizer_steps_total") != step):
        raise ValueError("Only the positively trained, validation-selected model is eligible for this diagnostic")
    if summary.get("selected_model_fingerprint") != fingerprint:
        raise ValueError("Selected model fingerprint does not match its training summary")
    if summary.get("validation_views_sha256") != metadata.get("validation_views_sha256"):
        raise ValueError("Selected model and summary disagree on fixed tuning views")
    for key, path in (("training_data_sha256", train_path), ("validation_data_sha256", validation_path),
                      ("training_config_sha256", config_path)):
        actual = sha256(path)
        if metadata.get(key) != actual or summary.get(key) != actual:
            raise ValueError(f"Selected metadata/summary do not match the supplied input: {key}")
    prior = config["policy"]["target_oos_prior"]
    if (not isinstance(prior, (float, int)) or isinstance(prior, bool) or not math.isfinite(prior)
            or not math.isclose(prior, 2 / 11, rel_tol=0, abs_tol=1e-12)
            or metadata.get("selection_target_oos_prior") != prior
            or summary.get("selection_target_oos_prior") != prior):
        raise ValueError("Diagnostic requires the same predeclared OOS prior 2/11 used for selection")
    if metadata.get("model") != config.get("model"):
        raise ValueError("Backbone/configuration differs from the selected model")
    return metadata, summary, config


def fixed_views(path, records, tokenizer, max_length, original_trainer):
    views, rejected = original_trainer.prepare_views(path, tokenizer, max_length, shuffle_candidates=False)
    expected = {(r["id"], stage) for r in records for stage in ("first", "expanded")}
    if rejected or len(views) != len(expected) or {(v["id"], v["stage"]) for v in views} != expected:
        raise ValueError("Refusing rejected, truncated, duplicated or incomplete diagnostic views")
    oos_by_id = {r["id"]: bool(r.get("is_oos", False)) for r in records}
    for view in views:
        view["is_oos"] = oos_by_id[view["id"]]
    return views


def view_sha256(views):
    return hashlib.sha256(json.dumps([[v["id"], v["stage"], v["input_ids"], v["target"]] for v in views],
                                     separators=(",", ":")).encode()).hexdigest()


def parameter_versions(model):
    return {"parameter:" + name: (parameter._version, parameter.requires_grad, parameter.grad is None)
            for name, parameter in model.named_parameters()} | {
                "buffer:" + name: value._version for name, value in model.named_buffers()}


def run_audit(args):
    started = time.monotonic()
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 300 or args.batch_size < 1:
        raise ValueError("Audit requires a positive batch size and a time cap in (0,300] seconds")
    model_path, summary_path, train_path, validation_path, config_path, output = (
        Path(getattr(args, name)).resolve() for name in ("model", "summary", "train", "validation", "config", "output"))
    if output.exists():
        raise FileExistsError("Audit output already exists; choose a new path")
    if output.is_relative_to(model_path) or output in (summary_path, train_path, validation_path, config_path):
        raise ValueError("Audit output must be outside its model bundle and input files")
    torch, model_module, original_trainer, extended_trainer = _runtime()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    deadline = started + args.max_seconds
    protected = {"summary": summary_path, "train": train_path, "validation": validation_path, "config": config_path}
    hashes_before = {key: sha256(path) for key, path in protected.items()}
    fingerprint_before = model_module.bundle_fingerprint(model_path)
    metadata, summary, config = validate_provenance(model_path, summary_path, train_path,
                                                    validation_path, config_path, fingerprint_before)
    check_deadline(deadline)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "in_progress", "classification": "post_selection_diagnostic_not_checkpoint_selection",
              "created_utc": datetime.now(timezone.utc).isoformat(), "selected_step": metadata["step"],
              "model_path": str(model_path), "model_fingerprint_before": fingerprint_before,
              "input_sha256": hashes_before, "target_oos_prior": config["policy"]["target_oos_prior"],
              "device": args.device, "batch_size": args.batch_size, "max_seconds": args.max_seconds,
              "optimizer_created": False, "optimizer_updates": 0, "checkpoint_selection_changed": False,
              "candidate_order": "fixed original retrieval order in both full splits; no candidate permutation",
              "training_mode": "eval; dropout disabled; inference_mode; all parameters require_grad=False",
              "source_sha256": {str(Path(inspect.getfile(module)).resolve()): sha256(inspect.getfile(module))
                                for module in (model_module, original_trainer, extended_trainer)},
              "script_sha256": sha256(__file__), "splits": {}}
    write_json(output, report)
    previous_alarm = None
    if hasattr(signal, "setitimer"):
        def timed_out(signum, frame):
            raise TimeoutError("Post-selection training-gap audit time cap reached")
        previous_alarm = signal.signal(signal.SIGALRM, timed_out)
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic()))
    model = None
    hooks = []
    versions_before = None
    try:
        training_records = extended_trainer.read_records(train_path)
        validation_records = extended_trainer.read_records(validation_path)
        extended_trainer.check_disjoint(training_records, validation_records)
        if (any(r.get("split") != "train" for r in training_records)
                or any(r.get("split") != "tuning" for r in validation_records)):
            raise ValueError("Only the complete declared train and tuning splits are allowed")
        model, tokenizer, loaded_metadata = model_module.load_model_bundle(model_path, args.device, trainable=False)
        if loaded_metadata != metadata:
            raise ValueError("Model metadata changed during loading")
        model.eval().requires_grad_(False)
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("Diagnostic model unexpectedly contains trainable parameters")
        versions_before = parameter_versions(model)
        # These hooks bound every evaluation batch without replacing the frozen evaluator.
        hooks.append(model.register_forward_pre_hook(lambda module, inputs: check_deadline(deadline)))
        hooks.append(model.register_forward_hook(lambda module, inputs, outputs: check_deadline(deadline)))
        for split, path, rows in (("train", train_path, training_records), ("tuning", validation_path, validation_records)):
            tick = time.monotonic()
            check_deadline(deadline)
            views = fixed_views(path, rows, tokenizer, int(metadata["max_length"]), original_trainer)
            if not {v["candidate_count"] for v in views} <= set(metadata["trained_candidate_counts"]):
                raise ValueError("Diagnostic candidate counts exceed the trained model support")
            views_sha = view_sha256(views)
            if split == "tuning" and views_sha != metadata.get("validation_views_sha256"):
                raise ValueError("Fixed tuning views differ from those used in original checkpoint selection")
            check_deadline(deadline)
            metrics = extended_trainer.evaluate_views(model, tokenizer, views, int(metadata["max_length"]),
                                                       args.batch_size, args.device, report["target_oos_prior"])
            check_deadline(deadline)
            report["splits"][split] = {"queries": len(rows), "views": len(views), "rejected_views": 0,
                                       "views_sha256": views_sha, "elapsed_seconds": time.monotonic() - tick,
                                       "metrics": metrics}
            write_json(output, report)
        train, validation = (report["splits"][split]["metrics"] for split in ("train", "tuning"))
        metrics = ("nll", "brier", "accuracy", "oos_standardized_nll", "oos_standardized_brier", "oos_standardized_accuracy")
        report["validation_minus_training"] = {
            "overall": {key: validation["overall"][key] - train["overall"][key] for key in metrics},
            "stages": {stage: {key: validation["stages"][stage][key] - train["stages"][stage][key] for key in metrics}
                       for stage in ("first", "expanded")},
            "selection_mean_nll": validation["selection_mean_nll"] - train["selection_mean_nll"]}
        report["validation_repeat"] = {
            "recorded_selection_mean_nll": summary["selected_validation_mean_nll"],
            "diagnostic_selection_mean_nll": validation["selection_mean_nll"],
            "difference": validation["selection_mean_nll"] - summary["selected_validation_mean_nll"],
            "selection_batch_size": config["training"].get("validation_batch_size", config["training"]["micro_batch_size"]),
            "diagnostic_batch_size": args.batch_size,
            "note": "Identical frozen weights/views; GPU batch/padding numerics may differ. This repeat never changes selection."}
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="incomplete", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if previous_alarm is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm)
        for hook in hooks:
            hook.remove()
        report["model_fingerprint_after"] = model_module.bundle_fingerprint(model_path)
        report["input_sha256_after"] = {key: sha256(path) for key, path in protected.items()}
        report["model_and_inputs_unchanged"] = (report["model_fingerprint_after"] == fingerprint_before
                                                and report["input_sha256_after"] == hashes_before)
        report["parameter_versions_unchanged"] = None if model is None else parameter_versions(model) == versions_before
        report["elapsed_seconds"] = time.monotonic() - started
        report["limitations"] = [
            "This is a post-selection diagnostic, never a new checkpoint selector or test evaluation.",
            "Train and tuning each use a fixed candidate order; training-time permutations/dropout are not reproduced.",
            "Unweighted scores reflect each split's own arrival mixture. OOS standardization equalizes only OOS prevalence.",
            "The mean uses complete splits and fixed selected weights, not final-minibatch or changing-weight training losses.",
            "A train/development gap alone cannot establish underfitting, overfitting, causation or deployment quality.",
            "Signal/batch checks bound scoring; final integrity hashing may briefly exceed the nominal time cap."]
        if not report["model_and_inputs_unchanged"] or (versions_before is not None and not report["parameter_versions_unchanged"]):
            report["status"] = "integrity_failure"
            write_json(output, report)
            raise RuntimeError("Model/input artifacts or in-memory parameter versions changed during read-only audit")
        write_json(output, report)
    print(json.dumps({"output": str(output), "status": report["status"], "selected_step": metadata["step"],
                      "train_queries": len(training_records), "tuning_queries": len(validation_records),
                      "weighted_nll_gap": report["validation_minus_training"]["selection_mean_nll"],
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "summary", "train", "validation", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seconds", type=float, default=300)
    return run_audit(parser.parse_args(argv))


if __name__ == "__main__":
    main()
