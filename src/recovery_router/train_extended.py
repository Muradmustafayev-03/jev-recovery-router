"""Fresh LoRA training with a locked, validation-only checkpoint selection rule.

Selection minimizes the mean of first/expanded raw NLL, each standardized to
the predeclared OOS prior. Test data and fitted temperatures never enter here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

import torch
from torch.nn import functional as F

from .model import NONE_INDEX, bundle_fingerprint, collate_views, create_model, save_model_bundle
from .train import prepare_views, seed_all


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def learning_rate_factor(step, planned_steps, warmup_ratio=.05, min_lr_ratio=.1):
    """One-based optimizer update; linear warmup then cosine to a nonzero floor."""
    if not 1 <= step <= planned_steps or not 0 <= warmup_ratio < 1 or not 0 <= min_lr_ratio <= 1:
        raise ValueError("Invalid learning-rate schedule arguments")
    warmup_steps = min(planned_steps - 1, max(1, math.ceil(planned_steps * warmup_ratio))) if warmup_ratio else 0
    if warmup_steps and step <= warmup_steps:
        return step / warmup_steps
    # The first post-warmup step is already on the cosine; the final step is its floor.
    progress = (step - warmup_steps) / max(1, planned_steps - warmup_steps)
    return min_lr_ratio + (1 - min_lr_ratio) * .5 * (1 + math.cos(math.pi * progress))


def is_better_checkpoint(candidate_nll, best_nll, min_delta=1e-4):
    if not math.isfinite(candidate_nll) or min_delta < 0:
        raise ValueError("Selection requires finite NLL and nonnegative min_delta")
    return best_nll is None or candidate_nll < best_nll - min_delta


@contextmanager
def validation_mode(model):
    """Restore module dropout flags and all training RNG streams, even on failure."""
    module_modes = [(module, module.training) for module in model.modules()]
    python_rng = random.getstate()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    model.eval()
    try:
        with torch.inference_mode():
            yield
    finally:
        for module, mode in module_modes:
            module.training = mode
        random.setstate(python_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def summarize_observations(observations, target_oos_prior):
    """Example-weighted raw metrics and OOS-standardized stage means (NONE kept)."""
    if not observations or not 0 < target_oos_prior < 1:
        raise ValueError("Nonempty validation and an OOS prior strictly between 0 and 1 are required")

    def summarize(rows):
        groups = {False: [r for r in rows if not r["is_oos"]], True: [r for r in rows if r["is_oos"]]}
        if not all(groups.values()):
            raise ValueError("Each validation stage needs both OOS and in-domain observations")
        result = {"views": len(rows), "oos_views": len(groups[True]),
                  "in_domain_views": len(groups[False]),
                  "none_target_views": sum(bool(r.get("none_target", False)) for r in rows)}
        for name in ("nll", "brier", "accuracy"):
            result[name] = math.fsum(r[name] for r in rows) / len(rows)
            result["oos_standardized_" + name] = math.fsum(
                (target_oos_prior if oos else 1 - target_oos_prior)
                * math.fsum(r[name] for r in grouped) / len(grouped)
                for oos, grouped in groups.items())
        return result

    stages = {stage: summarize([r for r in observations if r["stage"] == stage])
              for stage in ("first", "expanded")}
    return {"overall": summarize(observations), "stages": stages,
            "selection_mean_nll": .5 * math.fsum(v["oos_standardized_nll"] for v in stages.values()),
            "target_oos_prior": target_oos_prior,
            "probability_protocol": "raw FP32 log_softmax; all shown candidates plus NONE; no temperature"}


def evaluate_views(model, tokenizer, views, max_length, batch_size, device, target_oos_prior):
    observations = []
    with validation_mode(model):
        for start in range(0, len(views), batch_size):
            selected = views[start:start + batch_size]
            batch = {k: v.to(device) for k, v in collate_views(tokenizer, selected, max_length).items()}
            logits = model(**batch)["logits"].float()
            log_probs = logits.log_softmax(-1)
            nll = F.nll_loss(log_probs, batch["labels"], reduction="none")
            probs = log_probs.exp()
            targets = F.one_hot(batch["labels"], num_classes=logits.shape[-1]).to(probs.dtype)
            brier = (probs - targets).square().sum(-1)
            accuracy = logits.argmax(-1).eq(batch["labels"])
            if not torch.isfinite(nll).all() or not torch.isfinite(brier).all():
                raise FloatingPointError("Non-finite validation probability metrics")
            for view, loss, bs, correct in zip(selected, nll.cpu().tolist(), brier.cpu().tolist(), accuracy.cpu().tolist()):
                observations.append({"stage": view["stage"], "is_oos": view["is_oos"],
                                     "none_target": view["target"] == NONE_INDEX,
                                     "nll": loss, "brier": bs, "accuracy": float(correct)})
    return summarize_observations(observations, target_oos_prior)


def read_records(path):
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Input must be nonempty with unique record IDs")
    return records


def check_disjoint(training, validation):
    if {r["id"] for r in training} & {r["id"] for r in validation}:
        raise ValueError("Training and validation IDs overlap")
    normalized = lambda rows: {" ".join(str(r["text"]).casefold().split()) for r in rows}
    if normalized(training) & normalized(validation):
        raise ValueError("Training and validation normalized query texts overlap")


def prepared(path, tokenizer, max_length, seed, shuffled, records, rejected_path):
    views, rejected = prepare_views(path, tokenizer, max_length, seed, shuffled)
    write_json(rejected_path, rejected)
    if rejected:
        raise ValueError(f"Rejected {len(rejected)} views; refusing silent filtering/truncation. See {rejected_path}")
    oos = {r["id"]: bool(r.get("is_oos", False)) for r in records}
    for view in views:
        view["is_oos"] = oos[view["id"]]
    return views


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    started = time.monotonic()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_cfg, train_cfg = dict(config["model"]), dict(config["training"])
    seed = int(train_cfg.get("seed", 42))
    seed_all(seed)
    epochs = int(train_cfg.get("epochs", 12))
    max_seconds = args.max_seconds if args.max_seconds is not None else float(train_cfg.get("max_seconds", 3600))
    if not 1 <= epochs <= 12 or not 0 < max_seconds <= 3600:
        raise ValueError("This bounded revision permits 1..12 epochs and at most 3600 seconds")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty; existing runs are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    training_records, validation_records = read_records(args.data), read_records(args.validation)
    check_disjoint(training_records, validation_records)
    max_length = int(model_cfg.get("max_length", 512))
    micro_batch = int(train_cfg.get("micro_batch_size", 8))
    accumulation = int(train_cfg.get("gradient_accumulation_steps", 4))
    eval_batch = int(train_cfg.get("validation_batch_size", micro_batch))
    if min(micro_batch, accumulation, eval_batch) < 1:
        raise ValueError("Batch sizes and accumulation must be positive")
    model, tokenizer = create_model(model_cfg, args.device)
    validation_views = prepared(args.validation, tokenizer, max_length, seed, False, validation_records,
                                output / "rejected_validation_views.json")
    # Fixed candidates and input order across every checkpoint comparison.
    validation_views_sha = hashlib.sha256(json.dumps(
        [[v["id"], v["stage"], v["input_ids"], v["target"]] for v in validation_views],
        separators=(",", ":")).encode()).hexdigest()
    prior = float(config.get("policy", {}).get("target_oos_prior", 2 / 11))
    view_count = len(training_records) * 2
    total_batches = math.ceil(view_count / micro_batch)
    steps_per_epoch = math.ceil(total_batches / accumulation)
    planned_steps = steps_per_epoch * epochs
    validation_steps = int(train_cfg.get("validation_steps", math.ceil(steps_per_epoch / 2)))
    if validation_steps < 1:
        raise ValueError("validation_steps must be positive")
    min_delta = float(train_cfg.get("selection_min_delta", 1e-4))
    peak_lr = float(train_cfg.get("learning_rate", 1e-4))
    warmup_ratio = float(train_cfg.get("warmup_ratio", .05))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", .1))
    if peak_lr <= 0 or min_delta < 0:
        raise ValueError("Invalid learning rate or selection delta")
    learning_rate_factor(1, planned_steps, warmup_ratio, min_lr_ratio)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=peak_lr,
                                  weight_decay=float(train_cfg.get("weight_decay", .01)))
    provenance = {"training_data_sha256": sha256(args.data), "data_sha256": sha256(args.data),
                  "validation_data_sha256": sha256(args.validation),
                  "validation_views_sha256": validation_views_sha,
                  "training_config_sha256": sha256(args.config),
                  "max_length": max_length, "seed": seed,
                  "selection_rule": "mean(first,expanded) raw NLL standardized to fixed OOS prior; min_delta favors earlier",
                  "selection_min_delta": min_delta, "selection_target_oos_prior": prior,
                  "selection_validation_records": len(validation_records),
                  "selection_validation_views": len(validation_views),
                  "trainable_parameters": sum(p.numel() for p in parameters)}
    history, epochs_history, step_history = [], [], []
    best_nll, best_step, best_bundle = None, None, None
    global_step, examples_seen, training_loss_sum = 0, 0, 0.
    trained_counts = set()
    stop_reason = "epochs_complete"
    # Reserve time for the final validation and portable bundle copy; hard outer
    # process supervision is still required for a stalled device or filesystem.
    reserve_seconds = min(float(train_cfg.get("finalization_reserve_seconds", 90)), max_seconds / 4)
    if reserve_seconds < 0:
        raise ValueError("Finalization reserve cannot be negative")

    def evaluate_and_save(epoch_fraction):
        nonlocal best_nll, best_step, best_bundle
        tick = time.monotonic()
        metrics = evaluate_views(model, tokenizer, validation_views, max_length, eval_batch, args.device, prior)
        destination = output / "checkpoints" / f"step-{global_step:06d}"
        save_model_bundle(model, tokenizer, destination, model_cfg,
                          {**provenance, "step": global_step, "epoch_fraction": epoch_fraction,
                           "trained_candidate_counts": sorted(trained_counts),
                           "planned_candidate_counts": sorted({v["candidate_count"] for v in validation_views}),
                           "training_examples_seen": examples_seen,
                           "validation_metrics": metrics, "checkpoint_selection_eligible": global_step > 0})
        row = {"step": global_step, "epoch_fraction": epoch_fraction,
               "bundle": destination.relative_to(output).as_posix(),
               "model_fingerprint": bundle_fingerprint(destination),
               "training_examples_seen": examples_seen,
               "cumulative_training_mean_nll": training_loss_sum / examples_seen if examples_seen else None,
               "evaluation_and_save_seconds": time.monotonic() - tick,
               "elapsed_seconds": time.monotonic() - started, **metrics}
        row["selected_at_this_step"] = bool(global_step and is_better_checkpoint(metrics["selection_mean_nll"], best_nll, min_delta))
        if row["selected_at_this_step"]:
            best_nll, best_step, best_bundle = metrics["selection_mean_nll"], global_step, destination
        history.append(row)
        write_json(destination / "validation_metrics.json", row)
        write_json(output / "validation_curve.json", history)
        print(json.dumps({"event": "validation", **row}, allow_nan=False), flush=True)

    print(json.dumps({"event": "prepared", **provenance, "epochs": epochs,
                      "planned_optimizer_steps": planned_steps, "validation_steps": validation_steps,
                      "usable_training_views_per_epoch": view_count, "rejected_views": 0}), flush=True)
    evaluate_and_save(0.)  # Diagnostic only: never eligible for publication.
    model.train()
    epoch_fraction = 0.
    for epoch in range(epochs):
        if time.monotonic() - started >= max_seconds - reserve_seconds:
            stop_reason = "time_cap"
            break
        views = prepared(args.data, tokenizer, max_length, seed + epoch, True, training_records,
                         output / f"rejected_training_views_epoch_{epoch + 1:02d}.json")
        if len(views) != view_count:
            raise ValueError("Every training record must have exactly two views")
        order = list(range(len(views)))
        random.Random(seed + 1000 + epoch).shuffle(order)
        epoch_loss_sum, epoch_examples = 0., 0
        for group_start in range(0, total_batches, accumulation):
            if time.monotonic() - started >= max_seconds - reserve_seconds:
                stop_reason = "time_cap"
                break
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            group_stop = min(total_batches, group_start + accumulation)
            groups = [order[b * micro_batch:(b + 1) * micro_batch] for b in range(group_start, group_stop)]
            count = sum(map(len, groups))
            step_loss = 0.
            for indices in groups:
                batch = {k: v.to(args.device) for k, v in collate_views(tokenizer, [views[i] for i in indices], max_length).items()}
                loss = model(**batch)["loss"] * (len(indices) / count)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at step {global_step + 1}")
                loss.backward()
                step_loss += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(parameters, float(train_cfg.get("max_grad_norm", 1.)), error_if_nonfinite=True)
            lr = peak_lr * learning_rate_factor(global_step + 1, planned_steps, warmup_ratio, min_lr_ratio)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            if args.device == "cuda":
                torch.cuda.synchronize()
            global_step += 1
            examples_seen += count
            epoch_examples += count
            training_loss_sum += step_loss * count
            epoch_loss_sum += step_loss * count
            trained_counts.update(views[i]["candidate_count"] for indices in groups for i in indices)
            epoch_fraction = epoch + epoch_examples / view_count
            row = {"step": global_step, "epoch_fraction": epoch_fraction, "examples": count,
                   "mean_nll": step_loss, "learning_rate": lr,
                   "step_seconds": time.monotonic() - tick, "elapsed_seconds": time.monotonic() - started}
            step_history.append(row)
            if global_step == 1 or global_step % 10 == 0:
                print(json.dumps({"event": "step", **row}), flush=True)
            epoch_optimizer_step = group_start // accumulation + 1
            if epoch_optimizer_step % validation_steps == 0 or group_stop == total_batches:
                evaluate_and_save(epoch_fraction)
        epochs_history.append({"epoch": epoch + 1, "examples": epoch_examples,
                               "mean_nll": epoch_loss_sum / epoch_examples if epoch_examples else None,
                               "complete": epoch_examples == view_count})
        write_json(output / "epoch_training_metrics.json", epochs_history)
        if stop_reason != "epochs_complete":
            break
    if history[-1]["step"] != global_step:
        evaluate_and_save(epoch_fraction)
    if best_bundle is None:
        raise RuntimeError("No trained checkpoint was evaluated; only diagnostic step zero exists")
    # copytree refuses an existing destination: no old inference artifacts can leak in.
    shutil.copytree(best_bundle, output / "model")
    selected_report = {"stop_reason": "validation_selected", "optimizer_steps_total": best_step,
                       "optimizer_steps_this_run": best_step,
                       "actual_executed_optimizer_steps": global_step,
                       "selected_checkpoint": best_bundle.relative_to(output).as_posix(),
                       "selected_validation_mean_nll": best_nll,
                       "planned_optimizer_steps": planned_steps,
                       "usable_views": view_count, "rejected_views": 0,
                       "elapsed_seconds": time.monotonic() - started,
                       "note": "Selected bundle's step is reported above; full executed training history is in training_summary.json."}
    write_json(output / "model" / "training_report.json", selected_report)
    summary = {**provenance, "stop_reason": stop_reason, "actual_executed_optimizer_steps": global_step,
               "planned_optimizer_steps": planned_steps, "actual_training_examples_seen": examples_seen,
               "weighted_training_mean_nll": training_loss_sum / examples_seen,
               "selected_step": best_step, "selected_validation_mean_nll": best_nll,
               "selected_bundle": "model", "selected_checkpoint": best_bundle.relative_to(output).as_posix(),
               "selected_model_fingerprint": bundle_fingerprint(output / "model"),
               "trained_candidate_counts": sorted(trained_counts),
               "rejected_training_views": 0, "rejected_validation_views": 0,
               "epochs": epochs_history, "validation_curve": history, "optimizer_curve": step_history,
               "elapsed_seconds": time.monotonic() - started,
               "max_seconds": max_seconds, "finalization_reserve_seconds": reserve_seconds,
               "note": "Time checked between updates; outer supervisor must bound filesystem/device stalls. No test data used."}
    write_json(output / "training_summary.json", summary)
    print(json.dumps({"event": "finished", **selected_report, "training_stop_reason": stop_reason}), flush=True)


if __name__ == "__main__":
    main()
