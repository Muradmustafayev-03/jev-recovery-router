"""Summarize a recorded exploratory follow-up without changing either experiment.

Extraction and rendering use the standard library. Paired intervals lazily use the
existing recovery_router.metrics.paired_bootstrap routine (and its NumPy dependency).
Missing dependencies or evidence are reported explicitly, never approximated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("test_base", "test", "transfer")
COMPLETE = "measured_offline_evaluation_complete"
POINTS = ("n", "mean_utility", "correct_route_rate", "wrong_route_rate", "coverage",
          "accepted_error", "handoff_rate", "expand_rate", "oos_false_acceptance")


def mapping(value):
    return value if isinstance(value, dict) else {}


def pick(value, keys):
    value = mapping(value)
    return {key: value[key] for key in keys if key in value}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def reject_constant(value):
    raise ValueError("Nonfinite JSON number")


class Evidence:
    def __init__(self):
        self.sources = {}

    def read(self, path, *, jsonl=False):
        path = Path(path).resolve()
        source = {"path": str(path)}
        self.sources[str(path)] = source
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            source["availability"] = "missing"
            return None
        except OSError as error:
            source.update(availability="unreadable", error_type=type(error).__name__)
            return None
        source["sha256"] = hashlib.sha256(raw).hexdigest()
        try:
            text = raw.decode("utf-8-sig")
            value = ([json.loads(line, parse_constant=reject_constant) for line in text.splitlines() if line.strip()]
                     if jsonl else json.loads(text, parse_constant=reject_constant))
            if not isinstance(value, (dict, list)):
                raise ValueError("Expected JSON object or array")
        except (ValueError, UnicodeError):
            source["availability"] = "invalid_json_or_schema"
            return None
        source["availability"] = "available"
        return value


def primary(result, dataset):
    record = mapping(mapping(result.get("datasets")).get(dataset))
    scenario = mapping(mapping(record.get("scenarios")).get("primary"))
    point = mapping(mapping(mapping(scenario.get("methods")).get("rl_seed42")).get("overall"))
    return record, scenario, point


def checked_outcomes(rows, costs, point, dataset_record):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Missing or empty per-query outcomes")
    if set(costs) != {"correct", "wrong", "defer", "expand"} or not all(finite(v) for v in costs.values()):
        raise ValueError("Invalid recorded utility costs")
    by_id = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] in by_id:
            raise ValueError("Invalid or duplicate query ID")
        if not all(isinstance(row.get(key), bool) for key in ("accepted", "correct", "expanded", "is_oos")):
            raise ValueError("Missing outcome decisions")
        if row.get("intervention") != "natural":
            raise ValueError("Natural comparison contains intervention outcomes")
        parent = row.get("parent_id", row["id"])
        if not isinstance(parent, str):
            raise ValueError("Invalid parent query ID")
        utility = ((costs["correct"] if row["correct"] else -costs["wrong"])
                   if row["accepted"] else -costs["defer"])
        utility -= costs["expand"] if row["expanded"] else 0
        if not finite(row.get("utility")) or not math.isclose(row["utility"], utility, abs_tol=1e-10):
            raise ValueError("Recorded outcome utility disagrees with recorded costs and decisions")
        by_id[row["id"]] = row
    if len(rows) != point.get("n") or len(rows) != dataset_record.get("n"):
        raise ValueError("Outcome count disagrees with results artifact")
    mean = math.fsum(row["utility"] for row in rows) / len(rows)
    if not finite(point.get("mean_utility")) or not math.isclose(mean, point["mean_utility"], abs_tol=1e-10):
        raise ValueError("Outcome mean utility disagrees with results artifact")
    return by_id


def compare_dataset(evidence, baseline, followup, results, dataset, samples, seed):
    result = {"status": "incomplete", "dataset": dataset, "mixture": "empirical",
              "left": "followup/main/rl_seed42", "right": "baseline/main/rl_seed42"}
    records = [primary(record, dataset) for record in results]
    result.update(baseline_point=pick(records[0][2], POINTS), followup_point=pick(records[1][2], POINTS))
    hashes = [mapping(record.get("test_data_hashes")).get(dataset) for record in results]
    costs = [record[1].get("costs") for record in records]
    result["checks"] = {"matching_natural_data_hashes": bool(hashes[0] and hashes[0] == hashes[1]),
                        "matching_costs": bool(costs[0] and costs[0] == costs[1]),
                        "completed_evaluations": all(record.get("status") == COMPLETE for record in results)}
    result["costs"] = costs[0] if result["checks"]["matching_costs"] else None
    result["data_sha256"] = hashes[0] if result["checks"]["matching_natural_data_hashes"] else None
    if not all(result["checks"].values()):
        result["reason"] = "Missing, incomplete, or incompatible evaluation artifacts"
        return result
    outcomes = [evidence.read(run / "main/results" / dataset / "rl_seed42.jsonl", jsonl=True)
                for run in (baseline, followup)]
    try:
        indexed = [checked_outcomes(rows, cost, record[2], record[0])
                   for rows, cost, record in zip(outcomes, costs, records)]
        if indexed[0].keys() != indexed[1].keys():
            raise ValueError("Paired comparison needs identical query IDs")
        for key in indexed[0]:
            left, right = indexed[0][key], indexed[1][key]
            if any(left.get(field) != right.get(field) for field in ("group", "is_oos", "intervention")) or (
                    left.get("parent_id", key) != right.get("parent_id", key)):
                raise ValueError("Matched query grouping differs")
        result["checks"].update(unique_matching_query_ids=True, matching_parent_queries=True,
                                recorded_utility_verified=True, point_estimates_verified=True)
        # Preserve the repository's exact bootstrap, RNG and cluster weighting.
        sys.path.insert(0, str(ROOT / "src"))
        from recovery_router.metrics import paired_bootstrap
        interval = paired_bootstrap(outcomes[1], outcomes[0], samples=samples, seed=seed)
        expected = records[1][2]["mean_utility"] - records[0][2]["mean_utility"]
        if not math.isclose(interval["difference"], expected, abs_tol=1e-10):
            raise ValueError("Paired difference disagrees with recorded mean utilities")
        low, high = interval["ci95"]
        result.update(status="available_exploratory", paired_utility=interval, n=len(indexed[0]),
                      interval_direction="above_zero" if low > 0 else "below_zero" if high < 0 else "includes_zero")
    except ImportError as error:
        result.update(reason="Existing paired-bootstrap dependency unavailable", error_type=type(error).__name__)
    except (ValueError, KeyError, TypeError) as error:
        result.update(status="invalid_evidence", reason=str(error))
    return result


def training_evidence(evidence, baseline, followup):
    original = mapping(evidence.read(baseline / "main/model/training_report.json"))
    original_metadata = mapping(evidence.read(baseline / "main/model/metadata.json"))
    summary = mapping(evidence.read(followup / "main/training_summary.json"))
    metadata = mapping(evidence.read(followup / "main/model/metadata.json"))
    diagnostic = mapping(evidence.read(followup / "v1_validation_curve.json"))
    curve = summary.get("validation_curve")
    if not isinstance(curve, list):
        curve = evidence.read(followup / "main/validation_curve.json")
    curve = curve if isinstance(curve, list) else []
    epochs = summary.get("epochs")
    if not isinstance(epochs, list):
        epochs = evidence.read(followup / "main/epoch_training_metrics.json")
    epochs = epochs if isinstance(epochs, list) else []
    selected = [row for row in curve if isinstance(row, dict) and row.get("step") == summary.get("selected_step")]
    checks = {"summary_present": bool(summary), "selected_step_in_curve": len(selected) == 1,
              "selected_bundle_step_matches": bool(summary) and summary.get("selected_step") == metadata.get("step"),
              "trained_step_selected": isinstance(summary.get("selected_step"), int) and summary["selected_step"] > 0,
              "v1_diagnostic_complete": diagnostic.get("status") == "complete"}
    if len(selected) == 1:
        checks["selected_score_matches"] = (finite(summary.get("selected_validation_mean_nll"))
            and finite(selected[0].get("selection_mean_nll"))
            and math.isclose(summary["selected_validation_mean_nll"], selected[0]["selection_mean_nll"], abs_tol=1e-10))
    return {"status": "available" if all(checks.values()) else "incomplete", "checks": checks,
            "baseline": {"training_report": original, "final_step": original_metadata.get("step"),
                         "loss_interpretation": "last_loss is the final optimizer batch loss, not a mean over training or validation."},
            "followup": {**pick(summary, ("stop_reason", "actual_executed_optimizer_steps", "planned_optimizer_steps",
                "actual_training_examples_seen", "weighted_training_mean_nll", "selected_step",
                "selected_validation_mean_nll", "selected_checkpoint", "selected_model_fingerprint",
                "elapsed_seconds", "max_seconds", "training_data_sha256", "validation_data_sha256")),
                "selected_epoch_fraction": selected[0].get("epoch_fraction") if len(selected) == 1 else None,
                "selected_record": selected[0] if len(selected) == 1 else None,
                "validation_curve": curve, "epoch_training_mean_nll": epochs,
                "optimizer_curve": summary.get("optimizer_curve", []),
                "training_loss_weighting": "Example-weighted across executed updates; not OOS-prior standardization."},
            "v1_validation_diagnostic": diagnostic,
            "diagnostic_caveat": "The recorded final v1 model remains the baseline regardless of its diagnostic best checkpoint."}


def summarize(baseline, followup, samples=1000, seed=2026):
    if samples < 1:
        raise ValueError("Bootstrap samples must be positive")
    baseline, followup = Path(baseline).resolve(), Path(followup).resolve()
    evidence = Evidence()
    results = [mapping(evidence.read(run / "main/results/results.json")) for run in (baseline, followup)]
    ledgers = [mapping(evidence.read(run / "run_ledger.json")) for run in (baseline, followup)]
    comparisons = {name: compare_dataset(evidence, baseline, followup, results, name, samples, seed) for name in DATASETS}
    training = training_evidence(evidence, baseline, followup)
    complete = (all(record.get("status") == "complete" for record in ledgers)
                and training["status"] == "available"
                and all(row["status"] == "available_exploratory" for row in comparisons.values()))
    metrics_path = ROOT / "src/recovery_router/metrics.py"
    return {"schema_version": 1, "status": "complete_exploratory" if complete else "incomplete",
            "classification": "exploratory_after_v1_test_inspection", "baseline": str(baseline), "followup": str(followup),
            "claims_status": "descriptive_exploratory_only" if complete else "no_complete_followup_claim",
            "notes": ["Follow-up was specified after inspecting v1 tests; these are not fresh confirmatory tests.",
                      "Duration, learning-rate schedule, augmentation and selection changed together; no attribution to epochs alone.",
                      "Intervals resample matched parent queries in the empirical mixture using the existing routine.",
                      "Intervals are descriptive, without multiplicity correction or deployment-shift guarantees.",
                      "A missing result is not evidence of no effect. Recorded costs are simulated utility, not invoices."],
            "bootstrap": {"samples": samples, "seed": seed, "routine": "recovery_router.metrics.paired_bootstrap",
                          "source_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest()},
            "extractor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "natural_seed42_comparisons": comparisons, "training": training,
            "resources": {name: pick(ledger, ("status", "start_unix", "followup_start_unix", "end_unix",
                "followup_elapsed_seconds", "total_cloud_elapsed_hours", "elapsed_gpu_run_hours", "estimated_compute_usd",
                "estimated_total_usd", "gpu_rate_assumption_usd", "billing_caveat", "error_type", "error"))
                for name, ledger in zip(("baseline", "followup"), ledgers)},
            "sources": [evidence.sources[key] for key in sorted(evidence.sources)]}


def markdown(report):
    lines = ["# Recorded v1–v2 follow-up comparison", "", f"Status: **{report['status']}**.", "",
             "Exploratory follow-up specified after v1 test inspection. Changes cannot be attributed to epochs alone.", "",
             "| Natural dataset | Evidence | v1 utility | v2 utility | v2 − v1 (95% paired interval) |",
             "|---|---|---:|---:|---|"]
    number = lambda value: f"{value:.6f}" if finite(value) else "unavailable"
    for name, row in report["natural_seed42_comparisons"].items():
        interval = row.get("paired_utility", {})
        ci = interval.get("ci95")
        delta = f"{number(interval.get('difference'))} [{number(ci[0])}, {number(ci[1])}]" if ci else "unavailable"
        lines.append(f"| {name} | {row['status']} | {number(row['baseline_point'].get('mean_utility'))} | "
                     f"{number(row['followup_point'].get('mean_utility'))} | {delta} |")
    training = report["training"]["followup"]
    lines += ["", f"Selected trained step: {training.get('selected_step', 'unavailable')}; "
              f"epoch fraction: {number(training.get('selected_epoch_fraction'))}.",
              f"Example-weighted mean training NLL: {number(training.get('weighted_training_mean_nll'))}.",
              "Full validation, optimizer and epoch curves are retained in the JSON report.", "",
              "The v1 final-batch loss is not a training mean. Its final model remains the baseline even if an earlier checkpoint scores better on tuning.",
              "Intervals are descriptive and unadjusted for multiple comparisons. Missing or incompatible evidence prevents a completed follow-up claim.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="runs/a100-001")
    parser.add_argument("--followup", default="runs/a100-extended-001")
    parser.add_argument("--output", default="reports/extended-summary.json")
    parser.add_argument("--markdown", help="Defaults to the JSON output path with .md suffix")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    output = Path(args.output).resolve()
    md_output = Path(args.markdown).resolve() if args.markdown else output.with_suffix(".md")
    for path in (output, md_output):
        if any(path.is_relative_to(Path(run).resolve()) for run in (args.baseline, args.followup)):
            parser.error("Report outputs must be outside both experiment directories")
    if output == md_output:
        parser.error("JSON and Markdown output paths must differ")
    report = summarize(args.baseline, args.followup, args.bootstrap_samples, args.seed)
    for path, contents in ((output, json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"),
                           (md_output, markdown(report))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output), "markdown": str(md_output)}))


if __name__ == "__main__":
    main()
