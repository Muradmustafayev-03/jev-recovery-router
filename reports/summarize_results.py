"""Extract recorded evidence without recomputing metrics or inventing missing results.

Usage: python reports/summarize_results.py --run runs/a100-001 [--output summary.json]
Additional online artifacts: --online path/to/online.json (repeatable).
Standard library only. Identical artifact snapshots produce identical JSON output.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re

PROVIDERS = ("main", "qwen_base", "jev")
DATASETS = ("test_base", "test", "transfer")
COMPLETE = "measured_offline_evaluation_complete"
PRIMARY_COSTS = {"correct": 1., "wrong": 5., "defer": .2, "expand": .05}
POINT_FIELDS = ("n", "mean_utility", "correct_route_rate", "wrong_route_rate", "coverage",
                "accepted_error", "handoff_rate", "expand_rate", "oos_false_acceptance",
                "accounted_model_ms_p50", "accounted_model_ms_p95", "expansion_diagnostics")
TIMING_FIELDS = ("timing_status", "n", "batch_size", "warmup_requests", "warmup_ms", "p50_ms",
                 "p95_ms", "mean_ms", "min_ms", "max_ms", "action_counts", "expanded_requests",
                 "timing_scope", "limitations", "percentile_method", "policy", "candidate_count")
RETRY_FIELDS = ("workers", "requests_per_second", "max_retries", "max_validation_retries",
                "retry_legacy_cancelled", "retry_transient_http", "retry_uncertain_transport",
                "retry_legacy_unknown_transport")


def mapping(value):
    return value if isinstance(value, dict) else {}


def pick(value, fields):
    value = mapping(value)
    return {key: value[key] for key in fields if key in value}


def finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def reject_constant(value):
    raise ValueError("Nonfinite JSON number")


class Reader:
    def __init__(self, run):
        self.run = Path(run).resolve()
        self.sources = {}

    def load(self, path):
        path = Path(path).resolve()
        try:
            label = path.relative_to(self.run).as_posix()
        except ValueError:
            label = str(path)
        source = {"path": label}
        self.sources[label] = source
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            source["availability"] = "missing"
            return {}, source
        except OSError as error:
            source.update(availability="unreadable", error_type=type(error).__name__)
            return {}, source
        source["sha256"] = hashlib.sha256(raw).hexdigest()
        try:
            data = json.loads(raw.decode("utf-8-sig"), parse_constant=reject_constant)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
        except (ValueError, UnicodeError) as error:
            source.update(availability="invalid_json_or_schema", error_type=type(error).__name__)
            return {}, source
        source["availability"] = "available"
        if "status" in data:
            source["recorded_status"] = data["status"]
        return data, source


def primary_scenario(provider, dataset):
    record = mapping(mapping(provider.get("datasets")).get(dataset))
    return record, mapping(mapping(record.get("scenarios")).get("primary"))


def summarize_ledger_events(events):
    """Separate recorded usage estimates from failed/pending reservation bounds."""
    if not events or not isinstance(events[0], dict) or events[0].get("event") != "settings":
        raise ValueError("Missing ledger settings")
    reservations, successes, failures = {}, {}, []
    def amount(value):
        value = Decimal(str(value))
        if not value.is_finite() or value < 0:
            raise ValueError("Invalid recorded ledger amount")
        return value
    for event in events:
        kind = event.get("event")
        if kind == "settings":
            continue
        key = (event["request_hash"], event.get("attempt", 1))
        if kind == "reserved":
            if key in reservations:
                raise ValueError("Duplicate reservation")
            reservations[key] = amount(event["reserved_usd"])
        elif kind == "success":
            if key not in reservations or key in successes:
                raise ValueError("Unreserved or duplicate success")
            successes[key] = amount(event["estimated_usage_usd"])
            if successes[key] > reservations[key]:
                raise ValueError("Recorded usage exceeds its reservation")
        elif kind == "failed":
            if key not in reservations:
                raise ValueError("Unreserved failure")
            failures.append(event)
        else:
            raise ValueError("Unknown ledger event")
    usage = sum(successes.values(), Decimal(0))
    uncertain = sum((value for key, value in reservations.items() if key not in successes), Decimal(0))
    failed_keys = {(event["request_hash"], event.get("attempt", 1)) for event in failures}
    if failed_keys.intersection(successes):
        raise ValueError("Failed attempt also recorded as success")
    retry_policies = {json.dumps(pick(event.get("retry_policy"), RETRY_FIELDS), sort_keys=True)
                      for event in events if event.get("event") == "reserved" and event.get("retry_policy")}
    return {"recorded_settings": pick(events[0], ("model", "token_ceiling", "input_price_per_million")),
        "observed_retry_policies": [json.loads(policy) for policy in sorted(retry_policies)],
        "request_attempts": len(reservations), "validated_successes": len(successes),
        "failed_events": len(failures), "attempts_without_validated_response": len(reservations) - len(successes),
        "validated_response_input_usage_estimate_usd": str(usage),
        "failed_or_pending_reservation_upper_bound_usd": str(uncertain),
        "failed_attempt_reservations_usd": str(sum((reservations[key] for key in failed_keys), Decimal(0))),
        "conservative_cumulative_reserved_usd": str(usage + uncertain),
        "failed_by_http_status": dict(sorted(Counter(str(e.get("http_status", "not_recorded")) for e in failures).items())),
        "failed_by_error_type": dict(sorted(Counter(e.get("error_type", "not_recorded") for e in failures).items())),
        "recorded_validation_rejection_reasons": dict(sorted(Counter(mapping(e.get("validation")).get("reason", "not_recorded")
            for e in failures if e.get("http_status") == 200).items())),
        "recorded_transport_categories": dict(sorted(Counter(mapping(e.get("transport")).get("category", "not_recorded")
            for e in failures if "http_status" not in e).items())),
        "billing_status": "input_usage_estimates_and_conservative_reservations_not_an_invoice",
        "note": "Failed and pending attempts retain their full recorded reservations. Their actual billing is unknown; do not label the reserved upper bound as actual spend. No tax or extra fees are inferred."}


def read_jev_ledger(reader):
    path = reader.run / "jev/api_ledger.jsonl"
    source = {"path": "jev/api_ledger.jsonl"}
    reader.sources[source["path"]] = source
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        source["availability"] = "missing"
        return {"status": "not_available"}
    except OSError as error:
        source.update(availability="unreadable", error_type=type(error).__name__)
        return {"status": "unreadable"}
    source["sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        lines = raw.decode("utf-8-sig").splitlines()
        partial = bool(raw and not raw.endswith(b"\n"))
        if partial:
            lines = lines[:-1]
        events = [json.loads(line, parse_constant=reject_constant) for line in lines if line.strip()]
        summary = summarize_ledger_events(events)
    except (ValueError, KeyError, TypeError, AttributeError, InvalidOperation) as error:
        source.update(availability="invalid_json_or_schema", error_type=type(error).__name__)
        return {"status": "invalid_ledger_do_not_infer_cost"}
    source["availability"] = "available_prefix" if partial else "available"
    return {"status": "partial_append_snapshot" if partial else "complete_lines_snapshot_not_a_run_completion_claim",
            "source": source["path"], **summary}


def provider_complete(provider):
    return provider.get("status") == COMPLETE and all(
        mapping(primary_scenario(provider, dataset)[1].get("methods")) for dataset in DATASETS)


def interval_direction(comparison):
    interval = mapping(comparison).get("ci95")
    if (not isinstance(interval, list) or len(interval) != 2
            or not all(finite(value) for value in interval) or interval[0] > interval[1]):
        return "unavailable_or_invalid"
    return "above_zero" if interval[0] > 0 else "below_zero" if interval[1] < 0 else "includes_zero"


def summarize_h1(main):
    record, scenario = primary_scenario(main, "test_base")
    methods = mapping(scenario.get("methods"))
    comparison = mapping(mapping(scenario.get("paired_utility_comparisons")).get("confidence_gate"))
    result = {"dataset": "test_base", "mixture": "empirical", "left": "main/rl_seed42",
              "right": "main/confidence_gate", "status": "not_available",
              "recorded_provider_status": main.get("status"), "recorded_comparison": comparison,
              "interval_direction": interval_direction(comparison),
              "interpretation": "Inspect wrong-route rate and coverage alongside utility; interval direction alone is not a practical-benefit claim."}
    result.update(pick(record, ("n", "oos_prevalence")))
    result["costs"] = scenario.get("costs")
    for method, label in (("rl_seed42", "left_point"), ("confidence_gate", "right_point")):
        result[label] = pick(mapping(methods.get(method)).get("overall"), POINT_FIELDS)
    if comparison and result["left_point"] and result["right_point"]:
        result["status"] = "available_complete" if provider_complete(main) else "available_partial_do_not_claim_h1"
        if scenario.get("costs") != PRIMARY_COSTS:
            result["status"] = "incompatible_primary_costs"
        elif interval_direction(comparison) == "unavailable_or_invalid":
            result["status"] = "missing_or_invalid_primary_interval"
    return result


def extract_provider(name, data, result):
    result["provider_status"][name] = {
        "recorded_status": data.get("status"), "evaluation_complete": provider_complete(data),
        **pick(data, ("provider_fingerprint", "seal_sha256", "test_data_hashes", "comparison_scope"))}
    for dataset, record in sorted(mapping(data.get("datasets")).items()):
        record = mapping(record)
        scenario = mapping(mapping(record.get("scenarios")).get("primary"))
        dataset_key = f"{name}/{dataset}"
        result["datasets"][dataset_key] = {
            "arm": "natural" if dataset in DATASETS else "secondary_stress",
            **pick(record, ("n", "oos_prevalence", "candidate_diagnostics")),
            **pick(scenario, ("costs", "paired_utility_comparisons", "main_vs_confidence_gate_by_group",
                              "actor_seed_summary"))}
        result["calibration"][dataset_key] = pick(record, ("calibration", "calibration_by_group"))
        for method, grouped in sorted(mapping(scenario.get("methods")).items()):
            grouped = mapping(grouped)
            point = {"provider": name, "dataset": dataset, "method": method,
                     **pick(grouped.get("overall"), POINT_FIELDS)}
            result["primary_scenario_points"].append(point)
            if re.fullmatch(r"(?:rl|exact|fixed_cost)_seed\d+", method):
                result["actor_seed_utility"].append(pick(point, ("provider", "dataset", "method", "n", "mean_utility")))
            if dataset in ("test", "transfer"):
                for group, summary in sorted(mapping(grouped.get("group")).items()):
                    result["transfer_groups"].append({"provider": name, "dataset": dataset,
                        "method": method, "group": group, **pick(summary, POINT_FIELDS)})


def cross_provider_comparisons(matrix, providers, classical):
    records = []
    for key, comparison in sorted(mapping(matrix.get("paired_cross_provider_utility")).items()):
        dataset, separator, competitor = key.partition("_main_minus_")
        row = {"name": key, "recorded_comparison": comparison,
               "status": "unverified_comparison_name"}
        if not separator:
            records.append(row)
            continue
        left_provider = providers.get("main", {})
        _, left = primary_scenario(left_provider, dataset)
        left_point = mapping(mapping(mapping(left.get("methods")).get("rl_seed42")).get("overall"))
        if competitor.startswith("classical_"):
            reference = competitor.removeprefix("classical_")
            right_provider = classical
            right_point = mapping(mapping(mapping(classical.get("evaluations")).get(reference)).get("overall"))
            right_costs = mapping(mapping(classical.get("fitting")).get("utility"))
            costs_match = (left.get("costs", {}).get("correct") == right_costs.get("correct")
                and finite(right_costs.get("wrong")) and left.get("costs", {}).get("wrong") == -right_costs["wrong"]
                and finite(right_costs.get("handoff")) and left.get("costs", {}).get("defer") == -right_costs["handoff"])
            right_complete = len(mapping(classical.get("evaluations"))) == 12 and "elapsed_seconds" in classical
        else:
            right_provider = providers.get(competitor, {})
            _, right = primary_scenario(right_provider, dataset)
            right_point = mapping(mapping(mapping(right.get("methods")).get("rl_seed42")).get("overall"))
            costs_match = left.get("costs") == right.get("costs") and bool(left.get("costs"))
            right_complete = provider_complete(right_provider)
        row.update(dataset=dataset, left="main/rl_seed42", right=competitor,
                   left_point=pick(left_point, POINT_FIELDS), right_point=pick(right_point, POINT_FIELDS))
        left_hash = mapping(left_provider.get("test_data_hashes")).get(dataset)
        right_hash = mapping(right_provider.get("test_data_hashes")).get(dataset)
        recorded_difference = mapping(comparison).get("difference")
        values = (left_point.get("mean_utility"), right_point.get("mean_utility"), recorded_difference)
        points_match = all(finite(value) for value in values) and math.isclose(
            values[0] - values[1], values[2], rel_tol=1e-9, abs_tol=1e-9)
        row["snapshot_checks"] = {"matching_data_hashes": bool(left_hash and left_hash == right_hash),
                                  "matching_costs": costs_match, "matching_recorded_difference": points_match}
        row["status"] = ("available_complete" if provider_complete(left_provider) and right_complete
                         and all(row["snapshot_checks"].values()) and interval_direction(comparison) != "unavailable_or_invalid"
                         else "partial_or_stale_or_incompatible_do_not_claim_matched_comparison")
        records.append(row)
    return records


def summarize(run, online_paths=()):
    reader = Reader(run)
    run = reader.run
    result = {"schema_version": 1, "run": str(run),
              "extractor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "status": "available_artifact_snapshot_not_a_completion_certificate",
              "notes": ["Outcome metrics and intervals are copied; ledger totals are summed from recorded entries. Missing metrics are not invented.",
                        "Null metrics retain their source meaning, including undefined accepted error.",
                        "accounted_model_ms fields are cached component accounting, never online latency.",
                        "Primary scenario means primary costs; only natural main/test_base seed42 versus confidence_gate is H1.",
                        "Recorded source statuses may describe partial or stale downloaded artifacts."],
              "provider_status": {}, "datasets": {}, "primary_scenario_points": [],
              "transfer_groups": [], "actor_seed_utility": [], "calibration": {}}
    providers = {}
    for name in PROVIDERS:
        providers[name], source = reader.load(run / name / "results/results.json")
        extract_provider(name, providers[name], result)
        result["provider_status"][name]["availability"] = source["availability"]
    result["primary_h1"] = summarize_h1(providers["main"])
    matrix, _ = reader.load(run / "BENCHMARK_MATRIX.json")
    result["benchmark_matrix"] = pick(matrix, ("status", "benchmark_matrix", "note"))
    classical, source = reader.load(run / "classical/results.json")
    result["classical"] = {"availability": source["availability"],
        **pick(classical, ("status", "limitations", "fit_seconds", "elapsed_seconds")),
        "evaluations": {name: pick(mapping(record).get("overall"), POINT_FIELDS)
                        for name, record in sorted(mapping(classical.get("evaluations")).items())}}
    result["cross_provider_comparisons"] = cross_provider_comparisons(matrix, providers, classical)

    online = {run / "main/online_latency.json", *(Path(path) for path in online_paths)}
    online.update(run.glob("*/online*.json"))
    online.update(run.glob("online*.json"))
    result["online_latency"] = []
    for path in sorted({path.resolve() for path in online}, key=str):
        data, source = reader.load(path)
        result["online_latency"].append({"source": source["path"], "availability": source["availability"],
            **pick(data, TIMING_FIELDS),
            **pick(data, ("created_utc", "hardware", "model_fingerprint", "data_sha256", "retrieval",
                          "selection", "candidate_schedule", "costs", "artifact_and_index_load_ms", "online_comparison")),
            "direct_baseline": pick(data.get("direct_baseline"), TIMING_FIELDS)})
    ledger, _ = reader.load(run / "run_ledger.json")
    training, _ = reader.load(run / "main/model/training_report.json")
    orchestration, _ = reader.load(run / "jev/orchestration.json")
    recovery_note, recovery_source = reader.load(run / "jev/RECOVERY_NOTE.json")
    test_recovery, test_recovery_source = reader.load(run / "jev/TEST_PHASE_RECOVERY.json")
    result["resources"] = {
        "run_ledger": pick(ledger, ("status", "start_unix", "preparation_hours", "allowed_gpu_run_hours",
            "elapsed_gpu_run_hours", "estimated_compute_usd", "estimated_total_usd", "billing_caveat", "error_type", "missing_competitors")),
        "stages": [pick(stage, ("name", "start_utc", "seconds", "exit_code"))
                   for stage in ledger.get("stages", []) if isinstance(stage, dict)],
        "training": pick(training, ("stop_reason", "optimizer_steps_this_run", "optimizer_steps_total",
            "planned_optimizer_steps", "usable_views", "rejected_views", "elapsed_seconds",
            "mean_step_seconds_excluding_first_five", "last_loss", "note")),
        "jev_orchestration": pick(orchestration, ("status", "model", "api_cap_usd", "request_cap", "workers",
            "requests_per_second", "max_retries", "max_seconds", "started_utc", "finished_utc",
            "completed_stages", "price_assumption_per_million", "total_cost_reserve_usd", "cost_limitations")),
        "jev_cumulative_snapshots": [],
        "jev_ledger_snapshot": read_jev_ledger(reader),
        "cost_note": "GPU amounts are recorded estimates, not invoices. Jev metadata snapshots share a cumulative ledger: do not sum them or confuse orchestration caps/reserves with actual charges."}
    policy_disclosure = pick(orchestration, ("response_validation", "validation_retry_policy",
        "rejected_vendor_responses", "execution_source_sha256", "protocol_deviation"))
    if recovery_note:
        policy_disclosure["isolated_recovery_note"] = {"source": recovery_source["path"],
            **pick(recovery_note, ("execution_source_commit", "worktree", "reason", "protocol", "retry_policy",
                                  "source_sha256", "original_deadline_utc", "main_model_source_unchanged"))}
    if orchestration.get("resume_history"):
        policy_disclosure["resume_history"] = [pick(entry, ("resumed_utc", "completed_stages", *RETRY_FIELDS))
                                               for entry in orchestration["resume_history"]]
    if test_recovery:
        policy_disclosure["test_phase_operational_recovery"] = {"source": test_recovery_source["path"],
            **pick(test_recovery, ("status", "mode", "old_validation_retry_cap", "new_validation_retry_cap",
                "api_cap_usd", "request_cap", "original_deadline_unix", "seal_sha256", "final_seal_sha256",
                "started_utc", "finished_utc", "execution_source_commit", "source_sha256", "reason", "deviation",
                "original_prediction_metadata", "original_prediction_metadata_sha256", "final_prediction_metadata",
                "final_prediction_metadata_sha256", "exit_code"))}
    result["execution_disclosures"] = {"jev_rejected_responses_and_retries": {
        "policy_disclosure_status": "recorded_fields_available" if policy_disclosure else "not_recorded",
        "recorded_policy_disclosure": policy_disclosure,
        "recorded_execution_options": pick(orchestration, RETRY_FIELDS),
        "prediction_execution_options": [],
        "prediction_failures": [],
        "note": "A validation exception does not identify its cause. Record rejected-response handling, any changed retry rules and isolated runtime provenance explicitly; do not infer a rounding error."}}
    for path in sorted((run / "jev/predictions").glob("*.jsonl.meta.json")):
        data, source = reader.load(path)
        result["resources"]["jev_cumulative_snapshots"].append({"source": source["path"],
            **pick(data, ("model", "complete", "completed_records", "expected_records", "comparison_status",
                "request_attempts_in_ledger", "reserved_usd_in_ledger", "input_price_per_million_assumption",
                "billing_bound_assumption", "budget_accounting", "rejections_in_ledger", *RETRY_FIELDS))})
        result["execution_disclosures"]["jev_rejected_responses_and_retries"]["prediction_execution_options"].append({
            "source": source["path"], **pick(data, ("complete", "completed_records", "rejections_in_ledger", *RETRY_FIELDS))})
        if data.get("error"):
            result["execution_disclosures"]["jev_rejected_responses_and_retries"]["prediction_failures"].append({
                "source": source["path"], **pick(data, ("error", "complete", "completed_records", "expected_records"))})
    result["sources"] = [reader.sources[name] for name in sorted(reader.sources)]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--online", action="append", default=[])
    parser.add_argument("--output", help="Optional JSON output; otherwise writes to stdout")
    args = parser.parse_args(argv)
    if not Path(args.run).is_dir():
        parser.error("--run must be an existing directory")
    # Escaped Unicode keeps redirected stdout valid on Windows legacy code pages.
    rendered = json.dumps(summarize(args.run, args.online), ensure_ascii=True, indent=2,
                          sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
