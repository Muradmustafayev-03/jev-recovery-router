"""Optional paid Jev comparator: six splits, shared caps, fit/seal, evaluate.

Nothing invokes this script by default. --dry-run has no network effects.
Actual execution requires --execute-paid-api and TYPESAFE_API_KEY.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recovery_router.io import read_json, read_jsonl, read_verified_predictions, sha256_file, write_json
from recovery_router.jev import MAX_INPUT_TOKENS, build_request, canonical_hash

FIT = ("calibration", "tuning", "policy_train")
TEST = ("test_base", "test", "transfer")


def original_deadline(record):
    started = datetime.fromisoformat(record["started_utc"])
    if started.tzinfo is None:
        raise ValueError("Resume requires an unambiguous original start timestamp")
    deadline = started.timestamp() + record["max_seconds"]
    if not math.isfinite(deadline) or record["max_seconds"] <= 0:
        raise ValueError("Invalid original elapsed-time cap")
    if "deadline_unix" in record and abs(record["deadline_unix"] - deadline) > .001:
        raise ValueError("Original deadline differs from the recorded start and cap")
    return deadline


def comparable_stages(stages):
    result = []
    for stage in stages:
        tokens = list(stage["command"])
        if stage["name"] == "aggregate_comparisons" and len(tokens) > 1:
            script = Path(tokens[1])
            if script.name == "aggregate_results.py" and script.is_file():
                tokens[1] = "aggregate_results.py:sha256:" + sha256_file(script)
        command = iter(tokens)
        cleaned = []
        for token in command:
            if token == "--max-validation-retries":
                next(command)
            elif token not in ("--retry-legacy-cancelled", "--retry-transient-http", "--retry-uncertain-transport", "--retry-legacy-unknown-transport"):
                cleaned.append(token)
        result.append({"name": stage["name"], "command": cleaned})
    return result


def validate_completed_stage(stage, plan):
    command, name = stage["command"], stage["name"]
    def value(flag):
        return command[command.index(flag) + 1]
    provider = f"api-version:{plan['model']}:choice-protocol-v1"
    if name.startswith("jev_predict_"):
        read_verified_predictions(value("--data"), value("--output"), provider)
    elif name == "jev_fit_and_seal":
        from recovery_router.evaluate import verify_seal
        seal = verify_seal(Path(value("--output")) / "seal.json")
        if seal.get("provider_fingerprint") != provider:
            raise ValueError("Completed fitting seal has a different provider")
    elif name == "jev_evaluate":
        result = read_json(Path(value("--output")) / "results.json")
        if (result.get("status") != "measured_offline_evaluation_complete"
                or result.get("provider_fingerprint") != provider
                or result.get("seal_sha256") != sha256_file(Path(value("--policies")) / "seal.json")
                or any(result.get("test_data_hashes", {}).get(split) != plan["splits"][split]["sha256"]
                       for split in TEST)):
            raise ValueError("Completed evaluation is incomplete or has changed provenance")
    elif name == "aggregate_comparisons":
        run = Path(value("--run"))
        result = read_json(run / "BENCHMARK_MATRIX.json")
        if result.get("status") != "comparison_summary" or not (run / "COMPARISON.md").is_file():
            raise ValueError("Completed comparison summary is missing or invalid")


def load_resume(args, plan):
    output = Path(args.run).resolve() / "jev"
    record = read_json(output / "orchestration.json")
    fields = ("model", "splits", "config_sha256", "unique_requests_before_cache", "shared_api_ledger",
              "api_cap_usd", "request_cap", "workers", "requests_per_second", "max_retries",
              "max_seconds", "price_assumption_per_million")
    for field in fields:
        if record.get(field) != plan[field]:
            raise ValueError(f"Resume refuses changed {field}; original caps and inputs must be preserved")
    if comparable_stages(record["stages"]) != comparable_stages(plan["stages"]):
        raise ValueError("Resume refuses changed stage commands")
    if not Path(record["shared_api_ledger"]).is_file():
        raise ValueError("Original shared API ledger is missing; refusing to reset paid-request accounting")
    completed = record.get("completed_stages", [])
    if completed != [stage["name"] for stage in plan["stages"][:len(completed)]]:
        raise ValueError("Completed stages must be an ordered prefix of the original plan")
    for stage in plan["stages"][:len(completed)]:
        validate_completed_stage(stage, plan)
    original_deadline(record)
    return record


def prepare_plan(args):
    workers = getattr(args, "workers", 1)
    requests_per_second = getattr(args, "requests_per_second", 0.)
    max_retries = getattr(args, "max_retries", 0)
    validation_retries = getattr(args, "max_validation_retries", 0)
    if not 0 <= validation_retries <= 2:
        raise ValueError("Validation retry limit must be 0..2")
    if (not 1 <= workers <= 64 or not 0 <= max_retries <= 10
            or not math.isfinite(requests_per_second) or requests_per_second < 0):
        raise ValueError("Invalid worker, request-rate or retry limit")
    if not math.isfinite(args.max_usd) or args.max_usd <= 0:
        raise ValueError("--max-usd must be finite and positive")
    if args.max_requests <= 0 or args.max_seconds <= 0:
        raise ValueError("Request and elapsed-time caps must be positive")
    if not math.isfinite(args.input_price_per_million) or args.input_price_per_million <= 0:
        raise ValueError("Input-price assumption must be finite and positive")
    if not re.fullmatch(r"jev-\d+\.\d+\.\d+", args.model):
        raise ValueError("Use a pinned Jev version")
    run, data = Path(args.run).resolve(), Path(args.data_dir).resolve()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    budget = config.get("budget", {})
    reservation = MAX_INPUT_TOKENS * args.input_price_per_million / 1_000_000
    if args.max_usd < reservation:
        raise ValueError("API budget is below one conservative 64000-token request reservation")
    data_info, hashes = {}, set()
    for split in FIT + TEST:
        path = data / (split + ".jsonl")
        rows = read_jsonl(path)
        if not rows:
            raise ValueError(f"Empty required split: {split}")
        for row in rows:
            for expanded in (False, True):
                hashes.add(canonical_hash(build_request(row, expanded, args.model)))
        data_info[split] = {"rows": len(rows), "sha256": sha256_file(path)}
    if args.max_requests < len(hashes):
        raise ValueError(f"Request cap {args.max_requests} is below {len(hashes)} distinct uncached requests needed for six complete splits")
    # Reserve the full requested API cap and the maximum additional VM time.
    # Known previous cloud elapsed time is updated from the existing run ledger.
    previous = run / "run_ledger.json"
    tax = float(budget.get("assumed_tax_multiplier", 1.))
    gpu_rate = float(budget.get("gpu_usd_per_hour", 0.))
    hours_limit = float(budget.get("max_elapsed_hours", 12.))
    ceiling = float(budget.get("max_usd", 100.))
    incidentals = float(budget.get("incidentals_usd", 0.))
    if (any(not math.isfinite(v) or v <= 0 for v in (tax, hours_limit, ceiling))
            or any(not math.isfinite(v) or v < 0 for v in (gpu_rate, incidentals))):
        raise ValueError("Configured budget rates and limits must be finite and valid")
    if previous.exists():
        ledger = read_json(previous)
        start = float(ledger["start_unix"])
        preparation = float(ledger.get("preparation_hours", 0.))
        if not math.isfinite(start) or not math.isfinite(preparation) or preparation < 0:
            raise ValueError("Invalid existing run time ledger")
        elapsed = max(0., time.time() - start)
        prior_estimate = (elapsed / 3600 * gpu_rate + incidentals) * tax
        remaining_seconds = hours_limit * 3600 - preparation * 3600 - elapsed
        prior_status = "updated estimate from existing run start; billing may differ"
    else:
        prior_estimate = args.prior_spend_usd
        remaining_seconds = hours_limit * 3600
        prior_status = "explicit prior-spend assumption; no main run ledger available"
    if not math.isfinite(prior_estimate) or prior_estimate < 0:
        raise ValueError("Prior spend must be finite and nonnegative")
    reserved_seconds = args.max_seconds
    if getattr(args, "resume", False):
        previous_comparison = read_json(run / "jev" / "orchestration.json")
        reserved_seconds = max(0., original_deadline(previous_comparison) - time.time())
    if reserved_seconds > remaining_seconds:
        raise ValueError("Requested optional-comparison time exceeds the remaining overall time envelope")
    reserved_total = prior_estimate + (args.max_usd + reserved_seconds / 3600 * gpu_rate) * tax
    if reserved_total > ceiling:
        raise ValueError("Prior spend plus API/time reserve exceeds the configured total budget; reduce caps")
    output = run / "jev"
    shared_cache = output / "api_ledger.jsonl"
    python = sys.executable
    stages = []
    def predict(split):
        stages.append({"name": "jev_predict_" + split, "command": [python, "-m", "recovery_router.jev",
            "--data", str(data/(split+".jsonl")), "--output", str(output/"predictions"/(split+".jsonl")),
            "--cache", str(shared_cache), "--model", args.model, "--max-usd", str(args.max_usd),
            "--max-requests", str(args.max_requests), "--input-price-per-million", str(args.input_price_per_million),
            "--workers", str(workers), "--requests-per-second", str(requests_per_second),
            "--max-retries", str(max_retries), "--max-validation-retries", str(validation_retries),
            *(["--retry-transient-http"] if getattr(args, "retry_transient_http", False) else []),
            *(["--retry-uncertain-transport"] if getattr(args, "retry_uncertain_transport", False) else []),
            *(["--retry-legacy-unknown-transport"] if getattr(args, "retry_legacy_unknown_transport", False) else []),
            *(["--retry-legacy-cancelled"] if getattr(args, "retry_legacy_cancelled", False) else [])]})
    for split in FIT:
        predict(split)
    stages.append({"name": "jev_fit_and_seal", "command": [python, "-m", "recovery_router.evaluate", "fit",
        "--data-dir", str(data), "--pred-dir", str(output/"predictions"), "--output", str(output/"policies"),
        "--config", str(config_path)]})
    for split in TEST:
        predict(split)
    stages.append({"name": "jev_evaluate", "command": [python, "-m", "recovery_router.evaluate", "test",
        "--data-dir", str(data), "--pred-dir", str(output/"predictions"), "--policies", str(output/"policies"),
        "--output", str(output/"results")]})
    stages.append({"name": "aggregate_comparisons", "command": [python, str(ROOT/"scripts/aggregate_results.py"), "--run", str(run)]})
    return {"status": "plan_only_not_run", "model": args.model, "splits": data_info,
            "unique_requests_before_cache": len(hashes), "shared_api_ledger": str(shared_cache),
            "api_cap_usd": args.max_usd, "request_cap": args.max_requests,
             "workers": workers, "requests_per_second": requests_per_second, "max_retries": max_retries,
             "max_validation_retries": validation_retries,
             "retry_legacy_cancelled": getattr(args, "retry_legacy_cancelled", False),
             "retry_transient_http": getattr(args, "retry_transient_http", False),
             "retry_uncertain_transport": getattr(args, "retry_uncertain_transport", False),
             "retry_legacy_unknown_transport": getattr(args, "retry_legacy_unknown_transport", False),
            "max_seconds": args.max_seconds, "price_assumption_per_million": args.input_price_per_million,
            "per_request_conservative_reservation_usd": reservation,
            "prior_cost_estimate_usd": prior_estimate, "prior_cost_status": prior_status,
            "total_cost_reserve_usd": reserved_total, "config_sha256": sha256_file(config_path),
            "cost_limitations": "Configured input price/token ceiling and tax/VM assumptions; no output fees assumed. Verify prices before paid use.",
            "comparison_scope": "Matched processed queries/candidates; probability provider differs. Offline policy evaluation does not establish matched online serving latency.",
            "stages": stages}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Shared experiment root, e.g. runs/h100-001")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--config", default="configs/h100.json")
    parser.add_argument("--model", default="jev-1.13.0")
    parser.add_argument("--max-usd", type=float, required=True)
    parser.add_argument("--max-requests", type=int, required=True)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--prior-spend-usd", type=float, default=0., help="Used only without an existing main-run ledger; includes any prior charges")
    parser.add_argument("--input-price-per-million", type=float, default=.042)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--requests-per-second", type=float, default=0.)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--retry-transient-http", action="store_true",
                        help="Include transient 500/502/503/504/520-524 in the existing HTTP retry cap")
    parser.add_argument("--retry-uncertain-transport", action="store_true",
                        help="Include allowlisted uncertain network failures in the combined HTTP/transport retry cap")
    parser.add_argument("--retry-legacy-unknown-transport", action="store_true",
                        help="Allow one legacy reason-less URLError diagnostic retry within the same combined retry cap")
    parser.add_argument("--max-validation-retries", type=int, default=0, choices=range(3))
    parser.add_argument("--retry-legacy-cancelled", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Verify and continue the original plan, ledger and elapsed-time cap")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-paid-api", action="store_true")
    args = parser.parse_args(argv)
    plan = prepare_plan(args)
    previous_record = load_resume(args, plan) if args.resume else None
    if args.dry_run:
        if previous_record is not None:
            plan.update(resume_completed_stages=previous_record["completed_stages"],
                        original_deadline_unix=original_deadline(previous_record),
                        remaining_seconds=max(0., original_deadline(previous_record)-time.time()))
        print(json.dumps(plan, indent=2)); return
    if not args.execute_paid_api:
        parser.error("Actual requests require --execute-paid-api; use --dry-run to review first")
    if not os.environ.get("TYPESAFE_API_KEY", "").strip():
        parser.error("TYPESAFE_API_KEY is absent; no network request was sent")
    output = Path(args.run).resolve() / "jev"
    if previous_record is None and output.exists() and any(output.iterdir()):
        raise FileExistsError("Jev output already exists; choose a fresh experiment root. The immutable API ledger is retained; do not delete it to bypass caps.")
    output.mkdir(parents=True, exist_ok=True)
    record = previous_record or {**plan, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "completed_stages": []}
    record["deadline_unix"] = original_deadline(record)
    if time.time() >= record["deadline_unix"]:
        raise TimeoutError("Original optional-comparison time cap reached; resume cannot reset it")
    if previous_record is not None:
        record.setdefault("resume_history", []).append({"resumed_utc": datetime.now(timezone.utc).isoformat(),
            "completed_stages": list(record["completed_stages"]),
            "max_validation_retries": args.max_validation_retries,
            "retry_transient_http": args.retry_transient_http,
            "retry_uncertain_transport": args.retry_uncertain_transport,
            "retry_legacy_unknown_transport": args.retry_legacy_unknown_transport,
            "retry_legacy_cancelled": args.retry_legacy_cancelled})
        record["status"] = "running"
    write_json(output/"orchestration.json", record)
    deadline = time.monotonic() + max(0., record["deadline_unix"] - time.time())
    try:
        for stage in plan["stages"]:
            if stage["name"] in record["completed_stages"]:
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Optional comparison time cap reached")
            with (output/(stage["name"]+".log")).open("a" if args.resume else "w", encoding="utf-8") as log:
                completed = subprocess.run(stage["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=remaining, check=False)
            if completed.returncode:
                raise RuntimeError(f"Stage {stage['name']} failed; incomplete predictions are not a matched benchmark")
            record["completed_stages"].append(stage["name"])
            write_json(output/"orchestration.json", record)
        record["status"] = "six_split_comparison_complete"
    except BaseException as error:
        record.update(status="incomplete_do_not_claim_matched_comparison", error_type=type(error).__name__)
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output/"orchestration.json", record)
    print(json.dumps({"status": record["status"], "output": str(output)}))


if __name__ == "__main__":
    main()
