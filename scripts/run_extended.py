"""Run the declared exploratory follow-up without altering the original experiment."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from cloud_supervisor import Descendants, enable_subreaper
from run_experiment import build_stages, fingerprint, run_signature, write

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    return next(iter(fingerprint(path).values()))


def validate_comparator(provider, source, data, costs):
    """Check compatibility before launching training; never call a provider."""
    source, data = Path(source), Path(data)
    path = source / ("results.json" if provider == "classical" else "results/results.json")
    result = json.loads(path.read_text())
    if provider == "classical":
        if "elapsed_seconds" not in result or len(result.get("evaluations", {})) != 12:
            raise ValueError("Original classical comparator is incomplete")
        utility = result.get("fitting", {}).get("utility", {})
        if (utility.get("correct"), utility.get("wrong"), utility.get("handoff")) != (
                costs["correct"], -costs["wrong"], -costs["defer"]):
            raise ValueError("Classical comparator utility costs differ")
        artifact = source
    else:
        if result.get("status") != "measured_offline_evaluation_complete":
            raise ValueError(f"Original {provider} comparator is incomplete")
        for name, record in result.get("datasets", {}).items():
            if record.get("scenarios", {}).get("primary", {}).get("costs") != costs:
                raise ValueError(f"Comparator {provider} utility costs differ: {name}")
        artifact = source / "results"
    hashes = result.get("test_data_hashes", {})
    required = {"test_base", "test", "transfer"}
    if provider != "classical":
        required = {name + suffix for name in required for suffix in ("", "_drop_gold", "_shuffle")}
        if not required <= result.get("datasets", {}).keys():
            raise ValueError(f"Comparator {provider} is missing evaluations")
    for name in required:
        if hashes.get(name) != file_hash(data / (name + ".jsonl")):
            raise ValueError(f"Comparator {provider} dataset differs: {name}")
    return {"path": str(artifact.resolve()), "fingerprint": fingerprint(artifact)}


def stage_deadline(name, started, deadline, config):
    cap = {"v1_validation_learning_curve": 600,
           "train_validation_selected_beliefs": config["training"]["max_seconds"]}.get(name)
    return min(deadline, started + cap) if cap is not None else deadline


def extend_fit_seal(output, protocol):
    """Add immutable follow-up evidence using the existing verifier's input schema."""
    seal_path = output / "main/policies/seal.json"
    seal = json.loads(seal_path.read_text())
    evidence = [("followup_protocol", protocol), ("followup_driver", Path(__file__)),
                ("followup_scorer", ROOT / "scripts/score_checkpoints.py"),
                ("followup_launch", output / "launch_provenance.json"),
                ("followup_selection", output / "main/training_summary.json"),
                ("v1_validation_diagnostic", output / "v1_validation_curve.json")]
    seal["inputs"].extend({"role": role, "path": str(path.resolve()), "sha256": file_hash(path)}
                          for role, path in evidence)
    write(seal_path, seal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/a100-extended.json")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    config = json.loads(Path(args.config).read_text())
    for cap in (config["training"]["max_seconds"], config["followup"]["max_followup_seconds"]):
        if not math.isfinite(cap) or cap <= 0:
            raise ValueError("Training and follow-up time caps must be finite and positive")
    protocol = ROOT / config["followup"]["protocol_file"]
    if hashlib.sha256(protocol.read_bytes()).hexdigest() != config["followup"]["protocol_sha256"]:
        raise ValueError("Follow-up protocol changed after configuration was fixed")
    baseline = Path(args.baseline_run).resolve()
    old_ledger = json.loads((baseline / "run_ledger.json").read_text())
    if old_ledger["status"] != "complete":
        raise ValueError("Wait for the original run, including its latency benchmark, to finish")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a fresh follow-up output directory")
    reused = {}
    for provider in ("qwen_base", "classical"):
        source = baseline / provider
        reused[provider] = validate_comparator(provider, source, args.data_dir, config["policy"]["primary_cost"])
    # Jev is optional and is linked only if its existing natural and stress results are complete.
    jev_result = baseline / "jev/results/results.json"
    if jev_result.exists() and json.loads(jev_result.read_text()).get("status") == "measured_offline_evaluation_complete":
        reused["jev"] = validate_comparator("jev", baseline / "jev", args.data_dir, config["policy"]["primary_cost"])
    output.mkdir(parents=True, exist_ok=True)
    for provider in reused:
        (output / provider).symlink_to(baseline / provider, target_is_directory=True)
    stage_args = SimpleNamespace(config=args.config, data_dir=args.data_dir, output=str(output),
                                 skip_base=True, stress=True, batch_size=16)
    stages = []
    for name, command in build_stages(stage_args, config):
        if name in ("training_smoke", "classical_benchmarks"):
            continue
        if name == "train_beliefs":
            stages.append(("v1_validation_learning_curve", [sys.executable, "scripts/score_checkpoints.py",
                "--model-root", str(baseline / "main/model"), "--validation", str(Path(args.data_dir) / "tuning.jsonl"),
                "--config", args.config, "--output", str(output / "v1_validation_curve.json"),
                "--device", "cuda", "--max-seconds", "600"]))
            command = [sys.executable, "-m", "recovery_router.train_extended", "--config", args.config,
                "--data", str(Path(args.data_dir) / "train.jsonl"), "--validation", str(Path(args.data_dir) / "tuning.jsonl"),
                "--output", str(output / "main"), "--max-seconds", str(config["training"]["max_seconds"])]
            name = "train_validation_selected_beliefs"
        stages.append((name, command))
    started = time.time()
    cloud_start = old_ledger["start_unix"]
    absolute_deadline = cloud_start + 9 * 3600
    deadline = min(absolute_deadline, started + config["followup"]["max_followup_seconds"])
    state = dict(status="running", classification="exploratory_after_v1_test_inspection",
        start_unix=cloud_start, followup_start_unix=started, deadline_unix=deadline,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        protocol_sha256=config["followup"]["protocol_sha256"], signature=run_signature(stage_args, stages),
        reused_baseline_run=str(baseline), baseline_ledger_sha256=hashlib.sha256((baseline / "run_ledger.json").read_bytes()).hexdigest(),
        reused_comparators=reused, cleanup_reserve_seconds=70, stages=[])
    ledger = output / "run_ledger.json"
    write(ledger, state)
    write(output / "launch_provenance.json", state)
    # Leave time for bounded descendant cleanup inside both declared outer ceilings.
    execution_deadline = deadline - state["cleanup_reserve_seconds"]
    child = descendants = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        enable_subreaper()
        for name, command in stages:
            if name == "train_validation_selected_beliefs":
                cap = config["training"]["max_seconds"]
                if time.time() + cap + 2 * 3600 > absolute_deadline or time.time() + cap + 1800 > execution_deadline:
                    raise TimeoutError("Insufficient time for the declared training cap and evaluation reserve")
            if name in ("train_validation_selected_beliefs", "main_fit_and_seal_policies", "aggregate_comparisons"):
                if run_signature(stage_args, stages) != state["signature"] or file_hash(protocol) != state["protocol_sha256"]:
                    raise ValueError("Follow-up source, configuration, protocol or data changed after launch")
            if name == "aggregate_comparisons":
                for provider, record in reused.items():
                    if fingerprint(record["path"]) != record["fingerprint"]:
                        raise ValueError(f"Reused comparator changed after launch: {provider}")
            if time.time() >= execution_deadline:
                raise TimeoutError("Follow-up time cap reached")
            log_path = output / (name + ".log")
            entry = dict(name=name, command=command, start_unix=time.time(), log=str(log_path))
            entry["deadline_unix"] = stage_deadline(name, entry["start_unix"], execution_deadline, config)
            state["stages"].append(entry)
            write(ledger, state)
            print(f"Starting {name}; log={log_path}", flush=True)
            with log_path.open("w") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                descendants = Descendants(child.pid)
                while child.poll() is None:
                    descendants.refresh()
                    if time.time() >= entry["deadline_unix"]:
                        raise TimeoutError(f"Stage {name} time cap reached")
                    time.sleep(1)
                _, live = descendants.refresh()
                if live:
                    raise RuntimeError("Stage exited with live descendants")
            entry.update(exit_code=child.returncode, seconds=time.time()-entry["start_unix"])
            if child.returncode:
                raise RuntimeError(f"Stage {name} failed; see {log_path}")
            if name == "main_fit_and_seal_policies":
                extend_fit_seal(output, protocol)
            if "--output" in command:
                target = command[command.index("--output") + 1]
                if Path(target).exists():
                    entry.update(output_path=target, output_fingerprint=fingerprint(target))
            write(ledger, state)
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="incomplete", error_type=type(error).__name__, error=str(error))
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, signal.SIG_IGN)
        if descendants is not None:
            try:
                state["cleanup"] = descendants.stop(child)
            except Exception as cleanup_error:
                state["cleanup"] = dict(status="incomplete", error_type=type(cleanup_error).__name__,
                                        error=str(cleanup_error))
        raise
    finally:
        state.update(end_unix=time.time(), followup_elapsed_seconds=time.time()-started,
            total_cloud_elapsed_hours=(time.time()-cloud_start)/3600,
            gpu_rate_assumption_usd=config["budget"]["gpu_usd_per_hour"],
            billing_caveat="Notebook start is an observable proxy; stopping processes does not stop provider billing.")
        write(ledger, state)


if __name__ == "__main__":
    main()
