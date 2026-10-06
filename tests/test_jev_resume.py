"""Recovery keeps paid accounting, original deadlines and verified completed stages."""
import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def recovery(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("jev_resume_runner", root / "scripts/run_jev_comparison.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    data = tmp_path / "data"
    data.mkdir()
    for split in script.FIT + script.TEST:
        row = {"id": split, "text": "Query " + split,
               "candidates": [{"id": "a"}], "expanded_candidates": [{"id": "a"}, {"id": "b"}]}
        script.write_json(data / (split + ".jsonl"), row)
    # JSONL records must be single-line objects.
    for path in data.iterdir():
        path.write_text(json.dumps(json.loads(path.read_text())) + "\n")
    config = tmp_path / "config.json"
    script.write_json(config, {"budget": {"max_usd": 100., "max_elapsed_hours": 12., "gpu_usd_per_hour": 0.}})
    args = SimpleNamespace(run=str(tmp_path / "run"), data_dir=str(data), config=str(config),
        model="jev-1.13.0", max_usd=1., max_requests=12, max_seconds=60, prior_spend_usd=0.,
        input_price_per_million=.042, workers=1, requests_per_second=0., max_retries=0,
        max_validation_retries=0, retry_legacy_cancelled=False, resume=False)
    plan = script.prepare_plan(args)
    output = Path(args.run) / "jev"
    output.mkdir(parents=True)
    ledger = Path(plan["shared_api_ledger"])
    ledger.write_text('{"reserved_spend_must_survive": 0.5}\n')
    legacy = copy.deepcopy(plan)
    for stage in legacy["stages"]:
        command = stage["command"]
        if "--max-validation-retries" in command:
            offset = command.index("--max-validation-retries")
            del command[offset:offset + 2]
    legacy.pop("max_validation_retries")
    legacy.pop("retry_legacy_cancelled")
    legacy.update(started_utc=datetime.fromtimestamp(script.time.time() - 20, timezone.utc).isoformat(),
                  status="incomplete_do_not_claim_matched_comparison", completed_stages=[])
    script.write_json(output / "orchestration.json", legacy)
    return script, args, plan, legacy, output


def save_prediction(script, args, output, split):
    provider = f"api-version:{args.model}:choice-protocol-v1"
    data = Path(args.data_dir) / (split + ".jsonl")
    row = script.read_jsonl(data)[0]
    pred = {"id": row["id"], "model_fingerprint": provider,
            "first_probs": [.8, .2], "expanded_probs": [.7, .2, .1],
            "first_candidate_ids": ["a"], "expanded_candidate_ids": ["a", "b"]}
    path = output / "predictions" / (split + ".jsonl")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(pred) + "\n")
    script.write_json(path.with_suffix(".jsonl.meta.json"),
                      {"model_fingerprint": provider, "data_sha256": script.sha256_file(data)})
    return path


def command_line(args):
    return ["--run", args.run, "--data-dir", args.data_dir, "--config", args.config,
            "--max-usd", str(args.max_usd), "--max-requests", str(args.max_requests),
            "--max-seconds", str(args.max_seconds), "--resume", "--execute-paid-api",
            "--max-validation-retries", "2", "--retry-legacy-cancelled", "--retry-transient-http",
            "--retry-uncertain-transport", "--retry-legacy-unknown-transport"]


def test_legacy_resume_skips_verified_stage_and_preserves_caps_ledger_and_deadline(recovery, monkeypatch):
    script, args, plan, legacy, output = recovery
    prediction = save_prediction(script, args, output, "calibration")
    legacy["completed_stages"] = ["jev_predict_calibration"]
    script.write_json(output / "orchestration.json", legacy)
    ledger_before = Path(plan["shared_api_ledger"]).read_bytes()
    prediction_before = prediction.read_bytes()
    calls = []
    monkeypatch.setenv("TYPESAFE_API_KEY", "unit-test-never-sent")
    def run(command, **kwargs):
        calls.append((command, kwargs["timeout"]))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(script, "subprocess", SimpleNamespace(run=run, STDOUT=script.subprocess.STDOUT))
    script.main(command_line(args))
    result = script.read_json(output / "orchestration.json")
    assert len(calls) == 8
    assert all("calibration.jsonl" not in " ".join(command) for command, _ in calls)
    assert all(0 < timeout < 42 for _, timeout in calls)
    assert result["started_utc"] == legacy["started_utc"]
    assert result["deadline_unix"] == script.original_deadline(legacy)
    assert result["max_seconds"] == 60 and result["api_cap_usd"] == 1.
    assert Path(plan["shared_api_ledger"]).read_bytes() == ledger_before
    assert prediction.read_bytes() == prediction_before
    assert result["resume_history"][0]["max_validation_retries"] == 2
    assert result["resume_history"][0]["retry_legacy_cancelled"] is True
    assert result["resume_history"][0]["retry_transient_http"] is True
    assert result["resume_history"][0]["retry_uncertain_transport"] is True
    assert result["resume_history"][0]["retry_legacy_unknown_transport"] is True
    assert all("--retry-transient-http" in command for command, _ in calls
               if "recovery_router.jev" in command)
    assert result["stages"] == legacy["stages"]


@pytest.mark.parametrize("field,value", [("model", "jev-1.13.1"), ("api_cap_usd", 2.),
    ("request_cap", 13), ("max_seconds", 61), ("workers", 2), ("requests_per_second", 4.),
    ("max_retries", 1), ("config_sha256", "changed"), ("splits", {}),
    ("price_assumption_per_million", .1), ("shared_api_ledger", "different")])
def test_resume_rejects_changed_original_inputs_and_caps(recovery, field, value):
    script, args, plan, legacy, output = recovery
    changed = copy.deepcopy(plan)
    changed[field] = value
    with pytest.raises(ValueError, match="Resume refuses changed"):
        script.load_resume(args, changed)


def test_resume_rejects_changed_stage_commands_and_missing_ledger(recovery):
    script, args, plan, legacy, output = recovery
    changed = copy.deepcopy(plan)
    changed["stages"][0]["command"].append("--unexpected")
    with pytest.raises(ValueError, match="stage commands"):
        script.load_resume(args, changed)
    Path(plan["shared_api_ledger"]).unlink()
    with pytest.raises(ValueError, match="API ledger is missing"):
        script.load_resume(args, plan)


def test_resume_accepts_relocated_aggregate_only_with_identical_content(recovery, tmp_path):
    script, args, plan, legacy, output = recovery
    original = tmp_path / "original" / "aggregate_results.py"
    relocated = tmp_path / "isolated" / "aggregate_results.py"
    for path in (original, relocated):
        path.parent.mkdir()
        path.write_text("# unchanged aggregation\n")
    legacy["stages"][-1]["command"][1] = str(original)
    plan["stages"][-1]["command"][1] = str(relocated)
    script.write_json(output / "orchestration.json", legacy)
    script.load_resume(args, plan)
    relocated.write_text("# altered aggregation\n")
    with pytest.raises(ValueError, match="stage commands"):
        script.load_resume(args, plan)


def test_resume_revalidates_completed_prediction_content(recovery):
    script, args, plan, legacy, output = recovery
    path = save_prediction(script, args, output, "calibration")
    legacy["completed_stages"] = ["jev_predict_calibration"]
    script.write_json(output / "orchestration.json", legacy)
    script.load_resume(args, plan)
    prediction = json.loads(path.read_text())
    prediction["expanded_candidate_ids"].reverse()
    path.write_text(json.dumps(prediction) + "\n")
    with pytest.raises(ValueError, match="Candidate order"):
        script.load_resume(args, plan)


def test_resume_verifies_completed_fit_seal(recovery):
    script, args, plan, legacy, output = recovery
    for split in script.FIT:
        save_prediction(script, args, output, split)
    artifact = output / "policies" / "calibration.json"
    script.write_json(artifact, {"temperature": 1})
    script.write_json(artifact.parent / "seal.json", {"provider_fingerprint": "api-version:jev-1.13.0:choice-protocol-v1",
        "inputs": [{"role": "calibration", "path": str(artifact), "sha256": script.sha256_file(artifact)}]})
    legacy["completed_stages"] = [stage["name"] for stage in plan["stages"][:4]]
    script.write_json(output / "orchestration.json", legacy)
    script.load_resume(args, plan)
    script.write_json(artifact, {"temperature": 2})
    with pytest.raises(ValueError, match="Sealed artifact changed"):
        script.load_resume(args, plan)


def test_expired_resume_never_resets_deadline_or_runs_child(recovery, monkeypatch):
    script, args, plan, legacy, output = recovery
    legacy["started_utc"] = datetime.fromtimestamp(script.time.time() - 120, timezone.utc).isoformat()
    script.write_json(output / "orchestration.json", legacy)
    before = (output / "orchestration.json").read_bytes()
    monkeypatch.setenv("TYPESAFE_API_KEY", "unit-test-never-sent")
    monkeypatch.setattr(script.subprocess, "run", lambda *a, **kw: pytest.fail("Expired resume launched a child"))
    with pytest.raises(TimeoutError, match="cannot reset"):
        script.main(command_line(args))
    assert (output / "orchestration.json").read_bytes() == before


def test_resume_budget_reserves_only_original_remaining_runtime(recovery):
    script, args, plan, legacy, output = recovery
    legacy["started_utc"] = datetime.fromtimestamp(script.time.time() - 50, timezone.utc).isoformat()
    script.write_json(output / "orchestration.json", legacy)
    config = script.read_json(args.config)
    config["budget"]["max_elapsed_hours"] = 20 / 3600
    script.write_json(args.config, config)
    args.resume = True
    # Twenty overall seconds remain: original 60-second cap would fail if reset.
    assert script.prepare_plan(args)["max_seconds"] == 60
