"""Follow-up orchestration bounds and comparator provenance, without GPU work."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def driver(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("run_extended_test", scripts / "run_extended.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_caps_include_original_outer_deadline(driver):
    config = {"training": {"max_seconds": 3600}}
    assert driver.stage_deadline("v1_validation_learning_curve", 100, 10000, config) == 700
    assert driver.stage_deadline("train_validation_selected_beliefs", 100, 10000, config) == 3700
    assert driver.stage_deadline("train_validation_selected_beliefs", 100, 500, config) == 500
    assert driver.stage_deadline("main_evaluate", 100, 500, config) == 500


def comparator_fixture(driver, tmp_path):
    costs = {"correct": 1, "wrong": 5, "defer": .2, "expand": .05}
    data, source = tmp_path / "data", tmp_path / "baseline/qwen_base"
    data.mkdir()
    (source / "results").mkdir(parents=True)
    hashes = {}
    for base in ("test_base", "test", "transfer"):
        for suffix in ("", "_drop_gold", "_shuffle"):
            name = base + suffix
            path = data / (name + ".jsonl")
            path.write_text(name)
            hashes[name] = driver.file_hash(path)
    result = {"status": "measured_offline_evaluation_complete", "test_data_hashes": hashes,
              "datasets": {name: {"scenarios": {"primary": {"costs": costs.copy()}}} for name in hashes}}
    (source / "results/results.json").write_text(json.dumps(result))
    return source, data, costs, result


def test_reuse_rejects_mismatched_data_and_costs(driver, tmp_path):
    source, data, costs, result = comparator_fixture(driver, tmp_path)
    assert driver.validate_comparator("qwen_base", source, data, costs)["fingerprint"]
    result["datasets"]["test"]["scenarios"]["primary"]["costs"]["wrong"] = 7
    (source / "results/results.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="costs differ"):
        driver.validate_comparator("qwen_base", source, data, costs)
    result["datasets"]["test"]["scenarios"]["primary"]["costs"] = costs
    (source / "results/results.json").write_text(json.dumps(result))
    (data / "test_shuffle.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="dataset differs"):
        driver.validate_comparator("qwen_base", source, data, costs)


def test_classical_requires_all_twelve_evaluations(driver, tmp_path):
    source, data, costs, result = comparator_fixture(driver, tmp_path)
    classical = source.parent / "classical"
    classical.mkdir()
    result = {"elapsed_seconds": 1, "test_data_hashes": result["test_data_hashes"],
              "evaluations": {str(i): {} for i in range(12)},
              "fitting": {"utility": {"correct": 1, "wrong": -5, "handoff": -.2}}}
    (classical / "results.json").write_text(json.dumps(result))
    assert driver.validate_comparator("classical", classical, data, costs)
    result["evaluations"].pop("0")
    (classical / "results.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="incomplete"):
        driver.validate_comparator("classical", classical, data, costs)


def test_extended_evidence_is_enforced_by_existing_seal_verifier(driver, tmp_path, monkeypatch):
    from recovery_router.evaluate import verify_seal
    root, output = tmp_path / "root", tmp_path / "output"
    monkeypatch.setattr(driver, "ROOT", root)
    protocol = root / "docs/EXTENDED_PROTOCOL.md"
    paths = [protocol, root / "scripts/score_checkpoints.py", output / "launch_provenance.json",
             output / "main/training_summary.json", output / "v1_validation_curve.json"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    seal = output / "main/policies/seal.json"
    seal.parent.mkdir()
    seal.write_text(json.dumps({"inputs": []}))
    driver.extend_fit_seal(output, protocol)
    assert len(verify_seal(seal)["inputs"]) == 6
    protocol.write_text("changed")
    with pytest.raises(ValueError, match="followup_protocol"):
        verify_seal(seal)


@pytest.mark.parametrize("failure_mode", ["setup", "stage_cleanup"])
def test_failures_record_incomplete_status_without_masking_original(driver, tmp_path, monkeypatch, failure_mode):
    source, data, costs, _ = comparator_fixture(driver, tmp_path)
    baseline, output = source.parent, tmp_path / "output"
    (baseline / "run_ledger.json").write_text(json.dumps({"status": "complete", "start_unix": driver.time.time()}))
    protocol = tmp_path / "protocol.md"
    protocol.write_text("fixed")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"training": {"max_seconds": 3600}, "policy": {"primary_cost": costs},
        "budget": {"gpu_usd_per_hour": 3.18}, "followup": {"max_followup_seconds": 7200,
        "protocol_file": str(protocol), "protocol_sha256": driver.file_hash(protocol)}}))
    monkeypatch.setattr(driver.os, "chdir", lambda path: None)
    monkeypatch.setattr(driver, "validate_comparator", lambda *args: {"path": str(source), "fingerprint": {}})
    monkeypatch.setattr(Path, "symlink_to", lambda *args, **kwargs: None)
    monkeypatch.setattr(driver, "build_stages", lambda *args: [])
    monkeypatch.setattr(driver, "run_signature", lambda *args: {})
    monkeypatch.setattr(driver.subprocess, "check_output", lambda *args, **kwargs: "test-commit")
    monkeypatch.setattr(driver.signal, "signal", lambda *args: None)
    def fail():
        raise OSError("cannot enable subreaper")
    monkeypatch.setattr(driver, "enable_subreaper", fail if failure_mode == "setup" else lambda: None)
    if failure_mode == "stage_cleanup":
        monkeypatch.setattr(driver, "build_stages", lambda *args: [("fixture_stage", ["fixture_command"])])
        class Child:
            pid = 123
            returncode = 2
            def poll(self):
                return self.returncode
        class Tracker:
            def __init__(self, pid):
                pass
            def refresh(self):
                return {}, {}
            def stop(self, child):
                raise OSError("cleanup failed")
        monkeypatch.setattr(driver.subprocess, "Popen", lambda *args, **kwargs: Child())
        monkeypatch.setattr(driver, "Descendants", Tracker)
    monkeypatch.setattr(sys, "argv", ["run_extended.py", "--config", str(config), "--data-dir", str(data),
                                    "--baseline-run", str(baseline), "--output", str(output)])
    expected = OSError if failure_mode == "setup" else RuntimeError
    with pytest.raises(expected, match="cannot enable" if failure_mode == "setup" else "fixture_stage failed"):
        driver.main()
    state = json.loads((output / "run_ledger.json").read_text())
    assert state["status"] == "incomplete" and state["error_type"] == expected.__name__
    if failure_mode == "stage_cleanup":
        assert state["cleanup"]["status"] == "incomplete"
        assert state["cleanup"]["error"] == "cleanup failed"
