"""Prevent stale-model comparisons and omission of matched figure controls."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def figures(tmp_path):
    path = Path(__file__).resolve().parents[1] / "reports/build_figures.py"
    spec = importlib.util.spec_from_file_location("build_research_figures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline, followup = tmp_path / "baseline", tmp_path / "followup"
    costs = {"correct": 1., "wrong": 5., "defer": .2, "expand": .05}
    document = {
        "status": module.COMPLETE,
        "provider_fingerprint": "a" * 64,
        "test_data_hashes": {name: name + "-hash" for name, _ in module.DATASETS},
        "datasets": {name: {"scenarios": {"primary": {
            "costs": costs, "methods": {
                "rl_seed42": {"overall": {"mean_utility": .1}},
                "confidence_gate": {"overall": {"mean_utility": .2}},
                "never_expand": {"overall": {"mean_utility": .3}},
                "exact_seed42": {"overall": {"mean_utility": .4}},
                "full_feedback": {"overall": {"mean_utility": .5}},
                "fixed_cost_seed42": {"overall": {"mean_utility": .6}},
            }}}} for name, _ in module.DATASETS},
    }
    for root in (baseline, followup):
        target = root / "main/results/results.json"
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps(document), encoding="utf-8")
    return module, baseline, followup


def put_summary(followup, fingerprint="a" * 64):
    (followup / "main/training_summary.json").write_text(json.dumps({
        "selected_step": 25, "selected_model_fingerprint": fingerprint,
    }), encoding="utf-8")


def test_completed_results_without_selected_training_provenance_are_rejected(figures):
    module, baseline, followup = figures
    with pytest.raises(ValueError, match="require training_summary"):
        module.utility_data(module.Inputs(), baseline, followup)


@pytest.mark.parametrize("fingerprint", ["b" * 64, None])
def test_same_benchmark_results_from_wrong_or_unidentified_model_are_rejected(figures, fingerprint):
    module, baseline, followup = figures
    put_summary(followup)
    target = followup / "main/results/results.json"
    document = json.loads(target.read_text(encoding="utf-8"))
    document["provider_fingerprint"] = fingerprint
    target.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="provider_fingerprint differs"):
        module.utility_data(module.Inputs(), baseline, followup)


def test_completed_matched_results_keep_original_rows_and_all_v2_controls(figures):
    module, baseline, followup = figures
    put_summary(followup)
    inputs = module.Inputs()
    panels = module.utility_data(inputs, baseline, followup)
    expected = {
        "v1": [("rl_seed42", .1), ("confidence_gate", .2), ("never_expand", .3)],
        "v2": [("rl_seed42", .1), ("exact_seed42", .4), ("full_feedback", .5),
               ("fixed_cost_seed42", .6), ("confidence_gate", .2), ("never_expand", .3)],
    }
    for panel in panels:
        for version in ("v1", "v2"):
            rows = [row for row in panel["rows"] if row.get("provider") == version]
            assert [(row["method"], row["utility"]) for row in rows] == expected[version]
            assert {row["group"] for row in rows} == {version}
            assert len({module.COLORS[row["key"]] for row in rows}) == len(rows)
        assert [row["key"] for row in panel["rows"] if row["group"] == "comparators"] == [
            "qwen", "jev", "direct", "nearest"]
    assert any(path.endswith("training_summary.json") for path in inputs.sources)


def test_incomplete_followup_never_appears_as_zero_or_completed(figures):
    module, baseline, followup = figures
    target = followup / "main/results/results.json"
    document = json.loads(target.read_text(encoding="utf-8"))
    document["status"] = "in_progress"
    target.write_text(json.dumps(document), encoding="utf-8")
    panels = module.utility_data(module.Inputs(), baseline, followup)
    for panel in panels:
        rows = [row for row in panel["rows"] if row.get("provider") == "v2"]
        assert len(rows) == 6
        assert all(row["utility"] is None for row in rows)
