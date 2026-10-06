"""Offline tests: true sequential actions and honest local artifact staging."""
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from recovery_router.inference import (ModelForecaster, candidate_schedule, load_retriever,
                                       route_text, validate_trained_counts)
from recovery_router.model import DecisionModel, bundle_fingerprint
from recovery_router.io import sha256_file
from recovery_router.policy import COMMIT, DEFER, EXPAND


CALIBRATION = {"first_temperature": 1., "expanded_temperature": 1.}


class Retriever:
    def __init__(self, events):
        self.events = events

    def retrieve(self, text, k):
        self.events.append(("retrieve", k))
        return [{"id": f"intent:{i}", "label": f"label_{i}", "score": 1. / (i+1)} for i in range(k)]


class Actor:
    def __init__(self, decisions, events):
        self.decisions, self.events = list(decisions), events

    def choose(self, features):
        self.events.append(("actor", int(features[0, 9])))
        return np.asarray([self.decisions.pop(0)])


def forecast(events):
    def run(row, expanded):
        assert "label" not in row and "is_oos" not in row
        if not expanded:
            assert "expanded_candidates" not in row
        events.append(("forecast", expanded))
        n = len(row["expanded_candidates" if expanded else "candidates"])
        return {"probs": [.6 / n] * n + [.4], "model_ms": 1., "tokenize_and_transfer_ms": .2}
    return run


@pytest.mark.parametrize("initial_k,expanded_k", [(1, 5), (5, 20)])
def test_commit_never_retrieves_or_forecasts_expanded_state(initial_k, expanded_k):
    events = []
    result = route_text("An unlabeled request", Retriever(events), forecast(events),
                        Actor([COMMIT], events), CALIBRATION, initial_k=initial_k, expanded_k=expanded_k)
    assert events == [("retrieve", initial_k), ("forecast", False), ("actor", 0)]
    assert result["action"] == "commit" and result["expanded"] is False
    assert result["confidence"] == pytest.approx(.6 / initial_k)  # never divide out NONE
    assert result["stage_traces"][0]["none_probability"] == pytest.approx(.4)
    assert result["candidate_id"] is not None
    assert result["confidence_event"]["candidate_id"] == result["candidate_id"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("initial_k,expanded_k", [(1, 5), (5, 20)])
def test_expand_is_observed_only_after_the_actor_chooses_it(initial_k, expanded_k):
    events = []
    result = route_text("Another request", Retriever(events), forecast(events),
                        Actor([EXPAND, COMMIT], events), CALIBRATION, initial_k=initial_k, expanded_k=expanded_k)
    assert events == [("retrieve", initial_k), ("forecast", False), ("actor", 0),
                      ("retrieve", expanded_k), ("forecast", True), ("actor", 1)]
    assert result["expanded"] is True
    assert len(result["stage_traces"]) == 2
    assert len(result["stage_traces"][1]["candidates"]) == expanded_k
    assert result["confidence"] == pytest.approx(.6 / expanded_k)


def test_schedule_defaults_legacy_manifest_and_training_support():
    assert candidate_schedule({}) == (1, 5)
    assert candidate_schedule({"first_k": 5, "expanded_k": 20}) == (5, 20)
    validate_trained_counts({"trained_candidate_counts": [1, 5]}, 1, 5)
    with pytest.raises(ValueError, match="trained_candidate_counts"):
        validate_trained_counts({"trained_candidate_counts": [1, 5]}, 5, 20)
    with pytest.raises(ValueError, match="trained_candidate_counts"):
        validate_trained_counts({}, 1, 5)
    with pytest.raises(ValueError, match="Candidate schedule"):
        candidate_schedule({"initial_k": 1, "expanded_k": 21})


def test_handoff_has_no_automatic_label_and_does_not_expand():
    events = []
    result = route_text("Unknown request", Retriever(events), forecast(events),
                        Actor([DEFER], events), CALIBRATION)
    assert result["action"] == "handoff"
    assert result["candidate_id"] is None and result["label"] is None
    assert len(events) == 3


def test_invalid_second_expansion_fails_explicitly():
    events = []
    with pytest.raises(ValueError, match="second expansion"):
        route_text("Request", Retriever(events), forecast(events), Actor([EXPAND, EXPAND], events), CALIBRATION)


class TinyTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    def pad(self, records, **kwargs):
        return {name: torch.tensor([r[name] for r in records]) for name in ("input_ids", "attention_mask")}


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embedding = torch.nn.Embedding(8, 4)
        self.calls = 0

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_real_forecaster_makes_one_forward_and_preserves_none():
    events = []
    backbone = TinyBackbone()
    model = DecisionModel(backbone)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.bias[20] = math.log(10)
    result = route_text("No labels supplied", Retriever(events), ModelForecaster(model, TinyTokenizer()),
                        Actor([COMMIT], events), CALIBRATION)
    assert backbone.calls == 1
    assert result["confidence"] == pytest.approx(1/11, abs=1e-6)
    assert result["stage_traces"][0]["none_probability"] == pytest.approx(10/11, abs=1e-6)
    assert result["stage_traces"][0]["model_ms"] >= 0
    assert result["latency_ms"] >= result["stage_traces"][0]["model_ms"]


@pytest.fixture
def exporter():
    path = Path(__file__).resolve().parents[1] / "scripts/export_hf.py"
    spec = importlib.util.spec_from_file_location("local_export_hf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, dict) else value, encoding="utf-8")


@pytest.fixture
def export_inputs(tmp_path):
    """Structural dummy files; no model-performance assertion is made here."""
    root, model, policy = tmp_path / "project", tmp_path / "model", tmp_path / "policy"
    put(model / "metadata.json", {"step": 3, "trained_candidate_counts": [1, 5],
                                  "model": {"name": "Qwen/Qwen3-1.7B", "revision": "a" * 40}})
    put(model / "training_report.json", {"stop_reason": "epochs_complete", "optimizer_steps_total": 3})
    for relative in ("head.safetensors", "adapter/adapter_model.safetensors"):
        put(model / relative, "STRUCTURAL TEST ONLY")
    put(model / "adapter/adapter_config.json", {"r": 2})
    put(model / "adapter/README.md", "Adapter model card included in the model identity.")
    put(model / "tokenizer/tokenizer_config.json", {"tokenizer_class": "dummy"})
    put(model / "tokenizer/tokenizer.json", {"test": True})
    put(model / "tokenizer/added_tokens.json", {"<fixture>": 8})
    put(model / "tokenizer/special_tokens_map.json", {"eos_token": "<fixture>"})
    put(model / "tokenizer/vocab.json", {"fixture": 0})
    put(model / "tokenizer/merges.txt", "#version: 0.2\nf ixture\n")
    put(model / "tokenizer/nested/vocabulary.bin", "Nested tokenizer identity file")
    put(model / "tokenizer/nested/empty.txt", "")
    put(policy / "policy.safetensors", "STRUCTURAL TEST ONLY")
    put(root / "calibration.json", CALIBRATION)
    put(policy / "policy.json", {"training_rows": 2, "logs": [{"step": 1}], "method": "unit_test_only",
                                 "provider_fingerprint": bundle_fingerprint(model),
                                 "calibration_sha256": sha256_file(root / "calibration.json")})
    put(root / "MODEL_CARD.md", "# planned artifact, NOT TRAINED\n**Status: NOT TRAINED / NOT EVALUATED.**\nMetric: NOT MEASURED\n")
    put(root / "pyproject.toml", "[project]\nname = 'structural-test'\n")
    put(root / "LICENSE", "Dummy test license; not a release.")
    put(root / "src/recovery_router/inference.py", "# structural fixture")
    for name in ("manifest.json", "catalogs.json"):
        put(root / "data/processed" / name, {"fixture": True})
    for name in ("index.jsonl", "banking_index.jsonl"):
        put(root / "data/processed" / name, '{"id":"index-only"}\n')
    for name in ("CLINC_LICENSE.txt", "BANKING77_LICENSE.txt"):
        put(root / "data/downloads" / name, "Fixture license")
    return {"model": model, "policy": policy, "calibration": root / "calibration.json",
            "output": tmp_path / "export", "project_root": root}


def test_export_refuses_missing_artifacts_without_creating_output(exporter, tmp_path):
    with pytest.raises(ValueError, match="artifact missing"):
        exporter.export_bundle(tmp_path / "missing", tmp_path / "policy", tmp_path / "cal.json", tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_export_refuses_smoke_checkpoint(exporter, export_inputs):
    put(export_inputs["model"] / "training_report.json", {"stop_reason": "smoke_steps", "optimizer_steps_total": 3})
    with pytest.raises(ValueError, match="smoke-test"):
        exporter.export_bundle(**export_inputs)
    assert not export_inputs["output"].exists()


def test_export_refuses_untrained_model_and_mismatched_calibration(exporter, export_inputs):
    metadata = exporter._read(export_inputs["model"] / "metadata.json")
    metadata["step"] = 0
    put(export_inputs["model"] / "metadata.json", metadata)
    with pytest.raises(ValueError, match="untrained model"):
        exporter.export_bundle(**export_inputs)
    metadata["step"] = 3
    put(export_inputs["model"] / "metadata.json", metadata)
    meta = exporter._read(export_inputs["policy"] / "policy.json")
    meta["calibration_sha256"] = "0" * 64
    put(export_inputs["policy"] / "policy.json", meta)
    with pytest.raises(ValueError, match="different calibration"):
        exporter.export_bundle(**export_inputs)


def test_export_refuses_untrained_candidate_positions(exporter, export_inputs):
    put(export_inputs["project_root"] / "data/processed/manifest.json",
        {"retrieval": {"first_k": 5, "expanded_k": 20}})
    with pytest.raises(ValueError, match="trained_candidate_counts"):
        exporter.export_bundle(**export_inputs)


def test_export_refuses_wrong_or_unrecorded_model_provider(exporter, export_inputs):
    meta = exporter._read(export_inputs["policy"] / "policy.json")
    meta.pop("provider_fingerprint")
    put(export_inputs["policy"] / "policy.json", meta)
    with pytest.raises(ValueError, match="model provider fingerprint"):
        exporter.export_bundle(**export_inputs)
    meta["provider_fingerprint"] = "f" * 64
    put(export_inputs["policy"] / "policy.json", meta)
    with pytest.raises(ValueError, match="model provider fingerprint"):
        exporter.export_bundle(**export_inputs)
    assert not export_inputs["output"].exists()


def test_export_is_local_checksums_files_and_preserves_pending_metrics(exporter, export_inputs):
    result = exporter.export_bundle(**export_inputs)
    output = export_inputs["output"]
    assert result["publication_status"] == "local staging only; no upload performed"
    assert result["evaluation_status"] == "NOT MEASURED in this export"
    assert (output / "README.md").is_file()
    assert "Metric: NOT MEASURED" in (output / "README.md").read_text()
    assert "TRAINING RECORDED" in (output / "README.md").read_text()
    assert not any("training_state.pt" in name for name in result["files"])
    assert all(exporter._sha(output / name) == record["sha256"] for name, record in result["files"].items())
    with pytest.raises(FileExistsError):
        exporter.export_bundle(**export_inputs)
    exporter.export_bundle(**export_inputs, overwrite=True)
    put(output / "unexpected.txt", "Do not remove this untracked file")
    with pytest.raises(ValueError, match="untracked"):
        exporter.export_bundle(**export_inputs, overwrite=True)
    assert (output / "unexpected.txt").exists()


def test_export_preserves_complete_portable_model_identity(exporter, export_inputs):
    manifest = exporter.export_bundle(**export_inputs)
    model, output = export_inputs["model"], export_inputs["output"]
    assert manifest["model_fingerprint"] == bundle_fingerprint(model) == bundle_fingerprint(output / "model")
    for relative in ("adapter/README.md", "tokenizer/nested/vocabulary.bin", "tokenizer/nested/empty.txt",
                     "tokenizer/added_tokens.json", "tokenizer/special_tokens_map.json", "tokenizer/vocab.json",
                     "tokenizer/merges.txt", "tokenizer/tokenizer_config.json", "tokenizer/tokenizer.json"):
        assert "model/" + relative in manifest["files"]
        assert (output / "model" / relative).read_bytes() == (model / relative).read_bytes()


@pytest.mark.parametrize("mutation", ["omit", "alter"])
def test_export_refuses_copy_that_changes_model_identity(exporter, export_inputs, monkeypatch, mutation):
    copy = exporter.shutil.copy2

    def corrupt_copy(source, destination):
        if Path(destination).as_posix().endswith("/model/adapter/README.md"):
            if mutation == "omit":
                return destination
            copy(source, destination)
            Path(destination).write_text("Altered adapter card", encoding="utf-8")
            return destination
        return copy(source, destination)

    monkeypatch.setattr(exporter.shutil, "copy2", corrupt_copy)
    with pytest.raises(ValueError, match="Exported model fingerprint differs"):
        exporter.export_bundle(**export_inputs)
    assert not (export_inputs["output"] / "export_manifest.json").exists()


@pytest.mark.parametrize("relative", ["adapter/.env", "adapter/credentials.json",
                                      "tokenizer/nested/access_token.txt", "tokenizer/refresh_token.json",
                                      "tokenizer/auth_token.json", "tokenizer/private/vocab.json"])
def test_export_rejects_sensitive_identity_files_before_copying(exporter, export_inputs, relative):
    put(export_inputs["model"] / relative, "SENSITIVE FIXTURE")
    with pytest.raises(ValueError, match="hidden or sensitive filenames"):
        exporter.export_bundle(**export_inputs)
    assert not export_inputs["output"].exists()


@pytest.mark.parametrize("directory", [False, True])
def test_export_rejects_model_identity_links_before_hashing(exporter, export_inputs, monkeypatch, directory):
    target = export_inputs["model"] / ("tokenizer/nested" if directory else "adapter/README.md")
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == target or original(path))
    with pytest.raises(ValueError, match="must not contain links"):
        exporter.export_bundle(**export_inputs)
    assert not export_inputs["output"].exists()


def make_validation_selected_fixture(exporter, inputs):
    model = inputs["model"]
    metadata = exporter._read(model / "metadata.json")
    provenance = {"training_data_sha256": "a" * 64, "validation_data_sha256": "b" * 64,
                  "validation_views_sha256": "c" * 64, "training_config_sha256": "d" * 64,
                  "selection_min_delta": .0001, "selection_target_oos_prior": 2 / 11,
                  "selection_rule": "fixed validation raw NLL", "selection_validation_records": 4,
                  "selection_validation_views": 8}
    metadata.update({**provenance, "checkpoint_selection_eligible": True,
                     "validation_metrics": {"selection_mean_nll": .2}})
    put(model / "metadata.json", metadata)
    fingerprint = bundle_fingerprint(model)
    put(model / "training_report.json", {
        "stop_reason": "validation_selected", "optimizer_steps_total": 3,
        "actual_executed_optimizer_steps": 6, "selected_validation_mean_nll": .2,
        "selected_checkpoint": "checkpoints/step-000003"})
    proof = {**provenance, "stop_reason": "epochs_complete", "selected_step": 3,
             "actual_executed_optimizer_steps": 6, "selected_model_fingerprint": fingerprint,
             "selected_validation_mean_nll": .2, "selected_checkpoint": "checkpoints/step-000003",
             "validation_curve": [
                 {"step": 0, "selection_mean_nll": .1, "selected_at_this_step": False,
                  "model_fingerprint": "0" * 64, "bundle": "checkpoints/step-000000"},
                 {"step": 3, "selection_mean_nll": .2, "selected_at_this_step": True,
                  "model_fingerprint": fingerprint, "bundle": "checkpoints/step-000003"},
                 {"step": 6, "selection_mean_nll": .3, "selected_at_this_step": False,
                  "model_fingerprint": "6" * 64, "bundle": "checkpoints/step-000006"}]}
    put(model.parent / "training_summary.json", proof)
    policy = exporter._read(inputs["policy"] / "policy.json")
    policy["provider_fingerprint"] = fingerprint
    put(inputs["policy"] / "policy.json", policy)
    return proof


def test_export_validation_selected_requires_proof(exporter, export_inputs):
    put(export_inputs["model"] / "training_report.json", {
        "stop_reason": "validation_selected", "optimizer_steps_total": 3})
    with pytest.raises(ValueError, match="training_summary.json"):
        exporter.export_bundle(**export_inputs)
    assert not export_inputs["output"].exists()


@pytest.mark.parametrize("mutation", [
    "selected_step", "fingerprint", "nan_metric", "curve_metric", "curve_better_later",
    "missing_curve", "missing_provenance", "executed_steps", "unsafe_checkpoint",
])
def test_export_validation_selected_rejects_forged_proof(exporter, export_inputs, mutation):
    proof = make_validation_selected_fixture(exporter, export_inputs)
    if mutation == "selected_step":
        proof["selected_step"] = 6
    elif mutation == "fingerprint":
        proof["selected_model_fingerprint"] = "f" * 64
    elif mutation == "nan_metric":
        proof["selected_validation_mean_nll"] = float("nan")
    elif mutation == "curve_metric":
        proof["validation_curve"][1]["selection_mean_nll"] = .4
    elif mutation == "curve_better_later":
        proof["validation_curve"][2]["selection_mean_nll"] = .1
    elif mutation == "missing_curve":
        proof.pop("validation_curve")
    elif mutation == "missing_provenance":
        proof.pop("validation_data_sha256")
    elif mutation == "executed_steps":
        proof["actual_executed_optimizer_steps"] = 2
    elif mutation == "unsafe_checkpoint":
        proof["selected_checkpoint"] = "../outside"
    put(export_inputs["model"].parent / "training_summary.json", proof)
    with pytest.raises(ValueError, match="Validation selection"):
        exporter.export_bundle(**export_inputs)
    assert not export_inputs["output"].exists()


def test_export_validation_selected_rejects_ineligible_model(exporter, export_inputs):
    make_validation_selected_fixture(exporter, export_inputs)
    metadata = exporter._read(export_inputs["model"] / "metadata.json")
    metadata["checkpoint_selection_eligible"] = False
    put(export_inputs["model"] / "metadata.json", metadata)
    with pytest.raises(ValueError, match="eligible selected model"):
        exporter.export_bundle(**export_inputs)
    assert not export_inputs["output"].exists()


def test_export_validation_selected_copies_verified_proof(exporter, export_inputs):
    make_validation_selected_fixture(exporter, export_inputs)
    manifest = exporter.export_bundle(**export_inputs)
    proof_relative = "provenance/training_summary.json"
    assert manifest["training_stop_reason"] == "validation_selected"
    assert manifest["model_training_steps"] == 3
    assert proof_relative in manifest["files"]
    assert exporter._sha(export_inputs["output"] / proof_relative) == exporter._sha(export_inputs["model"].parent / "training_summary.json")
    assert manifest["evaluation_status"] == "NOT MEASURED in this export"


@pytest.mark.parametrize("fingerprint", [None, "incorrect-other-model-fingerprint"])
def test_export_rejects_completed_results_for_unidentified_or_wrong_model(exporter, export_inputs, fingerprint):
    path = export_inputs["project_root"] / "results.json"
    report = {"status": "measured_offline_evaluation_complete"}
    if fingerprint is not None:
        report["provider_fingerprint"] = fingerprint
    put(path, report)
    with pytest.raises(ValueError, match="Completed results.*provider fingerprint"):
        exporter.export_bundle(**export_inputs, results=path)
    assert not export_inputs["output"].exists()


def test_export_accepts_matching_completed_results_without_claiming_independent_validation(exporter, export_inputs):
    path = export_inputs["project_root"] / "results.json"
    put(path, {"status": "measured_offline_evaluation_complete",
               "provider_fingerprint": bundle_fingerprint(export_inputs["model"]), "fixture": True})
    manifest = exporter.export_bundle(**export_inputs, results=path)
    assert manifest["results_verification"] == {
        "status": "complete_provider_fingerprint_matched", "reported_status": "measured_offline_evaluation_complete",
        "provider_fingerprint_matched": True}
    assert manifest["evaluation_status"] == "results supplied but not independently validated"
    assert (export_inputs["output"] / "results.json").read_bytes() == path.read_bytes()


@pytest.mark.parametrize("document", [{"status": "incomplete"}, {"structural_fixture": True}, ["fixture"]])
def test_export_incomplete_or_fixture_results_remain_explicit_staging(exporter, export_inputs, document):
    path = export_inputs["project_root"] / "results.json"
    path.write_text(json.dumps(document))
    manifest = exporter.export_bundle(**export_inputs, results=path)
    assert manifest["results_verification"]["status"] == "incomplete_or_unrecognized_staging_only"
    assert manifest["results_verification"]["provider_fingerprint_matched"] is None
    assert manifest["evaluation_status"] == "results supplied but not independently validated"


def test_export_optional_reports_preserve_document_links_and_exclude_private_or_executable_assets(exporter, export_inputs):
    root = export_inputs["project_root"]
    put(root / "docs/TECHNICAL_NOTE.md", "[Results](../reports/RESULTS.md)\n")
    put(root / "reports/RESULTS.md", "![Curve](figures/curve.svg)\nMeasured fixture evidence only.")
    put(root / "reports/summary.json", {"status": "fixture", "n": 3})
    put(root / "reports/figures/curve.svg", '<svg xmlns="http://www.w3.org/2000/svg"></svg>')
    put(root / "reports/figures/curve.png", "structural image fixture")
    put(root / "reports/figures/CAPTIONS.md", "Fixture caption")
    put(root / "reports/figures/figure_data.json", {"source_sha256": "a" * 64})
    for relative in ("reports/build_figures.py", "reports/.env", "reports/__pycache__/cache.json",
                     "reports/figures/code.py", "reports/figures/nested/unreviewed.json",
                     "reports/secrets.json", "reports/.hidden.json"):
        put(root / relative, "DO NOT EXPORT")
    put(root / "reports/innocent_name.json", {"nested": {"api_key": "SENSITIVE_FIXTURE_DO_NOT_EXPORT"}})
    put(root / "reports/figures/note.md", "Credential fixture: apikey_" + "x" * 30)
    manifest = exporter.export_bundle(**export_inputs, include_reports=True)
    expected = {"reports/RESULTS.md", "reports/summary.json", "reports/figures/curve.svg", "reports/figures/curve.png",
                "reports/figures/CAPTIONS.md", "reports/figures/figure_data.json"}
    actual = {name for name in manifest["files"] if name.startswith("reports/")}
    assert actual == expected
    assert manifest["report_assets_included"] is True
    assert (export_inputs["output"] / "docs/../reports/RESULTS.md").is_file()
    assert (export_inputs["output"] / "reports/figures/curve.svg").is_file()


def test_export_does_not_include_optional_reports_without_flag(exporter, export_inputs):
    put(export_inputs["project_root"] / "reports/RESULTS.md", "Optional evidence")
    manifest = exporter.export_bundle(**export_inputs)
    assert not any(name.startswith("reports/") for name in manifest["files"])
    assert manifest["report_assets_included"] is False
