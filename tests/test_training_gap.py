"""Read-only gap audit: real evaluator on tiny CPU views and provenance failures."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from recovery_router import model as model_module
from recovery_router import train as original_trainer


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tokenizer:
    def __call__(self, text, **kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids": [1, 2], "attention_mask": [1, 1]}

    def pad(self, encoded, **kwargs):
        return {key: torch.tensor([r[key] for r in encoded]) for key in ("input_ids", "attention_mask")}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(21).float() / 20)
        self.dropout = torch.nn.Dropout(.5)

    def forward(self, input_ids, attention_mask, candidate_counts, labels=None):
        logits = model_module.mask_logits(self.dropout(self.weight.expand(input_ids.shape[0], -1)), candidate_counts)
        return {"logits": logits}


@pytest.fixture
def audit_fixture(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    audit = load_module("training_gap_audit", root / "scripts/audit_training_gap.py")
    frozen = root / "src/recovery_router/train_extended.py"
    extended = load_module("recovery_router.train_extended", frozen)
    model, tokenizer = TinyModel(), Tokenizer()
    train, validation, config, summary, bundle = (tmp_path / name for name in
                                                ("train.jsonl", "tuning.jsonl", "config.json", "training_summary.json", "model"))
    candidates = [{"id": "intent0", "label": "intent0"}, {"id": "intent1", "label": "intent1"}]
    for path, split, count in ((train, "train", 3), (validation, "tuning", 2)):
        rows = [{"id": f"{split}{i}", "split": split, "text": f"{split} example {i}", "label": "intent0",
                 "is_oos": i == 0, "candidates": candidates[:1], "expanded_candidates": candidates} for i in range(count)]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    configuration = {"model": {"max_length": 16}, "policy": {"target_oos_prior": 2 / 11},
                     "training": {"micro_batch_size": 2}}
    config.write_text(json.dumps(configuration))
    val_rows = extended.read_records(validation)
    val_views = audit.fixed_views(validation, val_rows, tokenizer, 16, original_trainer)
    val_metrics = extended.evaluate_views(model, tokenizer, val_views, 16, 2, "cpu", 2 / 11)
    metadata = {"step": 1, "checkpoint_selection_eligible": True, "model": configuration["model"],
                "max_length": 16, "trained_candidate_counts": [1, 2], "selection_target_oos_prior": 2 / 11,
                "training_data_sha256": audit.sha256(train), "validation_data_sha256": audit.sha256(validation),
                "training_config_sha256": audit.sha256(config), "validation_views_sha256": audit.view_sha256(val_views)}
    bundle.mkdir()
    (bundle / "metadata.json").write_text(json.dumps(metadata))
    (bundle / "training_report.json").write_text(json.dumps({"stop_reason": "validation_selected", "optimizer_steps_total": 1}))
    for folder in ("adapter", "tokenizer"):
        (bundle / folder).mkdir()
        (bundle / folder / "fixture.json").write_text("{}")
    (bundle / "head.safetensors").write_bytes(b"structural fixture only")
    summary.write_text(json.dumps({**metadata, "selected_step": 1,
                                  "selected_model_fingerprint": model_module.bundle_fingerprint(bundle),
                                  "selected_validation_mean_nll": val_metrics["selection_mean_nll"]}))
    monkeypatch.setattr(model_module, "load_model_bundle", lambda *a, **kw: (model, tokenizer, metadata))
    monkeypatch.setattr(audit, "_runtime", lambda: (torch, model_module, original_trainer, extended))
    args = SimpleNamespace(model=bundle, summary=summary, train=train, validation=validation, config=config,
                           output=tmp_path / "gap.json", device="cpu", batch_size=3, max_seconds=30)
    return audit, args, model, extended


def test_gap_audit_scores_full_fixed_splits_without_mutation(audit_fixture):
    audit, args, model, _ = audit_fixture
    before = model.weight.detach().clone()
    report = audit.run_audit(args)
    assert report["status"] == "complete"
    assert report["splits"]["train"]["views"] == 6 and report["splits"]["tuning"]["views"] == 4
    assert report["model_and_inputs_unchanged"] and report["parameter_versions_unchanged"]
    assert torch.equal(before, model.weight) and model.weight.grad is None and not model.weight.requires_grad
    assert report["optimizer_updates"] == 0 and report["checkpoint_selection_changed"] is False
    assert report["validation_repeat"]["difference"] == pytest.approx(0., abs=1e-7)
    # Identical per-group conditional distributions: OOS-standardized gap is zero,
    # despite different raw OOS mixture proportions in train and tuning.
    assert report["validation_minus_training"]["selection_mean_nll"] == pytest.approx(0., abs=1e-7)
    assert report["validation_minus_training"]["overall"]["nll"] != pytest.approx(0., abs=1e-7)
    saved = args.output.read_bytes()
    with pytest.raises(FileExistsError):
        audit.run_audit(args)
    assert args.output.read_bytes() == saved


def test_gap_audit_rejects_changed_input_before_output(audit_fixture):
    audit, args, _, _ = audit_fixture
    args.train.write_text(args.train.read_text() + "\n")
    with pytest.raises(ValueError, match="training_data_sha256"):
        audit.run_audit(args)
    assert not args.output.exists()


def test_gap_audit_failure_stays_incomplete_with_integrity_record(audit_fixture, monkeypatch):
    audit, args, _, extended = audit_fixture
    def timeout(*a, **kw):
        raise TimeoutError("deliberate test deadline")
    monkeypatch.setattr(extended, "evaluate_views", timeout)
    with pytest.raises(TimeoutError):
        audit.run_audit(args)
    report = json.loads(args.output.read_text())
    assert report["status"] == "incomplete" and report["error_type"] == "TimeoutError"
    assert report["model_and_inputs_unchanged"] and report["parameter_versions_unchanged"]
    assert "validation_minus_training" not in report
