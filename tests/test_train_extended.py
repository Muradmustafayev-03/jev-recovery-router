"""Focused CPU checks for validation selection and training bookkeeping."""
import copy
import json
from pathlib import Path
import random

import pytest
import torch
from torch import nn

from recovery_router.model import NONE_INDEX, mask_logits
from recovery_router.train_extended import (
    check_disjoint, evaluate_views, is_better_checkpoint, learning_rate_factor,
    summarize_observations, validation_mode,
)


class TinyTokenizer:
    def __call__(self, text, add_special_tokens=True, truncation=False):
        assert truncation is False
        return {"input_ids": [1], "attention_mask": [1]}

    def pad(self, encoded, padding=True, return_tensors="pt"):
        maximum = max(len(v["input_ids"]) for v in encoded)
        return {"input_ids": torch.tensor([v["input_ids"] + [0] * (maximum - len(v["input_ids"])) for v in encoded]),
                "attention_mask": torch.tensor([v["attention_mask"] + [0] * (maximum - len(v["attention_mask"])) for v in encoded])}


class TinyDecision(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(8, 21)
        self.dropout = nn.Dropout(.5)

    def forward(self, input_ids, attention_mask, candidate_counts, labels=None):
        logits = mask_logits(self.dropout(self.emb(input_ids[:, 0])), candidate_counts)
        return {"logits": logits, "loss": nn.functional.cross_entropy(logits, labels)}


def views():
    return [{"id": f"row-{i}", "stage": "first" if i < 3 else "expanded",
             "input_ids": [i + 1], "attention_mask": [1], "candidate_count": 1 if i < 3 else 5,
             "target": NONE_INDEX if i in (0, 3) else 0, "is_oos": i in (0, 3)} for i in range(6)]


def test_weighted_metrics_are_example_weighted_and_selection_is_stage_macro():
    rows = []
    for stage, losses in (("first", [2., 4., 6.]), ("expanded", [8., 10., 12., 14.])):
        rows.extend({"stage": stage, "is_oos": i == 0, "nll": loss,
                     "brier": loss / 10, "accuracy": float(i != 0)} for i, loss in enumerate(losses))
    result = summarize_observations(rows, .25)
    assert result["overall"]["nll"] == pytest.approx(8.)
    assert result["stages"]["first"]["oos_standardized_nll"] == pytest.approx(.25 * 2 + .75 * 5)
    assert result["stages"]["expanded"]["oos_standardized_nll"] == pytest.approx(.25 * 8 + .75 * 12)
    assert result["selection_mean_nll"] == pytest.approx((4.25 + 11) / 2)
    assert result["overall"]["oos_standardized_accuracy"] == pytest.approx(.75)


def test_validation_preserves_parameters_gradients_rng_and_module_modes():
    torch.manual_seed(7)
    random.seed(7)
    model = TinyDecision().train()
    model.dropout.eval()  # Mixed modes also survive validation.
    model.emb.weight.grad = torch.ones_like(model.emb.weight)
    state = copy.deepcopy(model.state_dict())
    gradient = model.emb.weight.grad.clone()
    rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    stats = evaluate_views(model, TinyTokenizer(), views(), 10, 4, "cpu", 2 / 11)
    assert model.training and not model.dropout.training
    assert torch.equal(rng, torch.get_rng_state()) and random.getstate() == python_rng
    assert torch.equal(gradient, model.emb.weight.grad)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
    assert stats["overall"]["views"] == 6 and stats["overall"]["none_target_views"] == 2
    # A short final minibatch must not change any metric's weight.
    repeated = evaluate_views(model, TinyTokenizer(), views(), 10, 1, "cpu", 2 / 11)
    assert repeated["selection_mean_nll"] == pytest.approx(stats["selection_mean_nll"], abs=1e-7)
    assert repeated["overall"]["brier"] == pytest.approx(stats["overall"]["brier"], abs=1e-7)


def test_validation_mode_restores_rng_even_when_validation_fails():
    model = TinyDecision().train()
    rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    with pytest.raises(RuntimeError):
        with validation_mode(model):
            assert not model.training and not torch.is_grad_enabled()
            torch.rand(3)
            random.random()
            raise RuntimeError("deliberate")
    assert model.training and torch.equal(rng, torch.get_rng_state())
    assert python_rng == random.getstate()


def test_schedule_warms_up_then_decays_to_floor():
    factors = [learning_rate_factor(step, 100, .05, .1) for step in range(1, 101)]
    assert factors[:5] == pytest.approx([.2, .4, .6, .8, 1.])
    assert all(a >= b for a, b in zip(factors[4:], factors[5:]))
    assert factors[-1] == pytest.approx(.1)
    assert all(.1 <= factor <= 1 for factor in factors)
    with pytest.raises(ValueError):
        learning_rate_factor(0, 100)


def test_selection_prefers_earlier_ties_and_ignores_later_overfit():
    losses = [1., .7, .69995, .8, .65, .9]
    best, best_index = None, None
    for index, loss in enumerate(losses):
        if is_better_checkpoint(loss, best, 1e-4):
            best, best_index = loss, index
    assert best == .65 and best_index == 4
    assert not is_better_checkpoint(.64995, best, 1e-4)
    with pytest.raises(ValueError):
        is_better_checkpoint(float("nan"), best)


def test_training_validation_overlap_rejected_before_training():
    train = [{"id": "a", "text": "Hello  World"}]
    check_disjoint(train, [{"id": "b", "text": "different"}])
    with pytest.raises(ValueError, match="IDs overlap"):
        check_disjoint(train, [{"id": "a", "text": "different"}])
    with pytest.raises(ValueError, match="query texts overlap"):
        check_disjoint(train, [{"id": "b", "text": "hello world"}])


def test_missing_validation_stratum_is_not_silently_renormalized():
    with pytest.raises(ValueError, match="both OOS and in-domain"):
        summarize_observations([{"stage": "first", "is_oos": False, "nll": 0., "brier": 0., "accuracy": 1.}], .2)


def test_full_loop_publishes_earlier_best_and_weights_short_updates(tmp_path, monkeypatch):
    import recovery_router.train_extended as trainer

    candidates = [{"id": "intent0", "label": "intent0"}, {"id": "intent1", "label": "intent1"}]
    def records(prefix, count):
        return [{"id": f"{prefix}{i}", "text": f"{prefix} query {i}",
                 "label": "intent0", "is_oos": i == 0,
                 "candidates": candidates[:1], "expanded_candidates": candidates} for i in range(count)]
    for name, count in (("train", 3), ("validation", 2)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records(name, count)))
    config = {"model": {"max_length": 16}, "training": {
        "epochs": 2, "micro_batch_size": 2, "gradient_accumulation_steps": 2,
        "validation_steps": 1, "learning_rate": .01}, "policy": {"target_oos_prior": .2}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    model = TinyDecision()
    monkeypatch.setattr(trainer, "create_model", lambda cfg, device: (model, TinyTokenizer()))
    development_losses = iter([1., .5, .2, .25, .3])
    monkeypatch.setattr(trainer, "evaluate_views", lambda *a, **kw: {"selection_mean_nll": next(development_losses)})

    def fake_save(model, tokenizer, destination, model_cfg, extra):
        destination = Path(destination)
        (destination / "adapter").mkdir(parents=True)
        (destination / "tokenizer").mkdir()
        (destination / "metadata.json").write_text(json.dumps({"model": model_cfg, **extra}))
        (destination / "head.safetensors").write_bytes(str(extra["step"]).encode())
        (destination / "adapter" / "weights.bin").write_bytes(b"tiny-test-only")
        (destination / "tokenizer" / "tokenizer.json").write_text("{}")

    monkeypatch.setattr(trainer, "save_model_bundle", fake_save)
    trainer.main(["--config", str(tmp_path / "config.json"), "--data", str(tmp_path / "train.jsonl"),
                  "--validation", str(tmp_path / "validation.jsonl"), "--output", str(tmp_path / "run"),
                  "--device", "cpu", "--max-seconds", "60"])
    summary = json.loads((tmp_path / "run" / "training_summary.json").read_text())
    report = json.loads((tmp_path / "run" / "model" / "training_report.json").read_text())
    metadata = json.loads((tmp_path / "run" / "model" / "metadata.json").read_text())
    assert summary["actual_executed_optimizer_steps"] == 4
    assert summary["actual_training_examples_seen"] == 12
    assert report["optimizer_steps_total"] == metadata["step"] == summary["selected_step"] == 2
    assert report["stop_reason"] == "validation_selected"
    assert len(summary["validation_curve"]) == 5
    assert summary["validation_curve"][0]["selected_at_this_step"] is False
    assert [r["examples"] for r in summary["optimizer_curve"]] == [4, 2, 4, 2]
    expected = sum(r["mean_nll"] * r["examples"] for r in summary["optimizer_curve"]) / 12
    assert summary["weighted_training_mean_nll"] == pytest.approx(expected)
    first_epoch = sum(r["mean_nll"] * r["examples"] for r in summary["optimizer_curve"][:2]) / 6
    assert summary["epochs"][0]["mean_nll"] == pytest.approx(first_epoch)
    assert summary["selected_model_fingerprint"] == summary["validation_curve"][2]["model_fingerprint"]
