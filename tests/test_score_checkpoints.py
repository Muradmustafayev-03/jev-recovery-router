"""Read-only learning curves preserve raw NONE probabilities and OOS weighting."""
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from test_model import tiny_local, record
from recovery_router.model import create_model, save_model_bundle, bundle_fingerprint


@pytest.fixture
def scorer():
    path = Path(__file__).resolve().parents[1] / "scripts/score_checkpoints.py"
    spec = importlib.util.spec_from_file_location("checkpoint_score_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_raw_nll_brier_accuracy_none_and_prior_weighting(scorer, monkeypatch):
    rows = [{"id": "stable", "is_oos": False}, {"id": "oos", "is_oos": True}]
    views = [
        {"id": "stable", "stage": "first", "target": 0, "candidate_count": 1, "index": 0},
        {"id": "stable", "stage": "expanded", "target": 0, "candidate_count": 2, "index": 1},
        {"id": "oos", "stage": "first", "target": 20, "candidate_count": 1, "index": 2},
        {"id": "oos", "stage": "expanded", "target": 20, "candidate_count": 2, "index": 3}]
    logits = torch.full((4, 21), -torch.inf)
    for index, values in enumerate(([.8, .2], [.6, .2, .2], [.1, .9], [.1, .1, .8])):
        logits[index, :len(values)-1] = torch.tensor(values[:-1]).log()
        logits[index, 20] = math.log(values[-1])
    class Model:
        def eval(self):
            return self
        def __call__(self, input_ids, **kwargs):
            assert not torch.is_grad_enabled()
            return {"logits": logits[input_ids[:, 0]]}
    def collate(tokenizer, chunk, max_length):
        return {"input_ids": torch.tensor([[v["index"]] for v in chunk]),
                "attention_mask": torch.ones(len(chunk), 1),
                "candidate_counts": torch.tensor([v["candidate_count"] for v in chunk]),
                "labels": torch.tensor([v["target"] for v in chunk])}
    monkeypatch.setattr(scorer, "collate_views", collate)
    result = scorer.score_views(Model(), None, views, rows, 2/11, batch_size=3)
    expected = .5 * ((9/11)*(-math.log(.8)-math.log(.6)) + (2/11)*(-math.log(.9)-math.log(.8)))
    assert result["selector_weighted_nll"] == pytest.approx(expected, abs=1e-7)
    assert result["stages"]["first"]["brier"] == pytest.approx(.05, abs=1e-7)
    assert result["stages"]["expanded"]["brier"] == pytest.approx(.15, abs=1e-7)
    assert all(stage["accuracy"] == 1 and stage["n_none_targets"] == 1 for stage in result["stages"].values())
    with pytest.raises(ValueError, match="every query"):
        scorer.score_views(Model(), None, views[:-1], rows, 2/11)


def test_cli_scores_all_checkpoints_without_mutating_bundles(scorer, tiny_local, tmp_path):
    root = tmp_path / "v1"
    model, tokenizer = create_model(tiny_local, "cpu")
    for path, step in ((root / "checkpoints/step-000100", 100), (root, 922)):
        save_model_bundle(model, tokenizer, path, tiny_local, {"step": step, "trained_candidate_counts": [1, 2]})
    original = {str(path): bundle_fingerprint(path) for path in scorer.checkpoint_paths(root)}
    rows = [record("one"), record("two")]
    for row in rows:
        row.update(split="tuning", group="stable")
        row["candidates"] = row["candidates"][:1]
    rows[1].update(is_oos=True, group="oos")
    validation = tmp_path / "tuning.jsonl"
    validation.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"policy": {"target_oos_prior": 2/11}}))
    output = tmp_path / "curve.json"
    args = ["--model-root", str(root), "--validation", str(validation), "--config", str(config),
            "--output", str(output), "--device", "cpu", "--max-seconds", "30"]
    scorer.main(args)
    report = json.loads(output.read_text())
    assert report["status"] == "complete" and report["diagnostic_only"]
    assert [entry["step"] for entry in report["checkpoints"]] == [100, 922]
    assert report["checkpoints"][-1]["is_final_baseline"]
    assert report["best_checkpoint_diagnostic_only"]["step"] == 100
    assert report["validation_sha256"] == scorer.sha256_file(validation)
    assert original == {str(path): bundle_fingerprint(path) for path in scorer.checkpoint_paths(root)}
    with pytest.raises(FileExistsError):
        scorer.main(args)
    args[args.index("--output") + 1] = str(root / "forbidden.json")
    with pytest.raises(ValueError, match="outside"):
        scorer.main(args)


def test_deadline_is_checked_before_inference(scorer):
    with pytest.raises(TimeoutError):
        scorer.check_deadline(0)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_nonpositive_or_nonfinite_cap_is_refused(scorer, value):
    with pytest.raises(SystemExit):
        scorer.main(["--model-root", "unused", "--validation", "unused", "--config", "unused",
                     "--output", "unused", "--device", "cpu", "--max-seconds", value])
