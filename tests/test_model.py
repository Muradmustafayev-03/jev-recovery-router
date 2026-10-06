"""No network, checkpoint downloads, or GPU required."""
import json
from pathlib import Path

import pytest
import torch
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from recovery_router.model import (DecisionModel, InputTooLong, collate_views, create_model,
                                  encode_view, load_model_bundle, make_view, pack_probabilities,
                                  save_model_bundle, serialize_record, validate_revision)


@pytest.fixture
def tiny_local(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(123)
    path = tmp_path / "tiny-qwen"
    config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, max_position_embeddings=2048, pad_token_id=0,
                        bos_token_id=2, eos_token_id=3, attention_dropout=0.)
    Qwen3Model(config).save_pretrained(path)
    raw = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3,
                               "transfer": 4, "balance": 5, "query": 6, "DECISION": 7},
                              unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]",
                                        unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]")
    tokenizer.save_pretrained(path)
    return {"name": str(path), "revision": None, "max_length": 512,
            "lora_rank": 2, "lora_alpha": 4, "lora_dropout": 0.}


def record(identifier="one"):
    candidates = [{"id": "transfer", "label": "transfer", "description": "transfer funds"},
                  {"id": "balance", "label": "balance", "description": "check balance"}]
    return {"id": identifier, "text": "transfer", "label": "transfer", "is_oos": False,
            "candidates": candidates, "expanded_candidates": candidates,
            "intervention": "natural", "secret_gold_annotation": "GOLD_NOT_IN_INPUT"}


def test_gold_never_serialized_and_none_target():
    example = record()
    original = make_view(example)
    example["label"] = "unsupported_gold"
    changed = make_view(example)
    assert original["text"] == changed["text"]
    assert "GOLD_NOT_IN_INPUT" not in original["text"]
    assert original["target"] == 0
    assert changed["target"] == 20
    example["label"] = "transfer"
    example["is_oos"] = True
    assert make_view(example)["target"] == 20


def test_none_mass_is_preserved():
    logits = torch.zeros(21)
    logits[20] = 4
    probabilities = pack_probabilities(logits, 2)
    assert len(probabilities) == 3
    assert sum(probabilities) == pytest.approx(1.)
    assert probabilities[-1] > .96
    assert probabilities[0] < .02
    with pytest.raises(ValueError):
        serialize_record("query", [])


def test_fail_instead_of_truncating(tiny_local):
    _, tokenizer = create_model(tiny_local, "cpu")
    view = make_view(record())
    with pytest.raises(InputTooLong):
        encode_view(tokenizer, view, 4)
    assert encode_view(tokenizer, view, 512)["candidate_count"] == 2


def test_tiny_qwen_padding_mask_and_backward(tiny_local):
    model, _ = create_model(tiny_local, "cpu")
    model.eval()
    a = model(torch.tensor([[4, 5, 7, 0]]), torch.tensor([[1, 1, 1, 0]]), torch.tensor([2]))
    b = model(torch.tensor([[0, 4, 5, 7]]), torch.tensor([[0, 1, 1, 1]]), torch.tensor([2]))
    assert torch.allclose(a["logits"], b["logits"], atol=1e-5)
    assert torch.isneginf(a["logits"][0, 2:20]).all()
    loss = model(torch.tensor([[4, 5, 7]]), torch.ones(1, 3, dtype=torch.long),
                 torch.tensor([2]), labels=torch.tensor([20]))["loss"]
    loss.backward()
    assert model.head.weight.grad is not None
    assert torch.isfinite(model.head.weight.grad).all()


def test_bundle_roundtrip_and_predict(tiny_local, tmp_path):
    from recovery_router.predict import predict_records
    model, tokenizer = create_model(tiny_local, "cpu")
    model.eval()
    before = predict_records(model, tokenizer, [record()], max_length=512)[0]
    output = tmp_path / "bundle"
    save_model_bundle(model, tokenizer, output, tiny_local)
    loaded, loaded_tok, meta = load_model_bundle(output, "cpu")
    after = predict_records(loaded, loaded_tok, [record()], max_length=512)[0]
    assert before["first_probs"] == pytest.approx(after["first_probs"], abs=1e-6)
    assert len(after["expanded_probs"]) == 3
    assert meta["none_index"] == 20
    assert after["first_candidate_ids"] == ["transfer", "balance"]


def test_short_train_resume(tiny_local, tmp_path):
    from recovery_router.train import main
    data = tmp_path / "data.jsonl"
    examples = [record(str(i)) for i in range(3)]
    for example in examples:
        example["candidates"] = example["candidates"][:1]
    data.write_text("\n".join(json.dumps(example) for example in examples), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model": tiny_local, "training": {
        "seed": 3, "epochs": 1, "micro_batch_size": 1, "gradient_accumulation_steps": 1,
        "learning_rate": .001, "checkpoint_steps": 0}}), encoding="utf-8")
    output = tmp_path / "run"
    common = ["--config", str(config), "--data", str(data), "--output", str(output), "--device", "cpu"]
    main(common + ["--smoke-steps", "1"])
    report = json.loads((output / "training_report.json").read_text())
    assert report["optimizer_steps_total"] == 1
    metadata = json.loads((output / "metadata.json").read_text())
    assert len(metadata["trained_candidate_counts"]) == 1
    assert metadata["planned_candidate_counts"] == [1, 2]
    main(common + ["--resume", str(output)])
    report = json.loads((output / "training_report.json").read_text())
    assert report["optimizer_steps_total"] == 6
    assert report["stop_reason"] == "epochs_complete"
    assert json.loads((output / "metadata.json").read_text())["trained_candidate_counts"] == [1, 2]


def test_remote_revision_must_be_pinned():
    with pytest.raises(ValueError):
        validate_revision("Qwen/Qwen3-1.7B", "main")
    validate_revision("Qwen/Qwen3-1.7B", "a" * 40)


def test_single_token_comparator_forward(tiny_local):
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    from recovery_router.predict import predict_records
    tokenizer = AutoTokenizer.from_pretrained(tiny_local["name"])
    tokenizer.padding_side = "left"
    model = Qwen3ForCausalLM.from_pretrained(tiny_local["name"])
    # Artificial distinct vocabulary IDs test the scoring branch; real CLI verifies A..U tokens.
    prediction = predict_records(model, tokenizer, [record()], max_length=512,
                                 letter_ids=list(range(21)))[0]
    assert len(prediction["first_probs"]) == 3
    assert sum(prediction["first_probs"]) == pytest.approx(1.)
    assert prediction["forecast_type"] == "constrained_letter_conditional_distribution"


def test_bundle_fingerprint_and_unsupported_count(tiny_local, tmp_path):
    from recovery_router.model import bundle_fingerprint
    from recovery_router.predict import predict_records
    from safetensors.torch import load_file, save_file
    model, tokenizer = create_model(tiny_local, "cpu")
    bundle = tmp_path / "bundle"
    save_model_bundle(model, tokenizer, bundle, tiny_local, {"trained_candidate_counts": [1, 5]})
    original = bundle_fingerprint(bundle)
    (bundle / "training_state.pt").write_bytes(b"excluded optimizer state")
    assert bundle_fingerprint(bundle) == original
    head = {k: v.clone() for k, v in load_file(str(bundle / "head.safetensors")).items()}
    head["bias"][0] += 1
    save_file(head, str(bundle / "head.safetensors"))
    assert bundle_fingerprint(bundle) != original
    result = predict_records(model, tokenizer, [record()], trained_candidate_counts=[1, 5])[0]
    assert "not present" in result["first_error"]
    assert "first_probs" not in result
