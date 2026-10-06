"""Offline first/expanded forecasts; expanded features never enter the first pass."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from .model import (InputTooLong, MAX_CANDIDATES, collate_views, encode_view,
                    load_model_bundle, make_view, pack_probabilities, validate_revision, bundle_fingerprint)


def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def predict_records(model, tokenizer, records, device="cpu", batch_size=16, max_length=512,
                    letter_ids=None, trained_candidate_counts=None, model_fingerprint=None):
    """All failures are explicit; output contains no invented replacement probabilities."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    outputs = {r["id"]: {"id": r["id"]} for r in records}
    if len(outputs) != len(records):
        raise ValueError("Duplicate record IDs")
    model.eval()
    for expanded in (False, True):
        stage = "expanded" if expanded else "first"
        views = []
        for record in records:
            begin = time.perf_counter()
            view = make_view(record, expanded)
            if trained_candidate_counts is not None and view["candidate_count"] not in trained_candidate_counts:
                outputs[record["id"]][f"{stage}_error"] = (
                    f"Candidate count {view['candidate_count']} was not present in accepted training views "
                    f"{trained_candidate_counts}; refusing an unsupported positional-head evaluation")
                continue
            if letter_ids is not None:
                choices = ", ".join(f"{chr(65+i)}=position {i}" for i in range(view["candidate_count"]))
                view["text"] += f"\nReturn one letter only: {choices}, U=NONE_OF_SHOWN.\nANSWER:"
                if getattr(tokenizer, "chat_template", None):
                    view["text"] = tokenizer.apply_chat_template(
                        [{"role": "user", "content": view["text"]}], tokenize=False,
                        add_generation_prompt=True, enable_thinking=False)
            try:
                view = encode_view(tokenizer, view, max_length)
            except InputTooLong as exc:
                outputs[record["id"]][f"{stage}_error"] = str(exc)
                continue
            view["tokenize_ms"] = (time.perf_counter() - begin) * 1000
            views.append(view)
        for offset in range(0, len(views), batch_size):
            chunk = views[offset:offset + batch_size]
            _sync(device)
            begin = time.perf_counter()
            batch = collate_views(tokenizer, chunk, max_length)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.inference_mode():
                if letter_ids is None:
                    result = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                   candidate_counts=batch["candidate_counts"])
                    logits = result["logits"]
                else:
                    # Padding is left-sided for CausalLM; final position is always the decision prefix.
                    result = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                   use_cache=False, logits_to_keep=1, return_dict=True)
                    logits = result.logits[:, -1, letter_ids]
                probs = [pack_probabilities(logits[i], v["candidate_count"]) for i, v in enumerate(chunk)]
            _sync(device)
            batch_ms = (time.perf_counter() - begin) * 1000
            for view, prob in zip(chunk, probs):
                row = outputs[view["id"]]
                row[f"{stage}_probs"] = prob
                row[f"{stage}_ms"] = view["tokenize_ms"] + batch_ms
                row[f"{stage}_batch_size"] = len(chunk)
                row[f"{stage}_batch_ms"] = batch_ms
                row[f"{stage}_tokenize_ms"] = view["tokenize_ms"]
                row[f"{stage}_candidate_ids"] = view["candidate_ids"]
    for row in outputs.values():
        if model_fingerprint is not None:
            row["model_fingerprint"] = model_fingerprint
        row["timing_scope"] = "encoding plus observed shared batch wall latency; retrieval excluded; use batch-size 1 for serial latency"
        row["forecast_type"] = ("constrained_letter_conditional_distribution" if letter_ids is not None
                                else "typed_head_distribution_including_none")
    return list(outputs.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Adapted model bundle")
    group.add_argument("--base-model", help="Optional unadapted forced-single-letter comparator")
    parser.add_argument("--revision", help="Immutable HF commit for --base-model")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    started = time.monotonic()
    letter_ids = None
    trained_candidate_counts = None
    if args.model:
        model, tokenizer, meta = load_model_bundle(args.model, args.device)
        max_length = args.max_length or int(meta["model"].get("max_length", 512))
        trained_candidate_counts = meta.get("trained_candidate_counts")
        if (not isinstance(trained_candidate_counts, list) or not trained_candidate_counts
                or any(isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_CANDIDATES
                       for n in trained_candidate_counts)):
            raise ValueError("Adapted CLI prediction requires explicit trained_candidate_counts metadata; legacy bundles must be regenerated from a recorded training run")
        fingerprint = bundle_fingerprint(args.model)
        base_name = meta["model"].get("name", meta["model"].get("name_or_path"))
        base_revision = meta["model"].get("revision")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        validate_revision(args.base_model, args.revision)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.revision, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        tokenized = [tokenizer.encode(chr(65+i), add_special_tokens=False) for i in range(MAX_CANDIDATES + 1)]
        if any(len(t) != 1 for t in tokenized) or len({t[0] for t in tokenized}) != MAX_CANDIDATES + 1:
            raise ValueError("This tokenizer does not provide distinct single-token A..U labels")
        letter_ids = [t[0] for t in tokenized]
        dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(args.base_model, revision=args.revision,
                                                    torch_dtype=dtype, attn_implementation="sdpa",
                                                    trust_remote_code=False).to(args.device)
        max_length = args.max_length or 512
        base_name, base_revision = args.base_model, args.revision
        fingerprint = hashlib.sha256(json.dumps({"base_model": base_name, "revision": base_revision,
                                                 "method": "constrained_single_letter_A_to_U"}, sort_keys=True).encode()).hexdigest()
    records = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    predictions = predict_records(model, tokenizer, records, args.device, args.batch_size, max_length,
                                  letter_ids, trained_candidate_counts, fingerprint)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8")
    rejected = sum("first_error" in r or "expanded_error" in r for r in predictions)
    summary = {"records": len(predictions), "records_with_errors": rejected,
               "elapsed_seconds": time.monotonic() - started, "output": str(output),
               "model": args.model or args.base_model, "revision": args.revision,
               "model_fingerprint": fingerprint, "base_model": base_name, "base_revision": base_revision,
               "provider_kind": "adapted_local_bundle" if args.model else "hf_pinned_single_letter",
               "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
               "trained_candidate_counts": trained_candidate_counts,
               "note": "Both views are cached offline for evaluation. Online first decisions must not access expanded-view data."}
    output.with_suffix(output.suffix + ".meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    if rejected:
        raise SystemExit("Prediction length failures were saved explicitly; repair inputs before policy evaluation")


if __name__ == "__main__":
    main()
