"""Sequential online inference: an expansion is observed only after choosing it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .calibration import checked_probs, scale_probs
from .io import read_json, read_jsonl, sha256_file
from .model import bundle_fingerprint, collate_views, load_model_bundle, make_view, pack_probabilities
from .policy import COMMIT, DEFER, EXPAND, Costs, load_actor, state_features
from .retrieval import MiniLMRetriever, TFIDFRetriever, stable_seed

CATALOGS = {"clinc_base": "clinc120", "clinc_updated": "clinc150", "banking": "banking77"}
ACTION_NAMES = {COMMIT: "commit", EXPAND: "expand", DEFER: "handoff"}


def candidate_schedule(retrieval):
    initial = retrieval.get("initial_k", retrieval.get("first_k", 1))
    expanded = retrieval.get("expanded_k", 5)
    if (not isinstance(initial, int) or isinstance(initial, bool)
            or not isinstance(expanded, int) or isinstance(expanded, bool)
            or not 1 <= initial < expanded <= 20):
        raise ValueError("Candidate schedule must satisfy 1 <= initial_k < expanded_k <= 20")
    return initial, expanded


def validate_trained_counts(metadata, initial_k, expanded_k):
    supported = metadata.get("trained_candidate_counts")
    if not isinstance(supported, list) or not {initial_k, expanded_k}.issubset(set(supported)):
        raise ValueError("Candidate schedule is not recorded in trained_candidate_counts; use a matching trained bundle and data manifest")


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


class ModelForecaster:
    """One observed state, one forward pass; gold labels are never supplied."""

    def __init__(self, model, tokenizer, max_length=512, device="cpu"):
        self.model, self.tokenizer = model, tokenizer
        self.max_length, self.device = max_length, device
        self.model.eval()

    def __call__(self, row, expanded=False):
        start = time.perf_counter()
        view = make_view(row, expanded)
        batch = collate_views(self.tokenizer, [view], self.max_length)
        # collate_views makes a dummy target for unlabeled inputs; do not pass it.
        batch = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
        tokenize_ms = (time.perf_counter() - start) * 1000
        _sync(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            result = self.model(**batch)
            probabilities = pack_probabilities(result["logits"][0], view["candidate_count"])
        _sync(self.device)
        return {"probs": probabilities, "model_ms": (time.perf_counter() - start) * 1000,
                "tokenize_and_transfer_ms": tokenize_ms}


def _retrieve(retriever, text, k, seed, query_id, stage):
    start = time.perf_counter()
    candidates = [dict(c) for c in retriever.retrieve(text, k=k)]
    if not candidates or len(candidates) > k:
        raise ValueError("Retriever must return between one and k candidates")
    if len({c["id"] for c in candidates}) != len(candidates):
        raise ValueError("Retriever returned duplicate candidate identifiers")
    # The fitted decision head was trained on shuffled positions, not rank order.
    random.Random(stable_seed(seed, query_id + ":" + stage)).shuffle(candidates)
    return candidates, (time.perf_counter() - start) * 1000


def route_text(text, retriever, forecaster, actor, calibration, costs=Costs(),
               catalog_id="unknown", seed=42, initial_k=1, expanded_k=5):
    """Route one unlabeled text, with injectable components for local testing.

    forecaster(row, expanded) returns a dictionary with `probs` including NONE.
    No counterfactual second pass is made unless the first action is EXPAND.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("A nonempty text string is required")
    candidate_schedule({"initial_k": initial_k, "expanded_k": expanded_k})
    for stage in ("first", "expanded"):
        value = float(calibration[stage + "_temperature"])
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Both calibration temperatures must be finite and positive")
    started = time.perf_counter()
    query_id = "online:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    row = {"id": query_id, "text": text}
    predictions, traces = {}, []
    expanded = False
    for stage_number in range(2):
        stage = "expanded" if expanded else "first"
        candidates, retrieval_ms = _retrieve(retriever, text, expanded_k if expanded else initial_k,
                                             seed, query_id, stage)
        row["expanded_candidates" if expanded else "candidates"] = candidates
        forecast = forecaster(row, expanded)
        raw = checked_probs(forecast["probs"], len(candidates))
        probabilities = scale_probs(raw, calibration[stage + "_temperature"])
        predictions[stage + "_probs"] = probabilities.tolist()
        tick = time.perf_counter()
        features = state_features(row, predictions, expanded, costs)
        choices = np.asarray(actor.choose(features[None, :])).reshape(-1)
        if len(choices) != 1 or int(choices[0]) not in ACTION_NAMES:
            raise ValueError("Actor returned an invalid typed action")
        action = int(choices[0])
        if expanded and action == EXPAND:
            raise ValueError("Actor requested an invalid second expansion")
        actor_ms = (time.perf_counter() - tick) * 1000
        best = int(np.argmax(probabilities[:-1]))
        selected = candidates[best]
        traces.append({
            "stage": stage, "action": ACTION_NAMES[action],
            "candidates": [{"candidate_id": c["id"], "label": c.get("label", c["id"]),
                            "probability": float(probabilities[i])} for i, c in enumerate(candidates)],
            "none_probability": float(probabilities[-1]),
            "retrieval_ms": retrieval_ms, "model_ms": forecast.get("model_ms"),
            "tokenize_and_transfer_ms": forecast.get("tokenize_and_transfer_ms"),
            "actor_ms": actor_ms,
        })
        if action == EXPAND:
            expanded = True
            continue
        return {
            "action": ACTION_NAMES[action], "expanded": expanded, "catalog_id": catalog_id,
            "candidate_schedule": {"initial_k": initial_k, "expanded_k": expanded_k},
            "candidate_id": selected["id"] if action == COMMIT else None,
            "label": selected.get("label", selected["id"]) if action == COMMIT else None,
            "confidence": float(probabilities[best]),
            "confidence_event": {"candidate_id": selected["id"],
                                 "meaning": "forecast that this shown candidate is correct; NONE mass retained"},
            "forecast_status": "held-out temperature applied; deployment calibration is not guaranteed",
            "stage_traces": traces,
            "costs": {"correct_reward": costs.correct, "wrong_penalty": costs.wrong,
                      "handoff_cost": costs.defer, "expansion_cost": costs.expand, "units": "simulated utility"},
            "latency_ms": (time.perf_counter() - started) * 1000,
            "latency_scope": "actual sequential retrieval, tokenization, forward passes and decisions; excludes artifact/index loading",
        }
    raise RuntimeError("No terminal action was produced")


def load_retriever(data_dir, catalog, device="cpu"):
    """Only declared train-index rows become searchable; no test labels needed."""
    data_dir = Path(data_dir)
    manifest = read_json(data_dir / "manifest.json")
    initial_k, expanded_k = candidate_schedule(manifest["retrieval"])
    catalogs = read_json(data_dir / "catalogs.json")
    catalog_id = CATALOGS[catalog]
    permitted = {entry["label"] for entry in catalogs[catalog_id]}
    filename = "banking_index.jsonl" if catalog == "banking" else "index.jsonl"
    for name in (filename, "catalogs.json"):
        declared_hash = manifest.get("files", {}).get(name, {}).get("sha256")
        if declared_hash and sha256_file(data_dir / name) != declared_hash:
            raise ValueError(f"Catalog artifact differs from its manifest: {name}")
    rows = [r for r in read_jsonl(data_dir / filename) if r["label"] in permitted]
    if {r["label"] for r in rows} != permitted:
        raise ValueError("Index rows do not cover the declared catalog")
    backend = manifest["retrieval"]["backend"]
    if backend == "minilm":
        retriever = MiniLMRetriever(rows, cache_dir=None, device=device)
    elif backend == "tfidf":
        retriever = TFIDFRetriever(rows)
    else:
        raise ValueError(f"Unsupported declared retrieval backend: {backend}")
    return retriever, {"catalog_id": catalog_id, "backend": backend,
                       "initial_k": initial_k, "expanded_k": expanded_k,
                       "index_sha256": sha256_file(data_dir / filename),
                       "catalog_sha256": sha256_file(data_dir / "catalogs.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local adapted model bundle")
    parser.add_argument("--policy", required=True, help="Local actor bundle")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--catalog", choices=tuple(CATALOGS), default="clinc_base")
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--wrong-cost", type=float, default=5.)
    parser.add_argument("--defer-cost", type=float, default=.2)
    parser.add_argument("--expand-cost", type=float, default=.05)
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    costs = Costs(wrong=args.wrong_cost, defer=args.defer_cost, expand=args.expand_cost)
    schedule = candidate_schedule(read_json(Path(args.data_dir) / "manifest.json")["retrieval"])
    validate_trained_counts(read_json(Path(args.model) / "metadata.json"), *schedule)
    fingerprint = bundle_fingerprint(args.model)
    policy_meta = read_json(Path(args.policy) / "policy.json")
    calibration_hash = sha256_file(args.calibration)
    if policy_meta.get("provider_fingerprint") != fingerprint:
        raise ValueError("Actor was fitted with a different model provider")
    if policy_meta.get("calibration_sha256") != calibration_hash:
        raise ValueError("Actor was fitted with a different calibration artifact")
    load_start = time.perf_counter()
    model, tokenizer, meta = load_model_bundle(args.model, args.device)
    actor = load_actor(args.policy)
    calibration = read_json(args.calibration)
    retriever, retrieval_meta = load_retriever(args.data_dir, args.catalog, args.device)
    load_ms = (time.perf_counter() - load_start) * 1000
    forecaster = ModelForecaster(model, tokenizer, int(meta["model"].get("max_length", 512)), args.device)
    result = route_text(args.text, retriever, forecaster, actor, calibration, costs,
                        retrieval_meta["catalog_id"], initial_k=retrieval_meta["initial_k"],
                        expanded_k=retrieval_meta["expanded_k"])
    result.update(model={"base": meta["model"], "bundle_fingerprint": fingerprint,
                         "bundle_metadata_sha256": sha256_file(Path(args.model)/"metadata.json")},
                  retrieval=retrieval_meta, artifact_and_index_load_ms=load_ms,
                  policy_sha256=sha256_file(Path(args.policy)/"policy.safetensors"),
                  calibration_sha256=calibration_hash)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
