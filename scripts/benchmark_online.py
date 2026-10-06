"""Measure actual sequential warm online latency; never time cached replay."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from recovery_router.calibration import checked_probs, scale_probs
from recovery_router.inference import (ModelForecaster, _retrieve, candidate_schedule, load_retriever,
                                       route_text, validate_trained_counts)
from recovery_router.io import read_json, sha256_file, write_json
from recovery_router.model import bundle_fingerprint, load_model_bundle
from recovery_router.policy import Costs, load_actor

SPLITS = {"clinc_base": "test_base", "clinc_updated": "test", "banking": "transfer"}


def route_direct(text, retriever, forecaster, calibration, costs=Costs(), expanded_k=5, seed=42):
    """Retrieve the full supported set directly, with one belief pass and gate.

    Uses the same expanded candidate order and temperature as the adaptive
    router's second stage. There is no first pass or expansion action.
    """
    if not isinstance(text, str) or not text.strip() or not 1 <= expanded_k <= 20:
        raise ValueError("Direct route requires text and 1..20 supported candidates")
    query_id = "online:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    candidates, retrieval_ms = _retrieve(retriever, text, expanded_k, seed, query_id, "expanded")
    row = {"id": query_id, "text": text, "expanded_candidates": candidates}
    forecast = forecaster(row, True)
    probabilities = scale_probs(checked_probs(forecast["probs"], len(candidates)),
                                calibration["expanded_temperature"])
    started = time.perf_counter()
    best = int(np.argmax(probabilities[:-1]))
    confidence = float(probabilities[best])
    commit = confidence * costs.correct - (1. - confidence) * costs.wrong >= -costs.defer
    action = "commit" if commit else "handoff"
    gate_ms = (time.perf_counter() - started) * 1000
    return {"action": action, "expanded": False,
            "candidate_id": candidates[best]["id"] if commit else None,
            "confidence": confidence, "direct_candidate_count": len(candidates),
            "stage_traces": [{"stage": "direct_full_candidate_set", "action": action,
                "candidates": [{"candidate_id": c["id"], "label": c.get("label", c["id"]),
                                "probability": float(probabilities[i])} for i, c in enumerate(candidates)],
                "none_probability": float(probabilities[-1]), "retrieval_ms": retrieval_ms,
                "model_ms": forecast.get("model_ms"), "tokenize_and_transfer_ms": forecast.get("tokenize_and_transfer_ms"),
                "gate_ms": gate_ms}]}


def select_queries(path, count=100):
    """Use file order and only text/ID; labels never select or tune the sample."""
    if not 1 <= count <= 1000:
        raise ValueError("count must be 1..1000 for this bounded latency benchmark")
    selected = []
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError("Benchmark query must contain a nonempty text")
            selected.append({"id": str(row["id"]), "text": row["text"]})
            if len(selected) == count:
                break
    if len(selected) != count:
        raise ValueError(f"Requested {count} queries but only {len(selected)} available")
    if len({q["id"] for q in selected}) != len(selected):
        raise ValueError("Duplicate IDs in benchmark sample")
    return selected


def measure_queries(queries, route, synchronize=lambda: None, warmup=3):
    if not queries or not 0 <= warmup <= 20:
        raise ValueError("Require a nonempty sample and 0..20 fixed warmup requests")
    warmup_started = time.perf_counter()
    for i in range(warmup):
        synchronize()
        route(queries[i % len(queries)]["text"])
        synchronize()
    warmup_ms = (time.perf_counter() - warmup_started) * 1000
    records = []
    for query in queries:
        synchronize()
        started = time.perf_counter()
        output = route(query["text"])
        synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        if output["action"] not in {"commit", "handoff"}:
            raise ValueError("Online route did not produce a terminal action")
        records.append({"id": query["id"], "latency_ms": elapsed,
                        "action": output["action"], "expanded": bool(output["expanded"]),
                        "candidate_id": output["candidate_id"], "stage_traces": output["stage_traces"]})
    values = np.asarray([r["latency_ms"] for r in records])
    return {"timing_status": "measured_sequential_online_latency",
            "n": len(records), "batch_size": 1, "warmup_requests": warmup, "warmup_ms": warmup_ms,
            "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
            "mean_ms": float(values.mean()), "min_ms": float(values.min()), "max_ms": float(values.max()),
            "percentile_method": "numpy linear interpolation", "action_counts": dict(Counter(r["action"] for r in records)),
            "expanded_requests": sum(r["expanded"] for r in records), "queries": records,
            "timing_scope": "synchronized real sequential retrieval, tokenization, belief and actor calls; artifact/index loading excluded and reported separately",
            "limitations": ["First queries in file order are deterministic, not a representative traffic sample",
                            "No labels, accuracy metrics, thresholds or policy parameters are optimized by this benchmark",
                            "No comparison to cached/offline component timing is made",
                            "Local batch-one latency does not measure concurrent service throughput or network serving overhead"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--catalog", choices=tuple(SPLITS), default="clinc_base")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--wrong-cost", type=float, default=5.)
    parser.add_argument("--defer-cost", type=float, default=.2)
    parser.add_argument("--expand-cost", type=float, default=.05)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Timing artifact already exists; use a fresh output path")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    data = Path(args.data_dir)
    query_path = data / (SPLITS[args.catalog] + ".jsonl")
    queries = select_queries(query_path, args.count)
    if not 0 <= args.warmup <= 20:
        parser.error("--warmup must be 0..20")
    schedule = candidate_schedule(read_json(data / "manifest.json")["retrieval"])
    metadata = read_json(Path(args.model) / "metadata.json")
    validate_trained_counts(metadata, *schedule)
    policy_meta = read_json(Path(args.policy) / "policy.json")
    cal_hash = sha256_file(args.calibration)
    model_fingerprint = bundle_fingerprint(args.model)
    if policy_meta.get("calibration_sha256") != cal_hash:
        raise ValueError("Actor/calibration mismatch")
    if policy_meta.get("provider_fingerprint") != model_fingerprint:
        raise ValueError("Actor/model provider mismatch")
    costs = Costs(wrong=args.wrong_cost, defer=args.defer_cost, expand=args.expand_cost)
    synchronize = torch.cuda.synchronize if args.device == "cuda" else lambda: None
    synchronize()
    load_started = time.perf_counter()
    model, tokenizer, metadata = load_model_bundle(args.model, args.device)
    actor = load_actor(args.policy)
    calibration = read_json(args.calibration)
    retriever, retrieval_meta = load_retriever(data, args.catalog, args.device)
    synchronize()
    load_ms = (time.perf_counter() - load_started) * 1000
    forecaster = ModelForecaster(model, tokenizer, metadata["model"].get("max_length", 512), args.device)
    def route(text):
        return route_text(text, retriever, forecaster, actor, calibration, costs,
                          retrieval_meta["catalog_id"], initial_k=schedule[0], expanded_k=schedule[1])
    report = measure_queries(queries, route, synchronize, args.warmup)
    direct = measure_queries(queries, lambda text: route_direct(text, retriever, forecaster, calibration,
                                                              costs, expanded_k=schedule[1]),
                             synchronize, args.warmup)
    direct.update(policy="direct_full_candidates_with_cost_derived_commit_or_handoff_gate",
                  candidate_count=schedule[1], decision_rule="p*correct-(1-p)*wrong >= -defer; retain NONE mass",
                  expansion_cost_status="No expansion action occurs. This timing comparison makes no expansion-cost or utility claim.")
    report["direct_baseline"] = direct
    report["online_comparison"] = {
        "status": "both_paths_measured_online_on_identical_unlabeled_queries",
        "execution_order": "adaptive block then direct block; each receives the same fixed warmups",
        "direct_baseline_scope": "one retrieval of expanded_k candidates and one expanded-stage belief pass; no initial-k pass or actor call",
        "limitations": "Fixed block order can be affected by clock/thermal drift. No claim of equal accuracy, coverage, utility or production speedup follows from these unlabeled timing measurements."}
    hardware = {"platform": platform.platform(), "processor": platform.processor(),
                "device": args.device, "python": platform.python_version(), "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda}
    if args.device == "cuda":
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        hardware.update(gpu_name=props.name, gpu_memory_bytes=props.total_memory,
                        cuda_device_index=torch.cuda.current_device())
    report.update(created_utc=datetime.now(timezone.utc).isoformat(), artifact_and_index_load_ms=load_ms,
                  hardware=hardware, model_fingerprint=model_fingerprint,
                  base_model=metadata["model"], actor_sha256=sha256_file(Path(args.policy)/"policy.safetensors"),
                  actor_metadata_sha256=sha256_file(Path(args.policy)/"policy.json"), calibration_sha256=cal_hash,
                  data_sha256=sha256_file(query_path), data_manifest_sha256=sha256_file(data/"manifest.json"),
                  retrieval=retrieval_meta, selection="first count rows in original file order; no label filtering",
                  candidate_schedule={"initial_k": schedule[0], "expanded_k": schedule[1]},
                  costs={"correct": costs.correct, "wrong": costs.wrong, "defer": costs.defer, "expand": costs.expand})
    write_json(output, report)
    print(json.dumps({"output": str(output), "timing_status": report["timing_status"],
                      "n": report["n"], "p50_ms": report["p50_ms"], "p95_ms": report["p95_ms"],
                      "direct_p50_ms": direct["p50_ms"], "direct_p95_ms": direct["p95_ms"]}))


if __name__ == "__main__":
    main()
