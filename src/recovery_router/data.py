"""Pinned public data, deterministic catalog splits, and explicit leakage audits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from .retrieval import (MINILM_MODEL, MINILM_REVISION, MiniLMRetriever, TFIDFRetriever,
                        attach_candidates, candidate_description, make_intervention, stable_seed)

CLINC_REVISION = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
BANKING_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCES = {
    "clinc_oos_plus.json": {"url": f"https://raw.githubusercontent.com/clinc/oos-eval/{CLINC_REVISION}/data/data_oos_plus.json", "revision": CLINC_REVISION, "license": "CC-BY-3.0"},
    "CLINC_LICENSE.txt": {"url": f"https://raw.githubusercontent.com/clinc/oos-eval/{CLINC_REVISION}/LICENSE", "revision": CLINC_REVISION, "license": "CC-BY-3.0"},
    "banking77_train.csv": {"url": f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{BANKING_REVISION}/banking_data/train.csv", "revision": BANKING_REVISION, "license": "CC-BY-4.0"},
    "banking77_test.csv": {"url": f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{BANKING_REVISION}/banking_data/test.csv", "revision": BANKING_REVISION, "license": "CC-BY-4.0"},
    "BANKING77_LICENSE.txt": {"url": f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{BANKING_REVISION}/LICENSE", "revision": BANKING_REVISION, "license": "CC-BY-4.0"},
}


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_sources(download_dir: Path) -> dict:
    download_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = download_dir / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    result = {}
    for name, source in SOURCES.items():
        path = download_dir / name
        if not path.exists():
            request = urllib.request.Request(source["url"], headers={"User-Agent": "recovery-router-research/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                contents = response.read()
            path.write_bytes(contents)
        digest = sha256(path)
        if name in previous and previous[name]["sha256"] != digest:
            raise ValueError(f"Cached source changed: {path}. Inspect rather than silently reuse it.")
        result[name] = {**source, "sha256": digest, "bytes": path.stat().st_size}
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _row(dataset: str, source_split: str, i: int, text: str, label: str) -> dict:
    record_id = f"{dataset}:{source_split}:{i:05d}"
    return {"id": record_id, "parent_id": record_id, "dataset": dataset, "source_split": source_split,
            "text": text, "label": label, "is_oos": label == "oos"}


def load_clinc(path: Path) -> dict[str, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    counts = {"train": 15000, "val": 3000, "test": 4500, "oos_train": 250, "oos_val": 100, "oos_test": 1000}
    if {k: len(data[k]) for k in counts} != counts:
        raise ValueError("Unexpected CLINC OOS+ split sizes; refusing implicit data-version change")
    return {split: [_row("clinc150", split, i, text, label) for i, (text, label) in enumerate(records)]
            for split, records in data.items()}


def load_banking(path: Path, split: str) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [_row("banking77", split, i, r["text"], r["category"]) for i, r in enumerate(csv.DictReader(stream))]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.casefold())).strip()


def leakage_audit(roles: dict[str, list[dict]]) -> dict:
    """Exact normalized-text collisions across mutually exclusive fitting roles.

    Evaluation variants are intentionally omitted: parent_id links duplicates.
    Public benchmark pretraining contamination and paraphrases are not detectable.
    """
    by_text = defaultdict(list)
    for role, rows in roles.items():
        for row in rows:
            by_text[normalize_text(row["text"])].append((role, row["id"], row["label"]))
    collisions = []
    for text, entries in by_text.items():
        if len({e[0] for e in entries}) > 1:
            collisions.append({"text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                               "entries": [{"role": r, "id": i, "label": l} for r, i, l in entries]})
    return {"normalization": "Unicode casefold; remove punctuation; collapse whitespace",
            "cross_role_text_collisions": len(collisions), "collisions": collisions,
            "does_not_detect": ["paraphrase leakage", "pretraining contamination"]}


def _grouped(rows: list[dict], seed: int, purpose: str) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["label"]].append(dict(row))
    for label, group in groups.items():
        random.Random(stable_seed(seed, purpose + ":" + label)).shuffle(group)
    return dict(groups)


def build_roles(clinc: dict, banking_train: list[dict], banking_test: list[dict], seed: int) -> tuple[dict, dict, dict]:
    by_label = _grouped(clinc["train"], seed, "clinc_roles")
    labels = sorted(by_label)
    shuffled = list(labels)
    random.Random(seed).shuffle(shuffled)
    stable = set(shuffled[:120])
    novel = set(shuffled[120:])
    roles = {k: [] for k in ["index", "train", "policy_train", "calibration", "tuning", "test", "test_base", "novel_validation", "transfer", "banking_index", "direct_classifier_train"]}
    for label, group in sorted(by_label.items()):
        roles["index"].extend(group[:20])
        if label in stable:
            roles["train"].extend(group[20:80])
            roles["policy_train"].extend(group[80:100])
    oos = _grouped(clinc["oos_train"], seed, "oos_roles")["oos"]
    roles["train"].extend(oos[:200])
    roles["policy_train"].extend(oos[200:])
    for label, group in sorted(_grouped(clinc["val"] + clinc["oos_val"], seed, "validation").items()):
        if label in novel:
            roles["novel_validation"].extend(group)
        else:
            midpoint = len(group) // 2
            roles["calibration"].extend(group[:midpoint])
            roles["tuning"].extend(group[midpoint:])
    roles["test"] = [dict(r) for r in clinc["test"] + clinc["oos_test"]]
    roles["test_base"] = [dict(r) for r in roles["test"] if r["label"] in stable or r["is_oos"]]
    roles["direct_classifier_train"] = [dict(r) for r in clinc["train"] + clinc["oos_train"]]
    for label, group in sorted(_grouped(banking_train, seed, "banking_index").items()):
        if len(group) < 20:
            raise ValueError(f"Insufficient banking index examples for {label}")
        roles["banking_index"].extend(group[:20])
    roles["transfer"] = [dict(r) for r in banking_test]
    catalogs = {"clinc120": sorted(stable), "clinc150": labels,
                "banking77": sorted({r["label"] for r in banking_train})}
    for role, rows in roles.items():
        catalog_id = "banking77" if role in {"transfer", "banking_index"} else ("clinc150" if role in {"test", "novel_validation", "index", "direct_classifier_train"} else "clinc120")
        for row in rows:
            row["split"] = role
            row["catalog_id"] = catalog_id
            row["group"] = "oos" if row["is_oos"] else ("transfer" if row["dataset"] == "banking77" else ("novel" if row["label"] in novel else "stable"))
    return roles, catalogs, {"stable_labels": sorted(stable), "novel_labels": sorted(novel)}


def resolve_collisions(roles: dict[str, list[dict]]) -> tuple[dict, dict]:
    """Preserve every official test row; filter fitting duplicates transparently.

    Evaluation text is used only for exact-deduplication, never ranking or model
    selection. The nominal per-class allocation can shrink and is not refilled.
    Evaluation/evaluation overlap remains flagged; those tests are not independent.
    """
    cleaned = {key: [dict(r) for r in rows] for key, rows in roles.items()}
    evaluation_roles = ["test", "transfer", "novel_validation"]
    reserved = {}
    for role in evaluation_roles:
        for row in roles[role]:
            reserved.setdefault(normalize_text(row["text"]), {"role": role, "id": row["id"]})
    baseline_forbidden = dict(reserved)
    for role in ("calibration", "tuning"):
        for row in roles[role]:
            baseline_forbidden.setdefault(normalize_text(row["text"]), {"role": role, "id": row["id"]})
    removed = []

    def filter_role(role: str, seen: dict) -> list[dict]:
        kept = []
        for row in roles[role]:
            normalized = normalize_text(row["text"])
            if normalized in seen:
                removed.append({"role": role, "id": row["id"], "label": row["label"],
                                "text_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
                                "reason": "normalized_text_duplicate", "matches": seen[normalized]})
            else:
                kept.append(dict(row))
                seen[normalized] = {"role": role, "id": row["id"]}
        return kept

    for role in ["calibration", "tuning", "index", "banking_index", "train", "policy_train"]:
        cleaned[role] = filter_role(role, reserved)
    # This separate practical baseline intentionally overlaps the model's index /
    # fitting data. Only evaluation/selection collisions and own duplicates go.
    cleaned["direct_classifier_train"] = filter_role("direct_classifier_train", baseline_forbidden)
    evaluation_audit = leakage_audit({r: roles[r] for r in evaluation_roles})
    fitting_audit = leakage_audit({r: cleaned[r] for r in ["index", "banking_index", "train", "policy_train", "calibration", "tuning", *evaluation_roles]})
    invalid = [c for c in fitting_audit["collisions"] if not all(e["role"] in evaluation_roles for e in c["entries"])]
    report = {"policy": "Keep all official test rows; remove normalized duplicates from fitting/index and direct-classifier rows; no refill",
              "removed_rows": removed, "removed_counts": dict(Counter(r["role"] for r in removed)),
              "evaluation_overlap": evaluation_audit, "remaining_fitting_collisions": len(invalid),
              "nominal_counts": {k: len(v) for k, v in roles.items()},
              "actual_counts": {k: len(v) for k, v in cleaned.items()}}
    if invalid:
        raise ValueError("Duplicate resolution left fitting-data leakage")
    return cleaned, report


def prepare(data_dir: str | Path = "data", seed: int = 42, backend: str = "tfidf", stress_limit: int = 1000,
            device: str = "cpu", download_only: bool = False, first_k: int = 1, expanded_k: int = 5,
            processed_subdir: str = "processed") -> dict:
    if not 1 <= first_k < expanded_k <= 20:
        raise ValueError("Require 1 <= first_k < expanded_k <= 20")
    if Path(processed_subdir).name != processed_subdir or processed_subdir in {".", "..", ""}:
        raise ValueError("processed_subdir must be one directory name")
    if stress_limit < 0:
        raise ValueError("stress_limit cannot be negative")
    data_dir = Path(data_dir)
    download_dir, processed = data_dir / "downloads", data_dir / processed_subdir
    sources = download_sources(download_dir)
    if download_only:
        return {"sources": sources}
    clinc = load_clinc(download_dir / "clinc_oos_plus.json")
    banking_train = load_banking(download_dir / "banking77_train.csv", "train")
    banking_test = load_banking(download_dir / "banking77_test.csv", "test")
    if len(banking_train) != 10003 or len(banking_test) != 3080:
        raise ValueError("Unexpected BANKING77 split sizes")
    roles, catalogs, label_partition = build_roles(clinc, banking_train, banking_test, seed)
    audit_roles = {k: roles[k] for k in ["index", "train", "policy_train", "calibration", "tuning", "test", "novel_validation", "banking_index", "transfer"]}
    audit = leakage_audit(audit_roles)
    roles, resolution = resolve_collisions(roles)
    audit["resolution"] = resolution
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "leakage_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    for role, expected_labels in [("index", set(catalogs["clinc150"])), ("banking_index", set(catalogs["banking77"]))]:
        if {r["label"] for r in roles[role]} != expected_labels:
            raise ValueError(f"Deduplication removed all index examples for at least one intent in {role}")
    indices = {"clinc120": [r for r in roles["index"] if r["label"] in catalogs["clinc120"]],
               "clinc150": roles["index"], "banking77": roles["banking_index"]}
    if backend == "tfidf":
        retrievers = {key: TFIDFRetriever(rows) for key, rows in indices.items()}
    elif backend == "minilm":
        retrievers = {key: MiniLMRetriever(rows, cache_dir=data_dir / "embedding_cache", device=device) for key, rows in indices.items()}
    else:
        raise ValueError("backend must be tfidf or minilm")
    counts, recovery, outputs = {}, {}, {}
    for role, rows in roles.items():
        if role in {"index", "banking_index", "direct_classifier_train"}:
            write_jsonl(processed / f"{role}.jsonl", rows)
            continue
        print(f"Retrieving {role}: {len(rows)} examples with {backend}", flush=True)
        ranked = retrievers[rows[0]["catalog_id"]].retrieve_many([r["text"] for r in rows], k=20) if rows else []
        records = [attach_candidates(row, candidates, seed, first_k, expanded_k) for row, candidates in zip(rows, ranked)]
        write_jsonl(processed / f"{role}.jsonl", records)
        outputs[role] = records
        counts[role] = {"total": len(records), "groups": dict(Counter(r["group"] for r in records))}
        recovery[role] = dict(Counter(r["retrieval_situation"] for r in records))
    for role in ("test", "test_base", "transfer"):
        selected = list(outputs[role])
        random.Random(stable_seed(seed, role + ":stress")).shuffle(selected)
        for kind in ("drop_gold", "shuffle"):
            stress = []
            for row in (selected if stress_limit else []):
                changed = make_intervention(row, kind, seed)
                if changed is not None:
                    stress.append(changed)
                if len(stress) >= stress_limit:
                    break
            write_jsonl(processed / f"{role}_{kind}.jsonl", stress)
            counts[f"{role}_{kind}"] = {"total": len(stress), "intervention": kind}
    catalog_records = {key: [{"id": f"{'banking77' if key == 'banking77' else 'clinc150'}:{label}",
                             "label": label, "description": candidate_description(label)} for label in labels]
                       for key, labels in catalogs.items()}
    (processed / "catalogs.json").write_text(json.dumps(catalog_records, indent=2) + "\n", encoding="utf-8")
    manifest = {"schema_version": 2, "seed": seed, "sources": sources, "label_partition": label_partition,
                "catalog_sizes": {k: len(v) for k, v in catalogs.items()}, "counts": counts,
                "role_counts": {k: len(v) for k, v in roles.items()}, "natural_recovery_counts": recovery,
                "retrieval": {"backend": backend, "first_k": first_k, "expanded_k": expanded_k,
                              "profile_id": f"{backend}-top{first_k}-to{expanded_k}",
                              "diagnostic_case_names": ["present_initially", "recoverable_after_expansion", "missing_after_expansion", "true_oos"],
                              "encoder_model": MINILM_MODEL if backend == "minilm" else None,
                              "encoder_revision": MINILM_REVISION if backend == "minilm" else None,
                              "fit_data": "index rows only; max example similarity per unique intent"},
                "fit_roles": ["train", "policy_train", "calibration", "tuning"],
                "profile_design_note": "Primary top1->5 selected from policy-training retrieval counts before backbone training: 243 recoverable versus 9 for top5->20; top3->10 had 36. This choice did not use novel/test model outcomes. Original top5->20 remains saturation control.",
                "evaluation_roles": {"test_base": "120-intent same-catalog natural test",
                                     "test": "150-intent catalog-addition natural test",
                                     "novel_validation": "audit only; forbidden for tuning",
                                     "transfer": "77-intent BANKING77 label/exemplar-exposed catalog transfer; no OOS or adaptation"},
                "leakage_audit": {"initial_cross_role_text_collisions": audit["cross_role_text_collisions"],
                                  "remaining_fitting_collisions": resolution["remaining_fitting_collisions"],
                                  "evaluation_overlap_count": resolution["evaluation_overlap"]["cross_role_text_collisions"],
                                  "removed_counts": resolution["removed_counts"], "details": "leakage_audit.json"},
                "limitations": ["TF-IDF is a lexical diagnostic; use pinned MiniLM for planned primary retrieval" if backend == "tfidf" else "Frozen MiniLM; no claim of optimal retrieval",
                                "All public data may have appeared in backbone pretraining",
                                "novel intent labels and index examples are exposed at catalog update; not zero-shot",
                                "Direct classifier uses original training rows including novel intents, after removing duplicate/evaluation-overlap text; stronger-data practical comparator",
                                "OOS priors differ across original and constructed splits",
                                "Test stress variants are separate artifacts; bootstrap by parent_id"]}
    manifest["files"] = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(processed.glob("*")) if p.is_file() and p.name != "manifest.json"}
    (processed / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=["tfidf", "minilm"], default="tfidf")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stress-limit", type=int, default=1000)
    parser.add_argument("--first-k", type=int, default=1)
    parser.add_argument("--expanded-k", type=int, default=5)
    parser.add_argument("--processed-subdir", default="processed")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args(argv)
    result = prepare(args.data_dir, args.seed, args.backend, args.stress_limit, args.device, args.download_only,
                     args.first_k, args.expanded_k, args.processed_subdir)
    print(json.dumps({"role_counts": result.get("role_counts"), "natural_recovery_counts": result.get("natural_recovery_counts"), "source_files": len(result["sources"])}, indent=2))


if __name__ == "__main__":
    main()
