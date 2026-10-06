"""Small strict IO helpers; experiment outputs are atomic and carry hashes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate row IDs in {path}")
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def join_predictions(rows, predictions):
    by_id = {row["id"]: row for row in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("Duplicate prediction IDs")
    if set(by_id) != {row["id"] for row in rows}:
        raise ValueError("Prediction IDs do not exactly match data; refusing an incomplete comparison")
    return [(row, by_id[row["id"]]) for row in rows]


def read_verified_predictions(data_path, prediction_path, expected_fingerprint=None):
    """Refuse stale, partial, reordered-candidate or mixed-provider forecasts."""
    from .calibration import checked_probs
    data_path, prediction_path = Path(data_path), Path(prediction_path)
    rows, predictions = read_jsonl(data_path), read_jsonl(prediction_path)
    metadata = read_json(prediction_path.with_suffix(prediction_path.suffix + ".meta.json"))
    identity = metadata.get("model_fingerprint")
    if not isinstance(identity, str) or not identity:
        raise ValueError("Prediction provider fingerprint is missing")
    if expected_fingerprint is not None and identity != expected_fingerprint:
        raise ValueError("Prediction provider differs from the sealed fitting provider")
    if metadata.get("data_sha256") != sha256_file(data_path):
        raise ValueError("Prediction data fingerprint differs from the supplied dataset")
    if metadata.get("records_with_errors", 0):
        raise ValueError("Prediction metadata reports errors")
    for row, prediction in join_predictions(rows, predictions):
        if prediction.get("model_fingerprint") != identity:
            raise ValueError("Mixed or missing per-record provider fingerprint")
        for stage, key in (("first", "candidates"), ("expanded", "expanded_candidates")):
            expected_ids = [candidate["id"] for candidate in row[key]]
            if prediction.get(stage + "_candidate_ids") != expected_ids:
                raise ValueError(f"Candidate order differs for {row['id']} at {stage}")
            if stage + "_error" in prediction or stage + "_probs" not in prediction:
                raise ValueError(f"Incomplete prediction for {row['id']} at {stage}")
            checked_probs(prediction[stage + "_probs"], len(expected_ids))
    return rows, predictions, metadata
