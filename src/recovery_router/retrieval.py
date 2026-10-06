"""Small, reproducible catalog retrieval; no test labels participate in ranking."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from collections import defaultdict
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
_ENCODERS = {}


def stable_seed(seed: int, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big")


def candidate_description(label: str) -> str:
    """Train-independent descriptions, deliberately without teacher-generated text."""
    return label.replace("_", " ")


class TFIDFRetriever:
    """Word + character TF-IDF nearest examples, collapsed to unique intents.

    Fitting is restricted to the explicit retrieval-index records. Each intent's
    score is its highest-scoring example. Stable tie-breaking uses the label.
    """

    def __init__(self, index_rows: Iterable[dict]):
        from scipy.sparse import hstack
        from sklearn.preprocessing import normalize

        self.rows = list(index_rows)
        if not self.rows:
            raise ValueError("The retrieval index must not be empty")
        self.word = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        texts = [r["text"] for r in self.rows]
        self.matrix = normalize(hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]))
        self.labels = np.array([r["label"] for r in self.rows])
        self.unique_labels = sorted(set(self.labels))
        self.label_indices = {label: np.where(self.labels == label)[0] for label in self.unique_labels}

    def retrieve_many(self, texts: list[str], k: int = 20, batch_size: int = 128) -> list[list[dict]]:
        from scipy.sparse import hstack
        from sklearn.preprocessing import normalize

        output = []
        dataset = self.rows[0]["dataset"]
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            q = normalize(hstack([self.word.transform(batch), self.char.transform(batch)]))
            similarities = (q @ self.matrix.T).toarray()
            for scores in similarities:
                intent_scores = [(label, float(scores[indices].max())) for label, indices in self.label_indices.items()]
                intent_scores.sort(key=lambda x: (-x[1], x[0]))
                output.append([
                    {"id": f"{dataset}:{label}", "label": label,
                     "description": candidate_description(label), "score": round(score, 8)}
                    for label, score in intent_scores[:k]
                ])
        return output

    def retrieve(self, text: str, k: int = 20) -> list[dict]:
        return self.retrieve_many([text], k=k)[0]


class MiniLMRetriever:
    """Pinned frozen semantic retriever. Model files may be downloaded on first use.

    Embedding cache keys include model revision and exact input text. Only index
    rows define the searchable examples; encoder weights are never adapted.
    """

    def __init__(self, index_rows: Iterable[dict], cache_dir: str | Path | None = None, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.rows = list(index_rows)
        if not self.rows:
            raise ValueError("The retrieval index must not be empty")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if device not in _ENCODERS:
            _ENCODERS[device] = SentenceTransformer(MINILM_MODEL, revision=MINILM_REVISION, device=device)
        self.encoder = _ENCODERS[device]
        self.matrix = self._encode([r["text"] for r in self.rows])
        labels = np.array([r["label"] for r in self.rows])
        self.label_indices = {label: np.where(labels == label)[0] for label in sorted(set(labels))}

    def _encode(self, texts: list[str]) -> np.ndarray:
        import json
        key = hashlib.sha256(json.dumps([MINILM_REVISION, texts], ensure_ascii=False).encode()).hexdigest()
        path = self.cache_dir / f"{key}.npy" if self.cache_dir else None
        if path and path.exists():
            return np.load(path, allow_pickle=False)
        result = self.encoder.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                                     show_progress_bar=False)
        if path:
            np.save(path, result, allow_pickle=False)
        return result

    def retrieve_many(self, texts: list[str], k: int = 20, batch_size: int = 256) -> list[list[dict]]:
        vectors = self._encode(texts)
        output = []
        dataset = self.rows[0]["dataset"]
        for start in range(0, len(vectors), batch_size):
            for scores in vectors[start:start + batch_size] @ self.matrix.T:
                ranked = [(label, float(scores[ix].max())) for label, ix in self.label_indices.items()]
                ranked.sort(key=lambda x: (-x[1], x[0]))
                output.append([{"id": f"{dataset}:{label}", "label": label,
                                "description": candidate_description(label), "score": round(score, 8)}
                               for label, score in ranked[:k]])
        return output

    def retrieve(self, text: str, k: int = 20) -> list[dict]:
        return self.retrieve_many([text], k=k)[0]


def attach_candidates(row: dict, ranked: list[dict], seed: int = 42,
                      first_k: int = 1, expanded_k: int = 5) -> dict:
    if not 1 <= first_k < expanded_k <= len(ranked):
        raise ValueError("Require 1 <= first_k < expanded_k <= ranked candidate count")
    result = dict(row)
    shown, expanded = [dict(c) for c in ranked[:first_k]], [dict(c) for c in ranked[:expanded_k]]
    random.Random(stable_seed(seed, row["id"] + ":first")).shuffle(shown)
    random.Random(stable_seed(seed, row["id"] + ":expanded")).shuffle(expanded)
    result.update(candidates=shown, expanded_candidates=expanded,
                  intervention="natural", parent_id=row.get("parent_id", row["id"]))
    result["retrieval_situation"] = retrieval_situation(result)
    return result


def retrieval_situation(row: dict) -> str:
    if row["is_oos"]:
        return "true_oos"
    label = row["label"]
    if label in {c["label"] for c in row["candidates"]}:
        return "present_initially"
    if label in {c["label"] for c in row["expanded_candidates"]}:
        return "recoverable_after_expansion"
    return "missing_after_expansion"


def make_intervention(row: dict, kind: str, seed: int = 42) -> dict | None:
    """Interventions are separate test artifacts, never natural observations.

    drop_gold removes gold only from the first shortlist, substituting a real
    wider-result distractor. It never inserts gold into the expanded retrieval.
    """
    if kind not in {"shuffle", "drop_gold"}:
        raise ValueError(f"Unknown intervention {kind}")
    result = dict(row)
    result["candidates"] = [dict(c) for c in row["candidates"]]
    result["expanded_candidates"] = [dict(c) for c in row["expanded_candidates"]]
    result["parent_id"] = row["parent_id"]
    result["id"] = row["id"] + ":" + kind
    result["intervention"] = kind
    if kind == "drop_gold":
        if row["is_oos"] or row["label"] not in {c["label"] for c in row["candidates"]}:
            return None
        used = {c["label"] for c in row["candidates"]}
        alternatives = [c for c in row["expanded_candidates"] if c["label"] not in used]
        if not alternatives:
            return None
        replacement = max(alternatives, key=lambda c: (c["score"], c["label"]))
        result["candidates"] = [c for c in result["candidates"] if c["label"] != row["label"]] + [dict(replacement)]
    rng = random.Random(stable_seed(seed, result["id"]))
    rng.shuffle(result["candidates"])
    rng.shuffle(result["expanded_candidates"])
    result["retrieval_situation"] = retrieval_situation(result)
    return result
