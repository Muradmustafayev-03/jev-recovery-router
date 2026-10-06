"""Cheap direct classifiers and retrieval baselines with explicit fit data."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline


def build_direct_classifier(seed: int = 42, estimator: str = "sgd") -> Pipeline:
    classifier = (SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=50, tol=1e-3,
                                random_state=seed, n_jobs=1) if estimator == "sgd" else
                  LogisticRegression(C=4.0, max_iter=200, solver="lbfgs", random_state=seed))
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ])),
        ("classifier", classifier),
    ])


def fit_direct_classifier(rows: list[dict], seed: int = 42):
    """Caller records exactly which rows were passed; no implicit data loading."""
    model = build_direct_classifier(seed)
    model.fit([r["text"] for r in rows], [r["label"] for r in rows])
    return model


def fit_embedding_classifier(embeddings: np.ndarray, labels: list[str], seed: int = 42):
    """Optional baseline for externally computed frozen sentence embeddings."""
    model = LogisticRegression(C=4.0, max_iter=200, random_state=seed)
    return model.fit(embeddings, labels)


def nearest_candidate(row: dict, expanded: bool = False) -> dict:
    candidates = row["expanded_candidates" if expanded else "candidates"]
    return max(candidates, key=lambda c: (c["score"], c["label"]))


def direct_predictions(model, rows: list[dict]) -> list[dict]:
    probabilities = model.predict_proba([r["text"] for r in rows])
    classes = model.classes_
    return [{"id": row["id"], "prediction": str(classes[p.argmax()]),
             "confidence": float(p.max()), "is_oos_prediction": str(classes[p.argmax()]) == "oos"}
            for row, p in zip(rows, probabilities)]


def scale_class_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    from scipy.special import softmax
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return softmax(np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature, axis=1)


def fit_class_temperature(probabilities: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> dict:
    from scipy.optimize import minimize_scalar
    def objective(log_t):
        p = scale_class_probabilities(probabilities, float(np.exp(log_t)))
        return float(np.average(-np.log(np.clip(p[np.arange(len(p)), targets], 1e-12, 1)), weights=weights))
    fitted = minimize_scalar(objective, bounds=(-2.3, 2.3), method="bounded")
    return {"temperature": float(np.exp(fitted.x)), "weighted_nll_before": objective(0.0),
            "weighted_nll_after": float(fitted.fun), "method": "full_class_temperature_scaling"}


def choose_utility_threshold(rows: list[dict], predictions: list[str], confidence: np.ndarray,
                             weights: np.ndarray | None = None) -> dict:
    """A declared grid selected exclusively on tuning labels. Conservative ties."""
    gold = np.array([r["label"] for r in rows])
    pred = np.array(predictions)
    correct = pred == gold
    weights = np.ones(len(rows)) if weights is None else weights
    best = None
    for threshold in np.linspace(0, 1, 101):
        accept = (pred != "oos") & (confidence >= threshold)
        utility = np.where(accept, np.where(correct, 1.0, -5.0), -0.2)
        score = float(np.average(utility, weights=weights))
        if best is None or score >= best["weighted_tuning_utility"]:
            best = {"threshold": float(threshold), "weighted_tuning_utility": score,
                    "grid": "0.00..1.00 inclusive in increments of 0.01; conservative ties"}
    return best


def nearest_features(rows: list[dict]) -> tuple[list[str], np.ndarray]:
    predictions, features = [], []
    for row in rows:
        ranked = sorted(row["candidates"], key=lambda c: (-c["score"], c["label"]))
        predictions.append(ranked[0]["label"])
        # Top-one exposes no runner-up score: do not peek at expanded retrieval.
        gap = ranked[0]["score"] - ranked[1]["score"] if len(ranked) > 1 else 0.0
        features.append([ranked[0]["score"], gap])
    return predictions, np.asarray(features)


def routing_outcomes(rows: list[dict], predictions: list[str], confidence: np.ndarray,
                     threshold: float, force_route: bool = False) -> list[dict]:
    result = []
    for row, pred, probability in zip(rows, predictions, confidence):
        accepted = pred != "oos" and (force_route or float(probability) >= threshold)
        correct = pred == row["label"] and not row["is_oos"]
        result.append({"id": row["id"], "parent_id": row.get("parent_id", row["id"]),
                       "group": row["group"], "recovery_case": row.get("retrieval_situation", "unknown"),
                       "intervention": row.get("intervention", "natural"), "is_oos": row["is_oos"],
                       "label": row["label"], "prediction": pred, "confidence": float(probability),
                       "accepted": bool(accepted), "correct": bool(correct), "expanded": False,
                       "utility": (1.0 if correct else -5.0) if accepted else -0.2,
                       "accounted_ms": None})
    return result


def run_classical(data_dir, output_dir, backend="tfidf", seed=42, target_oos_prior=2/11):
    """Fit fixed classical configurations and run sealed tests exactly once.

    This runner is a CPU diagnostic. It never trains a Jev-inspired backbone or
    an RL actor. BANKING77 fits only supplied catalog exemplars and has no test
    calibration. Stored retrieval scores must match the requested backend.
    """
    import json
    import time
    import warnings
    from pathlib import Path
    import joblib
    from sklearn.metrics import recall_score
    from threadpoolctl import threadpool_limits
    from contextlib import nullcontext
    from .calibration import mixture_weights
    from .data import read_jsonl, write_jsonl, sha256
    from .metrics import grouped_summary
    from .retrieval import MINILM_MODEL, MINILM_REVISION, MiniLMRetriever

    start = time.perf_counter()
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["retrieval"]["backend"] != backend:
        raise ValueError("Requested baseline backend differs from prepared retrieval backend; regenerate or select matching data")
    fit_rows = read_jsonl(data_dir / "direct_classifier_train.jsonl")
    calibration = read_jsonl(data_dir / "calibration.jsonl")
    tuning = read_jsonl(data_dir / "tuning.jsonl")
    banking_index = read_jsonl(data_dir / "banking_index.jsonl")
    encoder = None
    if backend == "minilm":
        encoder = MiniLMRetriever(banking_index, cache_dir=data_dir.parent / "embedding_cache", device="cpu")

    def features(rows):
        texts = [r["text"] for r in rows]
        return encoder._encode(texts) if encoder is not None else texts

    warning_messages = []
    def limit_threads():
        try:
            return threadpool_limits(limits=4)
        except OSError as exc:
            # Some Windows library-enumeration paths lose non-ASCII characters.
            # Thread limiting is optional; numerical work can use default pools.
            warning_messages.append(f"Could not limit BLAS threads: {type(exc).__name__}; using default pools")
            return nullcontext()
    fit_started = time.perf_counter()
    with limit_threads(), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if backend == "tfidf":
            direct = fit_direct_classifier(fit_rows, seed)
            bank_direct = fit_direct_classifier(banking_index, seed)
        else:
            direct = fit_embedding_classifier(features(fit_rows), [r["label"] for r in fit_rows], seed)
            bank_direct = fit_embedding_classifier(features(banking_index), [r["label"] for r in banking_index], seed)
        warning_messages.extend(str(w.message) for w in captured)
    fit_seconds = time.perf_counter() - fit_started
    print(f"Fitted {backend} direct classifiers in {fit_seconds:.1f}s", flush=True)
    joblib.dump(direct, output_dir / "clinc_direct.joblib")
    joblib.dump(bank_direct, output_dir / "banking_catalog_classifier.joblib")
    classes = np.asarray(direct.classes_)
    class_index = {label: i for i, label in enumerate(classes)}
    with limit_threads():
        calibration_probs = direct.predict_proba(features(calibration))
        tuning_probs = direct.predict_proba(features(tuning))
    weights = mixture_weights(calibration, target_oos_prior)
    direct_calibration = fit_class_temperature(calibration_probs, np.array([class_index[r["label"]] for r in calibration]), weights)
    tuning_scaled = scale_class_probabilities(tuning_probs, direct_calibration["temperature"])
    tuning_pred = classes[tuning_scaled.argmax(axis=1)].tolist()
    direct_rule = choose_utility_threshold(tuning, tuning_pred, tuning_scaled.max(axis=1), mixture_weights(tuning, target_oos_prior))
    near_cal_pred, near_cal_features = nearest_features(calibration)
    near_cal_correct = np.array([int(p == r["label"] and not r["is_oos"]) for p, r in zip(near_cal_pred, calibration)])
    nearest_calibrator = LogisticRegression(C=1.0, max_iter=200, random_state=seed)
    nearest_calibrator.fit(near_cal_features, near_cal_correct, sample_weight=weights)
    near_tune_pred, near_tune_features = nearest_features(tuning)
    near_confidence = nearest_calibrator.predict_proba(near_tune_features)[:, 1]
    nearest_rule = choose_utility_threshold(tuning, near_tune_pred, near_confidence, mixture_weights(tuning, target_oos_prior))
    joblib.dump(nearest_calibrator, output_dir / "nearest_correctness_calibrator.joblib")
    fitting = {"seed": seed, "backend": backend, "n_clinc_training": len(fit_rows), "n_banking_catalog_examples": len(banking_index),
               "n_calibration": len(calibration), "n_tuning": len(tuning), "target_oos_prior_assumed": target_oos_prior,
               "direct_estimator": "word+char TF-IDF; SGD log_loss alpha=1e-5 max_iter=50 tol=1e-3" if backend == "tfidf" else "frozen MiniLM; logistic regression C=4 max_iter=200",
               "encoder": {"model": MINILM_MODEL, "revision": MINILM_REVISION} if backend == "minilm" else None,
               "direct_calibration": direct_calibration, "direct_rule": direct_rule,
               "nearest_calibration": "logistic regression on top-example similarity and exposed top-two intent gap (zero for top-one profile); correctness target",
               "nearest_rule": nearest_rule,
               "utility": {"correct": 1, "wrong": -5, "handoff": -0.2},
               "calibration_assumption": "OOS reweighting assumes sufficiently stable within-group distributions; novel classes absent from calibration",
               "banking_rule": "No BANKING77 test or validation labels fit calibration; classifier shown always-route and fixed raw-probability 0.8 gate",
               "test_selection": "All fit/calibration/threshold choices fixed before test prediction; no test-based hyperparameter search"}
    # Persist fitting choices before loading any test file.
    (output_dir / "fitting.json").write_text(json.dumps(fitting, indent=2) + "\n", encoding="utf-8")
    results = {"status": "MEASURED CPU CLASSICAL BASELINES ONLY; no Jev-inspired backbone or RL training",
               "test_data_hashes":{s:sha256(data_dir/(s+".jsonl")) for s in ("test_base","test","transfer")},
               "fitting": fitting, "source_data_manifest_sha256": sha256(data_dir / "manifest.json"),
               "fit_seconds": fit_seconds, "warnings": warning_messages, "evaluations": {},
               "limitations": ["Direct CLINC classifier has supervised labels for all 150 intents; stronger labeled-data baseline than proposed 120-intent model",
                               "Base test also permits the direct classifier to output any of its 150 intents; mismatched catalog declared, not hidden",
                               "BANKING77 classifier is supervised on provided catalog exemplars, not a zero-shot model",
                               "Calibration only uses stable CLINC classes and cannot validate novel-catalog or BANKING77 calibration",
                               "Exact duplicate removal used test text but never test outcomes or candidate ranking",
                               "Three overlapping evaluation text groups mean datasets are not completely independent",
                               "TF-IDF is a CPU lexical diagnostic, not the primary semantic-retrieval experiment" if backend == "tfidf" else "Frozen MiniLM retrieval and classifiers are classical baselines, not the adapted model",
                               "Single seed, fixed assumed costs; observed utility is not measured business savings",
                               "Batch CPU timings exclude online queuing and do not imply serving p95 latency"]}

    def record(name, rows, pred, confidence, threshold, force_route=False, probabilities=None, class_names=None, predict_seconds=None):
        outcomes = routing_outcomes(rows, pred, confidence, threshold, force_route)
        summary = grouped_summary(outcomes)
        summary["overall"]["classification_macro_accuracy"] = float(recall_score([r["label"] for r in rows], pred,
                                                                                 labels=sorted({r["label"] for r in rows}), average="macro", zero_division=0))
        labels = sorted({r["label"] for r in rows if not r["is_oos"]})
        summary["overall"]["in_scope_macro_correct_route_rate"] = float(np.mean([
            np.mean([o["accepted"] and o["correct"] for o in outcomes if o["label"] == label]) for label in labels]))
        if any(r["is_oos"] for r in rows):
            summary["overall"]["utility_at_declared_oos_prior"] = float(np.average([o["utility"] for o in outcomes], weights=mixture_weights(rows, target_oos_prior)))
        if probabilities is not None:
            yi = {c: i for i, c in enumerate(class_names)}
            targets = np.array([yi[r["label"]] for r in rows])
            p_target = probabilities[np.arange(len(rows)), targets]
            summary["probabilities"] = {"full_class_nll": float(-np.log(np.clip(p_target, 1e-12, 1)).mean()),
                                         "full_class_brier_sum": float((np.square(probabilities).sum(axis=1) - 2 * p_target + 1).mean())}
        summary["prediction_batch_seconds"] = predict_seconds
        summary["threshold"] = None if force_route else threshold
        write_jsonl(output_dir / f"{name}.outcomes.jsonl", outcomes)
        results["evaluations"][name] = summary

    for split in ["test_base", "test", "transfer"]:
        rows = read_jsonl(data_dir / f"{split}.jsonl")
        begin = time.perf_counter()
        classifier = bank_direct if split == "transfer" else direct
        with limit_threads():
            raw = classifier.predict_proba(features(rows))
        predict_seconds = time.perf_counter() - begin
        if split == "transfer":
            probabilities = raw
            threshold = 0.8
            suffix = "raw_fixed_gate"
        else:
            probabilities = scale_class_probabilities(raw, direct_calibration["temperature"])
            threshold = direct_rule["threshold"]
            suffix = "calibrated_gate"
        pred = np.asarray(classifier.classes_)[probabilities.argmax(axis=1)].tolist()
        confidence = probabilities.max(axis=1)
        record(f"{split}_direct_{suffix}", rows, pred, confidence, threshold,
               probabilities=probabilities, class_names=classifier.classes_, predict_seconds=predict_seconds)
        record(f"{split}_direct_always_route", rows, pred, confidence, 0.0, force_route=True,
               predict_seconds=predict_seconds)
        near_pred, near_features = nearest_features(rows)
        near_conf = nearest_calibrator.predict_proba(near_features)[:, 1]
        record(f"{split}_nearest_frozen_gate", rows, near_pred, near_conf, nearest_rule["threshold"])
        record(f"{split}_nearest_always_route", rows, near_pred, near_conf, 0.0, force_route=True)
        print(f"Evaluated {split}: {len(rows)} examples", flush=True)
    results["elapsed_seconds"] = time.perf_counter() - start
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Measured CPU classical baselines", "", results["status"], "",
             "These results do not measure a trained Jev-inspired model or RL policy. Hyperparameters and calibration were fixed before tests.", "",
             f"Backend: **{backend}**. Fit time: {fit_seconds:.1f}s; total elapsed: {results['elapsed_seconds']:.1f}s.", "",
             "| Evaluation | Correct route | Wrong route | Coverage | Accepted error | OOS false acceptance | Utility |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, result in results["evaluations"].items():
        m = result["overall"]
        values = ["—" if m[k] is None else f"{m[k]:.4f}" for k in ["correct_route_rate", "wrong_route_rate", "coverage", "accepted_error", "oos_false_acceptance", "mean_utility"]]
        lines.append("| " + name + " | " + " | ".join(values) + " |")
    lines += ["", "Limitations:", ""] + ["- " + text for text in results["limitations"]]
    lines += ["", "The nearest frozen gate transfers its CLINC correctness calibration to BANKING77 unchanged; calibration there is unvalidated. Always-route variants provide the corresponding accuracy reference. Direct always-route still hands off an explicit OOS prediction.", "",
              "Detailed class/group metrics, NLL/Brier where applicable, training warnings and batch timings are in `results.json`. Per-query outcomes support paired analysis. `fitting.json` records choices before test loading."]
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Run bounded CPU classical baselines; no Jev/RL training")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default="runs/classical")
    parser.add_argument("--backend", choices=["tfidf", "minilm"], default="tfidf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-oos-prior", type=float, default=2/11)
    args = parser.parse_args(argv)
    run_classical(args.data_dir, args.output, args.backend, args.seed, args.target_oos_prior)


if __name__ == "__main__":
    main()
