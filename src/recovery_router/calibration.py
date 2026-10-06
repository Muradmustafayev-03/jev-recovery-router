"""Held-out temperature scaling with explicitly declared OOS mixture weights."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .io import join_predictions


def checked_probs(values, count):
    p = np.asarray(values, dtype=np.float64)
    if p.shape != (count + 1,) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Expected finite nonnegative candidate probabilities followed by NONE")
    if not np.isclose(p.sum(), 1.0, atol=1e-4):
        raise ValueError("Probabilities must include NONE and sum to one")
    return p / p.sum()


def target_index(row, expanded=False):
    candidates = row["expanded_candidates" if expanded else "candidates"]
    labels = [candidate["label"] for candidate in candidates]
    if len(labels) != len(set(labels)):
        raise ValueError("Candidate labels must be unique")
    return labels.index(row["label"]) if row["label"] in labels and not row["is_oos"] else len(labels)


def mixture_weights(rows, target_oos_prior):
    if not 0 <= target_oos_prior <= 1:
        raise ValueError("OOS prior must be in [0, 1]")
    oos = np.array([row["is_oos"] for row in rows], dtype=bool)
    if not len(oos):
        raise ValueError("Empty dataset")
    empirical = oos.mean()
    if empirical in (0, 1):
        if target_oos_prior != empirical:
            raise ValueError("Cannot reweight a missing class into existence")
        return np.ones(len(rows))
    return np.where(oos, target_oos_prior / empirical, (1-target_oos_prior) / (1-empirical))


def scale_probs(values, temperature):
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("Temperature must be finite and positive")
    logits = np.log(np.clip(np.asarray(values, dtype=np.float64), 1e-12, 1)) / temperature
    ex = np.exp(logits - logits.max())
    return ex / ex.sum()


def fit_calibration(rows, predictions, target_oos_prior):
    pairs = join_predictions(rows, predictions)
    weights = mixture_weights(rows, target_oos_prior)
    result = {"method":"temperature_scaling", "target_oos_prior":target_oos_prior,
              "n":len(rows), "n_oos":sum(row["is_oos"] for row in rows),
              "assumption":"Within-group distributions remain sufficiently stable; weighting does not guarantee calibration."}
    for expanded, key in [(False,"first"),(True,"expanded")]:
        probabilities = [checked_probs(pred[key+"_probs"], len(row["expanded_candidates" if expanded else "candidates"])) for row,pred in pairs]
        targets = [target_index(row,expanded) for row in rows]
        def objective(log_temperature):
            t = np.exp(log_temperature)
            losses = [-np.log(max(scale_probs(p,t)[y],1e-12)) for p,y in zip(probabilities,targets)]
            return float(np.average(losses,weights=weights))
        fitted = minimize_scalar(objective,bounds=(-2.3,2.3),method="bounded")
        result[key+"_temperature"] = float(np.exp(fitted.x))
        result[key+"_weighted_nll_before"] = objective(0.0)
        result[key+"_weighted_nll_after"] = float(fitted.fun)
    return result


def apply_calibration(prediction, calibration):
    out = dict(prediction)
    for key in ["first","expanded"]:
        out[key+"_probs"] = scale_probs(prediction[key+"_probs"],calibration[key+"_temperature"]).tolist()
    return out
