import numpy as np

from recovery_router.baselines import (choose_utility_threshold, fit_class_temperature,
                                      routing_outcomes, scale_class_probabilities, nearest_features)


def test_temperature_preserves_classes_and_normalization():
    probabilities = np.array([[.8, .2], [.3, .7]])
    result = fit_class_temperature(probabilities, np.array([0, 1]), np.ones(2))
    scaled = scale_class_probabilities(probabilities, result["temperature"])
    assert np.allclose(scaled.sum(axis=1), 1)
    assert np.array_equal(scaled.argmax(axis=1), probabilities.argmax(axis=1))
    assert result["weighted_nll_after"] <= result["weighted_nll_before"]


def test_threshold_excludes_explicit_oos_and_selects_utility():
    rows = [{"label": "a"}, {"label": "b"}, {"label": "oos"}]
    result = choose_utility_threshold(rows, ["a", "a", "oos"], np.array([.95, .2, .99]))
    assert .2 < result["threshold"] <= .95
    assert np.isclose(result["weighted_tuning_utility"], (1 - .2 - .2) / 3)


def test_oos_nearest_acceptance_is_wrong_route_not_correct_oos():
    rows = [{"id": "x", "label": "oos", "is_oos": True, "group": "oos"}]
    accepted = routing_outcomes(rows, ["intent"], np.array([.9]), .8)[0]
    assert accepted["accepted"] and not accepted["correct"] and accepted["utility"] == -5
    deferred = routing_outcomes(rows, ["oos"], np.array([.9]), .8)[0]
    assert not deferred["accepted"] and deferred["utility"] == -.2


def test_nearest_gate_does_not_peek_at_expanded_scores_in_top_one_profile():
    predictions, features = nearest_features([{"candidates": [{"label": "a", "score": .8}],
                                               "expanded_candidates": [{"label": "b", "score": .79}]}])
    assert predictions == ["a"]
    assert features.tolist() == [[.8, 0.0]]
