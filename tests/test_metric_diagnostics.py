import copy

import pytest

from recovery_router.metrics import (
    accepted_subset_pruning_curve,
    actor_seed_summary,
    expansion_diagnostics,
    grouped_paired_bootstrap,
    grouped_probability_metrics,
)
from recovery_router.policy import DEFER, EXPAND, outcomes
from test_policy_metrics import fixture_rows


def test_deferred_outcomes_retain_potential_selector_correctness():
    rows, predictions = fixture_rows()
    result = outcomes(rows, predictions, [DEFER] * len(rows), [DEFER] * len(rows))
    assert not result[0]["accepted"] and not result[0]["correct"]
    assert result[0]["first_would_be_correct"] and result[0]["expanded_would_be_correct"]
    assert not result[1]["first_would_be_correct"] and result[1]["expanded_would_be_correct"]
    assert not result[2]["first_would_be_correct"] and not result[2]["expanded_would_be_correct"]


def test_expansion_diagnostics_separate_selector_change_from_accepted_result():
    rows, predictions = fixture_rows()
    result = outcomes(rows, predictions, [EXPAND] * len(rows), [DEFER] * len(rows))
    diagnostic = expansion_diagnostics(result)
    assert diagnostic["expanded_n"] == 30
    assert diagnostic["counts"] == {
        "helpful_selector_change": 10,
        "harmful_selector_change": 0,
        "unnecessary_ex_post_both_correct": 10,
        "unresolved_both_wrong": 10,
    }
    assert diagnostic["accepted_recovery_n"] == 0
    assert diagnostic["expanded_then_deferred_n"] == 30
    result[0].update(expanded_would_be_correct=False, accepted=True, correct=False)
    diagnostic = expansion_diagnostics(result)
    assert diagnostic["counts"]["harmful_selector_change"] == 1
    assert diagnostic["accepted_harm_n"] == 1
    assert sum(diagnostic["counts"].values()) == diagnostic["expanded_n"]
    assert expansion_diagnostics([])["fractions_of_expanded"]["helpful_selector_change"] is None
    assert expansion_diagnostics([{"expanded": True}])["status"].startswith("unavailable")


def test_probability_groups_separate_retrieval_absence_from_oos():
    rows, predictions = fixture_rows()
    before = copy.deepcopy(predictions)
    result = grouped_probability_metrics(rows, list(reversed(predictions)), predictions)
    assert result["group"]["stable"]["raw_initial"]["n"] == 20
    assert result["catalog_membership"]["out_of_scope"]["scaled_expanded"]["n"] == 10
    first = result["candidate_membership"]["initial"]
    assert first["gold_present"]["raw_initial"]["n"] == 10
    assert first["in_scope_gold_absent"]["scaled_initial"]["n"] == 10
    assert first["out_of_scope"]["raw_expanded"]["n"] == 10
    expanded = result["candidate_membership"]["expanded"]
    assert expanded["gold_present"]["scaled_expanded"]["n"] == 20
    assert expanded["in_scope_gold_absent"]["scaled_initial"] == {"n": 0}
    assert first["gold_present"]["raw_initial"]["ece_bins"] == 10
    assert predictions == before


def test_pruning_curve_cannot_claim_coverage_beyond_original_accepts():
    rows = [{"accepted": False, "confidence": .999, "correct": False},
            {"accepted": True, "confidence": .9, "correct": True},
            {"accepted": True, "confidence": .8, "correct": False},
            {"accepted": False, "confidence": .1, "correct": False}]
    result = accepted_subset_pruning_curve(rows)
    assert result["original_policy_coverage"] == .5
    assert max(p["coverage"] for p in result["points"]) == .5
    assert result["points"][-1]["accepted_error"] == .5
    assert "not a full risk-coverage curve" in result["description"]
    assert accepted_subset_pruning_curve([])["points"][-1]["accepted_error"] is None


def test_seed_summary_is_actor_only_and_sample_std_is_undefined_for_one_seed():
    methods = {"rl_seed42": [{"utility": 0.}, {"utility": 2.}],
               "rl_seed43": [{"utility": 1.}, {"utility": 3.}],
               "exact_seed42": [{"utility": -.2}],
               "fixed_cost_seed42": [{"utility": 999.}]}
    result = actor_seed_summary(methods)
    assert result["rl"]["n_actor_seeds"] == 2
    assert result["rl"]["mean_utility_across_actor_seeds"] == 1.5
    assert result["rl"]["sample_std_across_actor_seeds"] == pytest.approx(2 ** -.5)
    assert result["exact"]["sample_std_across_actor_seeds"] is None
    assert "Not backbone-training uncertainty" in result["rl"]["scope"]


def test_group_bootstrap_keeps_paired_ids_and_parent_clusters():
    left = [{"id": str(i), "parent_id": "parent" + str(i // 2), "group": "stable",
             "recovery_case": "recoverable_candidate", "intervention": "natural", "utility": 1.}
            for i in range(4)]
    right = [dict(r, utility=.5) for r in reversed(left)]
    result = grouped_paired_bootstrap(left, right, samples=20)
    metric = result["group"]["stable"]
    assert metric["n"] == 4 and metric["difference"] == .5
    assert metric["ci95"] == [.5, .5]
    assert metric["resampling_unit"] == "parent_query"
    assert metric["analysis_status"].startswith("exploratory")
    with pytest.raises(ValueError):
        grouped_paired_bootstrap(left, right[:-1], samples=20)
