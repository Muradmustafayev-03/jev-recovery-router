import copy
import numpy as np
import pytest
import torch

from recovery_router.calibration import checked_probs, fit_calibration, apply_calibration, target_index, mixture_weights
from recovery_router.policy import Actor, Costs, COMMIT, EXPAND, DEFER, state_features, commit_result, run_policy, train_actor, save_actor, load_actor, fit_full_feedback
from recovery_router.metrics import summarize, probability_metrics, paired_bootstrap


def fixture_rows():
    rows=[]; predictions=[]
    for i in range(30):
        missing=i%3==1; oos=i%3==2
        candidates=[{"id":"a","label":"alpha","description":"alpha request","score":.7},
                    {"id":"b","label":"beta","description":"beta request","score":.6}]
        expanded=candidates+[{"id":"c","label":"gamma","description":"gamma request","score":.5}]
        rows.append({"id":str(i),"parent_id":str(i),"text":"query","label":"oos" if oos else ("gamma" if missing else "alpha"),
                     "is_oos":oos,"candidates":candidates,"expanded_candidates":expanded,"group":"oos" if oos else "stable","intervention":"natural"})
        predictions.append({"id":str(i),"first_probs":[.9,.04,.06] if not missing and not oos else [.15,.1,.75],
                            "expanded_probs":[.85,.03,.02,.1] if not missing and not oos else ([.04,.04,.86,.06] if missing else [.05,.05,.05,.85]),
                            "first_ms":1.,"expanded_ms":2.})
    return rows,predictions


def test_none_mass_is_not_renormalized():
    rows,preds=fixture_rows(); preds[0]["first_probs"]=[.2,.1,.7]
    assert commit_result(rows[0],preds[0])[1]==pytest.approx(.2)
    out=run_policy("never_expand",rows,preds)
    assert not out[0]["accepted"]


def test_first_state_has_no_future_or_gold_features():
    rows,preds=fixture_rows(); x=state_features(rows[0],preds[0])
    altered=copy.deepcopy(rows[0]); altered.update(label="secret",is_oos=True)
    altered["expanded_candidates"]=[{"label":"secret"}]
    prediction=copy.deepcopy(preds[0]); prediction["expanded_probs"]=[.999,.001]
    np.testing.assert_array_equal(x,state_features(altered,prediction))


def test_two_stage_rewards_and_no_impossible_actions():
    rows,preds=fixture_rows()
    outcomes=run_policy("oracle_upper_bound",rows,preds)
    assert outcomes[0]["utility"]==pytest.approx(1)
    assert outcomes[1]["expanded"] and outcomes[1]["utility"]==pytest.approx(.95)
    assert outcomes[2]["utility"]==pytest.approx(-.2)
    actor=Actor()
    x=np.stack([state_features(rows[0],preds[0],True)])
    assert actor(torch.from_numpy(x))[0][0,EXPAND]<-1e20


def test_calibration_preserves_valid_distribution_and_uses_labels_only_when_fitting():
    rows,preds=fixture_rows()
    fitted=fit_calibration(rows,preds,1/3)
    p=apply_calibration(preds[0],fitted)
    assert sum(p["first_probs"])==pytest.approx(1)
    assert len(p["first_probs"])==3
    assert fitted["first_weighted_nll_after"]<=fitted["first_weighted_nll_before"]+1e-5
    assert target_index(rows[1])==2
    assert target_index(rows[1],True)==2
    assert target_index(rows[2],True)==3
    with pytest.raises(ValueError): checked_probs([.4,.4],1)
    with pytest.raises(ValueError): mixture_weights([rows[0]],.2)


def test_policy_training_and_safe_reload(tmp_path):
    rows,preds=fixture_rows()
    for exact in [False,True]:
        actor,metadata=train_actor(rows,preds,{"steps":5,"batch_size":12,"target_oos_prior":1/3},full_information=exact)
        assert all(np.isfinite(log["loss"]) for log in metadata["logs"])
        location=tmp_path/str(exact); save_actor(actor,location,metadata); loaded=load_actor(location)
        features=np.stack([state_features(rows[0],preds[0])])
        np.testing.assert_array_equal(actor.choose(features),loaded.choose(features))


def test_full_feedback_never_uses_test_labels():
    rows,preds=fixture_rows(); fitted=fit_full_feedback(rows,preds,1/3)
    before=run_policy("full_feedback",rows,preds,full_feedback=fitted)
    changed=copy.deepcopy(rows)
    for r in changed: r["label"]="unseen_truth"
    after=run_policy("full_feedback",changed,preds,full_feedback=fitted)
    assert [(x["expanded"],x["accepted"]) for x in before]==[(x["expanded"],x["accepted"]) for x in after]


def test_metrics_are_defined_for_no_accepts_and_paired_samples():
    rows,preds=fixture_rows(); out=run_policy("never_expand",rows,preds)
    metrics=summarize(out)
    assert metrics["coverage"]==pytest.approx(1/3)
    assert metrics["accepted_error"]==0
    assert metrics["oos_false_acceptance"]==0
    assert probability_metrics(rows,preds)["n"]==30
    assert paired_bootstrap(out,out,samples=30)["ci95"]==[0.,0.]


def test_invalid_predictions_and_costs_fail_closed():
    with pytest.raises(ValueError): Costs(wrong=-1)
    rows,preds=fixture_rows(); preds=preds[:-1]
    with pytest.raises(ValueError): run_policy("actor",rows,preds,actor=Actor())
