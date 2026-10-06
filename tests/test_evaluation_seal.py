from argparse import Namespace
import pytest
from recovery_router.io import read_json, write_json, write_jsonl, sha256_file
from recovery_router.evaluate import fit, evaluate, verify_seal
from test_policy_metrics import fixture_rows


def test_fit_seal_evaluate_and_reject_modified_config(tmp_path):
    data=tmp_path/"data"; predictions=tmp_path/"predictions"; policy=tmp_path/"policies"
    rows,preds=fixture_rows()
    for row,pred in zip(rows,preds):
        pred.update(model_fingerprint="synthetic-unit-test")
        pred["first_candidate_ids"]=[c["id"] for c in row["candidates"]]
        pred["expanded_candidate_ids"]=[c["id"] for c in row["expanded_candidates"]]
    for split in ["calibration","tuning","policy_train","test_base","test","transfer"]:
        write_jsonl(data/(split+".jsonl"),rows); write_jsonl(predictions/(split+".jsonl"),preds)
        write_json(predictions/(split+".jsonl.meta.json"),{"model_fingerprint":"synthetic-unit-test","data_sha256":sha256_file(data/(split+".jsonl"))})
    config={"policy":{"seeds":[42],"steps":2,"batch_size":8,"target_oos_prior":1/3,
                       "wrong_costs":[5.],"defer_costs":[.2],"expand_costs":[.05],
                       "primary_cost":{"wrong":5.,"defer":.2,"expand":.05},
                       "heldout_cost":{"wrong":7.,"defer":.3,"expand":.08}},
            "evaluation":{"bootstrap_samples":10,"seed":2026}}
    config_path=tmp_path/"config.json"; write_json(config_path,config)
    fit(Namespace(data_dir=data,pred_dir=predictions,output=policy,config=config_path))
    assert verify_seal(policy/"seal.json")["schema_version"]==2
    output=tmp_path/"results"
    evaluate(Namespace(data_dir=data,pred_dir=predictions,policies=policy,output=output,stress=False))
    result=read_json(output/"results.json")
    assert result["datasets"]["test"]["n"]==30
    record=result["datasets"]["test"]
    assert record["calibration_by_group"]["group"]["stable"]["scaled_initial"]["n"]==20
    primary=record["scenarios"]["primary"]
    assert "risk_coverage" not in primary
    assert "accepted_subset_pruning_curve" in primary
    assert primary["actor_seed_summary"]["rl"]["n_actor_seeds"]==1
    assert primary["actor_seed_summary"]["rl"]["sample_std_across_actor_seeds"] is None
    assert primary["main_vs_confidence_gate_by_group"]["group"]["stable"]["n"]==20
    assert primary["methods"]["always_expand"]["overall"]["expansion_diagnostics"]["expanded_n"]==30
    config["evaluation"]["seed"]=3; write_json(config_path,config)
    with pytest.raises(ValueError,match="Sealed artifact changed"): verify_seal(policy/"seal.json")
