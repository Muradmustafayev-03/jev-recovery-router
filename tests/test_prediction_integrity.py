import copy
import pytest
from recovery_router.io import read_verified_predictions, sha256_file, write_json, write_jsonl


def artifacts(tmp_path):
    data=tmp_path/'data.jsonl'; preds=tmp_path/'predictions.jsonl'
    row={'id':'a','candidates':[{'id':'i'}],'expanded_candidates':[{'id':'i'},{'id':'j'}]}
    pred={'id':'a','model_fingerprint':'fixed-provider','first_candidate_ids':['i'],'expanded_candidate_ids':['i','j'],
          'first_probs':[.8,.2],'expanded_probs':[.7,.2,.1]}
    write_jsonl(data,[row]); write_jsonl(preds,[pred])
    write_json(preds.with_suffix('.jsonl.meta.json'),{'model_fingerprint':'fixed-provider','data_sha256':sha256_file(data)})
    return data,preds,pred


@pytest.mark.parametrize('mutation',['order','probabilities','provider'])
def test_predictions_reject_stale_or_invalid_forecasts(tmp_path,mutation):
    data,preds,pred=artifacts(tmp_path)
    read_verified_predictions(data,preds,'fixed-provider')
    if mutation=='order': pred['expanded_candidate_ids'].reverse()
    if mutation=='probabilities': pred['first_probs']=[1.2,-.2]
    if mutation=='provider': pred['model_fingerprint']='wrong-provider'
    write_jsonl(preds,[pred])
    with pytest.raises(ValueError): read_verified_predictions(data,preds,'fixed-provider')


def test_forecasts_cannot_be_reused_with_other_data_or_provider(tmp_path):
    data,preds,_=artifacts(tmp_path)
    with pytest.raises(ValueError,match='provider differs'): read_verified_predictions(data,preds,'other-model')
    write_jsonl(data,[{'id':'a','candidates':[{'id':'other'}],'expanded_candidates':[{'id':'i'},{'id':'j'}]}])
    with pytest.raises(ValueError,match='data fingerprint'): read_verified_predictions(data,preds)
