import copy
import importlib.util
from pathlib import Path
import pytest
from recovery_router.io import write_json, write_jsonl

spec=importlib.util.spec_from_file_location('aggregate_results',Path(__file__).resolve().parents[1]/'scripts/aggregate_results.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_missing_comparators_are_not_wins(tmp_path):
    result=module.aggregate(tmp_path)
    assert all(r['status']=='not_measured' for r in result['benchmark_matrix'].values())
    assert result['paired_cross_provider_utility']=={}


@pytest.mark.parametrize('mismatch',['data','cost'])
def test_cross_provider_comparisons_require_equal_data_and_costs(tmp_path,mismatch):
    base={'status':'measured_offline_evaluation_complete','test_data_hashes':{'test_base':'same'},
          'datasets':{'test_base':{'scenarios':{'primary':{'costs':{'correct':1,'wrong':5,'defer':.2,'expand':.05}}}}}}
    other=copy.deepcopy(base)
    if mismatch=='data': other['test_data_hashes']['test_base']='different'
    else: other['datasets']['test_base']['scenarios']['primary']['costs']['wrong']=10
    write_json(tmp_path/'main/results/results.json',base)
    write_json(tmp_path/'qwen_base/results/results.json',other)
    with pytest.raises(ValueError,match='Cannot compare different'): module.aggregate(tmp_path)
