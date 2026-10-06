"""Combine measured runs without silently presenting missing competitors as wins."""
import argparse
import json
from pathlib import Path
from recovery_router.io import read_json, read_jsonl, write_json
from recovery_router.metrics import paired_bootstrap


def aggregate(run):
    run=Path(run); matrix={}; measured={}
    for name in ["main","qwen_base","jev"]:
        path=run/name/"results/results.json"
        if path.exists():
            results=read_json(path)
            status=results.get("status","")
            complete=status=="measured_offline_evaluation_complete"
            matrix[name]={"status":"measured" if complete else "incomplete","path":str(path)}
            if complete: measured[name]=results
        else: matrix[name]={"status":"not_measured","reason":"No completed results artifact; no superiority claim is permitted."}
    classical=run/"classical/results.json"
    classic=read_json(classical) if classical.exists() else None
    classic_complete=classic is not None and len(classic.get("evaluations",{}))==12 and "elapsed_seconds" in classic
    matrix["classical"]={"status":"measured" if classic_complete else ("incomplete" if classic else "not_measured"),"path":str(classical)}
    comparisons={}
    if "main" in measured:
        for competitor in ["qwen_base","jev"]:
            if competitor not in measured: continue
            for dataset in ["test_base","test","transfer"]:
                left_hash=measured["main"].get("test_data_hashes",{}).get(dataset)
                right_hash=measured[competitor].get("test_data_hashes",{}).get(dataset)
                if not left_hash or left_hash!=right_hash:
                    raise ValueError(f"Cannot compare different or unverified datasets: {dataset}")
                if measured["main"]["datasets"][dataset]["scenarios"]["primary"]["costs"]!=measured[competitor]["datasets"][dataset]["scenarios"]["primary"]["costs"]:
                    raise ValueError(f"Cannot compare different primary cost vectors: {dataset}")
                # Both use preregistered seed42; no choosing a best test seed.
                left=run/"main/results"/dataset/"rl_seed42.jsonl"
                right=run/competitor/"results"/dataset/"rl_seed42.jsonl"
                if left.exists() and right.exists():
                    comparisons[dataset+"_main_minus_"+competitor]=paired_bootstrap(read_jsonl(left),read_jsonl(right))
        if classic_complete:
            for dataset in ("test_base","test","transfer"):
                if measured["main"]["test_data_hashes"].get(dataset)!=classic.get("test_data_hashes",{}).get(dataset):
                    raise ValueError("Classical comparison requires identical verified test data")
                costs=measured["main"]["datasets"][dataset]["scenarios"]["primary"]["costs"]
                if (costs["correct"],-costs["wrong"],-costs["defer"])!=(classic["fitting"]["utility"]["correct"],classic["fitting"]["utility"]["wrong"],classic["fitting"]["utility"]["handoff"]):
                    raise ValueError("Classical comparator and learned run use different utility costs")
                left=read_jsonl(run/"main/results"/dataset/"rl_seed42.jsonl")
                for name in classic["evaluations"]:
                    if name.startswith(dataset+"_") and not (dataset=="test" and name.startswith("test_base_")):
                        right=run/"classical"/(name+".outcomes.jsonl")
                        if right.exists(): comparisons[dataset+"_main_minus_classical_"+name]=paired_bootstrap(left,read_jsonl(right))
    report={"benchmark_matrix":matrix,"paired_cross_provider_utility":comparisons,
            "status":"comparison_summary","note":"Costs are simulated. Jev absence cannot be interpreted as a win. Classical methods have different allowed training/candidate access; see protocol."}
    write_json(run/"BENCHMARK_MATRIX.json",report)
    lines=["# Benchmark comparison status","","| Comparator | Status |","|---|---|"]
    lines += [f"| {name} | {record['status']} |" for name,record in matrix.items()]
    lines += ["","## Learned model and unadapted backbone","","| Dataset | Provider / policy | Correct route | Coverage | Accepted error | Utility |","|---|---|---:|---:|---:|---:|"]
    for name,results in measured.items():
        for dataset,record in results["datasets"].items():
            for method in ["rl_seed42","exact_seed42","fixed_cost_seed42","full_feedback","confidence_gate","never_expand","always_expand"]:
                m=record["scenarios"]["primary"]["methods"][method]["overall"]
                error=f"{m['accepted_error']:.3%}" if m["accepted_error"] is not None else "undefined"
                lines.append(f"| {dataset} | {name}/{method} | {m['correct_route_rate']:.3%} | {m['coverage']:.3%} | {error} | {m['mean_utility']:.4f} |")
    if classic_complete:
        lines += ["","## Classical references (different labeled-data budget)","","| Evaluation | Correct route | Coverage | Accepted error | Utility |","|---|---:|---:|---:|---:|"]
        for name,record in classic["evaluations"].items():
            m=record["overall"]; error=f"{m['accepted_error']:.3%}" if m["accepted_error"] is not None else "undefined"
            lines.append(f"| {name} | {m['correct_route_rate']:.3%} | {m['coverage']:.3%} | {error} | {m['mean_utility']:.4f} |")
    lines += ["","Classical results are in classical/RESULTS.md. Detailed group, calibration, intervention, seed and cost results remain in each provider's results.json.",
              "","These are offline decisions with cached counterfactual observations. Online inference latency requires scripts/benchmark_online.py; do not use cached replay time as serving latency."]
    (run/"COMPARISON.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--run",required=True)
    print(json.dumps(aggregate(p.parse_args().run),indent=2))
