"""Fit decision policies on declared fitting splits, seal them, then evaluate.

Commands:
  python -m recovery_router.evaluate fit --data-dir data/processed --pred-dir runs/main/predictions --output runs/main/policies
  python -m recovery_router.evaluate test --data-dir data/processed --pred-dir runs/main/predictions --policies runs/main/policies --output runs/main/results
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import subprocess
import time

import joblib
import numpy as np

from .calibration import apply_calibration, fit_calibration
from .io import read_json, read_jsonl, write_json, write_jsonl, sha256_file, read_verified_predictions
from .metrics import (grouped_summary, paired_bootstrap, probability_metrics, standardized_summary, candidate_diagnostics,
                      grouped_probability_metrics, grouped_paired_bootstrap, actor_seed_summary, accepted_subset_pruning_curve)
from .policy import (Costs, train_actor, save_actor, load_actor, fit_full_feedback, run_policy, select_threshold)

FITTING_SPLITS=("calibration","tuning","policy_train")
TEST_SPLITS=("test_base","test","transfer")


def cost_grid(config):
    p=config["policy"]
    primary=Costs(**p["primary_cost"]); heldout=Costs(**p["heldout_cost"])
    result=[("primary",primary),("heldout_interpolation",heldout)]
    for i,(w,d,e) in enumerate(itertools.product(p["wrong_costs"],p["defer_costs"],p["expand_costs"])):
        c=Costs(wrong=w,defer=d,expand=e)
        if c!=primary: result.append((f"grid_{i:02d}",c))
    return result


def seal_inputs(paths):
    result=[]
    for role,path in paths:
        path=Path(path).resolve()
        result.append({"role":role,"path":str(path),"sha256":sha256_file(path)})
    return result


def verify_seal(path):
    seal=read_json(path)
    for item in seal["inputs"]:
        if not Path(item["path"]).exists() or sha256_file(item["path"])!=item["sha256"]:
            raise ValueError(f"Sealed artifact changed: {item['role']}; start a new versioned experiment")
    if seal.get("model_bundle"):
        from .model import bundle_fingerprint
        if bundle_fingerprint(seal["model_bundle"]) != seal["provider_fingerprint"]:
            raise ValueError("Sealed model bundle changed; start a new experiment")
    return seal


def fit(args):
    started=time.monotonic(); data=Path(args.data_dir); preds=Path(args.pred_dir); out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if (out/"seal.json").exists(): raise FileExistsError("This experiment is already sealed; use a new output directory")
    config=read_json(args.config)
    rows,raw,metas={},{},{}
    provider=None
    for s in FITTING_SPLITS:
        rows[s],raw[s],metas[s]=read_verified_predictions(data/(s+".jsonl"),preds/(s+".jsonl"),provider)
        provider=metas[s]["model_fingerprint"]
    bundle=Path(metas["calibration"].get("model", "__not_a_local_model__"))
    if metas["calibration"].get("provider_kind")=="adapted_local_bundle" and not (bundle/"metadata.json").exists():
        raise ValueError("The adapted provider bundle is missing; cannot seal unverifiable weights")
    bundle=str(bundle.resolve()) if (bundle/"metadata.json").exists() else None
    if bundle:
        from .model import bundle_fingerprint
        if bundle_fingerprint(bundle)!=provider: raise ValueError("Forecasts are stale relative to the model bundle")
    calibration=fit_calibration(rows["calibration"],raw["calibration"],config["policy"]["target_oos_prior"])
    write_json(out/"calibration.json",calibration)
    forecasts={s:[apply_calibration(p,calibration) for p in raw[s]] for s in FITTING_SPLITS}
    actor_paths=[]
    for full_information in [False,True]:
        for seed in config["policy"]["seeds"]:
            name=("exact" if full_information else "rl")+f"_seed{seed}"
            actor,metadata=train_actor(rows["policy_train"],forecasts["policy_train"],config["policy"],seed,full_information)
            metadata.update(calibration_sha256=sha256_file(out/"calibration.json"),provider_fingerprint=provider)
            save_actor(actor,out/name,metadata)
            actor_paths.append(name)
            print(json.dumps({"stage":"policy_fit","name":name,"seconds":metadata["seconds"]}),flush=True)
    primary=config["policy"]["primary_cost"]
    fixed_config=dict(config["policy"],wrong_costs=[primary["wrong"]],defer_costs=[primary["defer"]],expand_costs=[primary["expand"]])
    fixed,metadata=train_actor(rows["policy_train"],forecasts["policy_train"],fixed_config,42,False)
    metadata.update(serving_cost_override=primary,calibration_sha256=sha256_file(out/"calibration.json"),provider_fingerprint=provider)
    save_actor(fixed,out/"fixed_cost_seed42",metadata); actor_paths.append("fixed_cost_seed42")
    comparator=fit_full_feedback(rows["policy_train"],forecasts["policy_train"],config["policy"]["target_oos_prior"],config=config["policy"])
    joblib.dump(comparator,out/"full_feedback.joblib",compress=3)
    thresholds={name:select_threshold(rows["tuning"],forecasts["tuning"],cost,config["policy"]["target_oos_prior"]) for name,cost in cost_grid(config)}
    selection={"actor_paths":actor_paths,"thresholds":thresholds,"config":config,"fit_seconds":time.monotonic()-started,
               "primary_actor":"rl_seed42","selection_rule":"Seed42 was specified before results; report all seeds, never pick the test winner.",
               "cost_status":"simulated normalized utility; not measured business dollars"}
    write_json(out/"selection.json",selection)
    sources=sorted(Path(__file__).parent.glob("*.py"))
    inputs=[("config",args.config)]+[(f"data_{s}",data/(s+".jsonl")) for s in FITTING_SPLITS]+[(f"pred_{s}",preds/(s+".jsonl")) for s in FITTING_SPLITS]
    inputs += [("source_"+p.name,p) for p in sources]
    root=Path(__file__).resolve().parents[2]
    for name in ["docs/RESEARCH_PROTOCOL.md","docs/DESIGN_DECISIONS.md","pyproject.toml","requirements-lock.txt"]:
        path=root/name
        if path.exists(): inputs.append(("protocol_environment_"+name,path))
    if (root/"runs/requirements-gpu-resolved.txt").exists():
        inputs.append(("resolved_gpu_environment",root/"runs/requirements-gpu-resolved.txt"))
    inputs += [(f"prediction_metadata_{s}",preds/(s+".jsonl.meta.json")) for s in FITTING_SPLITS]
    inputs += [("artifact_"+str(p.relative_to(out)),p) for p in sorted(out.rglob("*")) if p.is_file()]
    test_hashes={}
    for split in TEST_SPLITS:
        for suffix in ("", "_drop_gold", "_shuffle"):
            name=split+suffix; path=data/(name+".jsonl")
            if path.exists():
                inputs.append(("sealed_test_"+name,path)); test_hashes[name]=sha256_file(path)
    if (data/"manifest.json").exists(): inputs.append(("data_manifest",data/"manifest.json"))
    try: commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except (subprocess.SubprocessError,OSError): commit=None
    write_json(out/"seal.json",{"schema_version":2,"created_unix":time.time(),"git_commit":commit,"inputs":seal_inputs(inputs),
             "provider_fingerprint":provider,"model_bundle":bundle,"test_data_hashes":test_hashes,
             "notes":"Test files are fingerprinted without scoring. Changing source/data/config after seal requires a new experiment version."})
    print(json.dumps({"status":"sealed","output":str(out),"seconds":time.monotonic()-started}))


def evaluate(args):
    data=Path(args.data_dir); pred_dir=Path(args.pred_dir); policies=Path(args.policies); output=Path(args.output)
    seal=verify_seal(policies/"seal.json")
    selection=read_json(policies/"selection.json"); config=selection["config"]
    calibration=read_json(policies/"calibration.json")
    actors={name:load_actor(policies/name) for name in selection["actor_paths"]}
    # Only deserialize locally produced, sealed joblib artifacts.
    comparator=joblib.load(policies/"full_feedback.joblib")
    output.mkdir(parents=True,exist_ok=True)
    datasets=list(TEST_SPLITS)
    if args.stress:
        datasets += [s+suffix for s in TEST_SPLITS for suffix in ["_drop_gold","_shuffle"]]
    complete={"status":"evaluation_in_progress", "seal_sha256":sha256_file(policies/"seal.json"),"datasets":{},
              "provider_fingerprint":seal["provider_fingerprint"],"test_data_hashes":seal["test_data_hashes"],
              "costs":"simulated utility, not actual business costs","comparison_scope":"One forecast provider with multiple policies. Separate provider/classical/Jev results must be linked before cross-model claims.",
              "latency":"Offline component accounting; use inference CLI for actual online decision latency."}
    for dataset in datasets:
        if sha256_file(data/(dataset+".jsonl"))!=seal["test_data_hashes"].get(dataset):
            raise ValueError("Test dataset differs from the pre-evaluation seal")
        rows,raw,_=read_verified_predictions(data/(dataset+".jsonl"),pred_dir/(dataset+".jsonl"),seal["provider_fingerprint"])
        forecasts=[apply_calibration(p,calibration) for p in raw]
        record={"n":len(rows),"oos_prevalence":float(np.mean([r["is_oos"] for r in rows])),
                "candidate_diagnostics":candidate_diagnostics(rows,forecasts),
                "calibration":{"raw_initial":probability_metrics(rows,raw),"scaled_initial":probability_metrics(rows,forecasts),
                               "raw_expanded":probability_metrics(rows,raw,True),"scaled_expanded":probability_metrics(rows,forecasts,True)},
                "calibration_by_group":grouped_probability_metrics(rows,raw,forecasts),"scenarios":{}}
        for scenario,cost in cost_grid(config):
            # Stress tests retain the primary scenario to keep runtime bounded.
            if dataset not in TEST_SPLITS and scenario!="primary": continue
            methods={}
            for method in ["never_expand","always_expand","confidence_gate","full_feedback","oracle_upper_bound"]:
                methods[method]=run_policy(method,rows,forecasts,cost,full_feedback=comparator,threshold=selection["thresholds"][scenario])
            for name,actor in actors.items():
                if scenario!="primary" and not name.endswith("seed42"): continue
                methods[name]=run_policy("actor",rows,forecasts,cost,actor=actor)
            summary={"costs":asdict(cost),"methods":{name:grouped_summary(values) for name,values in methods.items()}}
            for name,values in methods.items():
                summary["methods"][name]["standardized_mixture"]=standardized_summary(values,config["policy"]["target_oos_prior"])
            if scenario=="primary":
                primary=methods[selection["primary_actor"]]
                summary["paired_utility_comparisons"]={name:paired_bootstrap(primary,values,config["evaluation"]["bootstrap_samples"],config["evaluation"]["seed"])
                                                      for name,values in methods.items() if name!=selection["primary_actor"]}
                summary["main_vs_confidence_gate_by_group"]=grouped_paired_bootstrap(
                    primary,methods["confidence_gate"],config["evaluation"]["bootstrap_samples"],config["evaluation"]["seed"])
                summary["actor_seed_summary"]=actor_seed_summary(methods)
                summary["accepted_subset_pruning_curve"]={name:accepted_subset_pruning_curve(values) for name,values in methods.items()}
                for name,values in methods.items(): write_jsonl(output/dataset/(name+".jsonl"),values)
            record["scenarios"][scenario]=summary
        complete["datasets"][dataset]=record
        write_json(output/"results.json",complete)
        print(json.dumps({"evaluated":dataset,"n":len(rows)}),flush=True)
    complete["status"]="measured_offline_evaluation_complete"
    write_json(output/"results.json",complete)
    write_report(output/"results.md",complete)
    print(f"Evaluation report: {output/'results.md'}")


def write_report(path,results):
    lines=["# Measured offline results", "", "These values come from the recorded run. Utilities use declared simulated costs. The oracle uses ground truth and is not deployable.", "",
           "Shared-batch component timing is not a serving-latency benchmark. Frozen model reuse on BANKING77 includes labeled catalog exemplars.", ""]
    for dataset,record in results["datasets"].items():
        lines += [f"## {dataset}","",f"Queries: {record['n']}; OOS prevalence: {record['oos_prevalence']:.3%}.","",
                  "| Method | Correct route | Wrong route | Coverage | Error when accepted | Expand | Utility |", "|---|---:|---:|---:|---:|---:|---:|"]
        for name,grouped in record["scenarios"]["primary"]["methods"].items():
            m=grouped["overall"]
            error=f"{m['accepted_error']:.3%}" if m["accepted_error"] is not None else "undefined (no accepted cases)"
            lines.append(f"| {name} | {m['correct_route_rate']:.3%} | {m['wrong_route_rate']:.3%} | {m['coverage']:.3%} | {error} | {m['expand_rate']:.3%} | {m['mean_utility']:.4f} |")
        lines += ["", "Per-group metrics, paired intervals, probability scores and the full cost grid are in results.json.",""]
    Path(path).write_text("\n".join(lines),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__); commands=parser.add_subparsers(dest="command",required=True)
    fitp=commands.add_parser("fit"); testp=commands.add_parser("test")
    for p in [fitp,testp]:
        p.add_argument("--data-dir",default="data/processed"); p.add_argument("--pred-dir",required=True); p.add_argument("--output",required=True)
    fitp.add_argument("--config",default="configs/h100.json")
    testp.add_argument("--policies",required=True); testp.add_argument("--stress",action="store_true")
    args=parser.parse_args(); (fit if args.command=="fit" else evaluate)(args)


if __name__=="__main__": main()
