"""Run bounded CUDA stages. No publishing or paid API calls are automatic."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def fingerprint(path):
    path=Path(path)
    paths=sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    result={}
    for item in paths:
        digest=hashlib.sha256()
        with item.open("rb") as source:
            for block in iter(lambda:source.read(1<<20),b""): digest.update(block)
        result[str(item.relative_to(path)) if path.is_dir() else item.name]=digest.hexdigest()
    return result


def run_signature(args,stages):
    source={}
    for directory in ["src","scripts"]:
        for path in sorted((ROOT/directory).rglob("*")):
            if path.suffix in (".py",".sh"):
                source[str(path.relative_to(ROOT))]=fingerprint(path)
    for name in ["pyproject.toml","requirements-lock.txt"]:
        source[name]=fingerprint(ROOT/name)
    return {"config":fingerprint(args.config),"data":fingerprint(args.data_dir),"commands":stages,"source":source}


def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8"); os.replace(temp,path)


def build_stages(args,config):
    python=sys.executable; run=Path(args.output); data=Path(args.data_dir)
    stages=[]
    def stage(name,command): stages.append((name,[str(x) for x in command]))
    stage("hardware_preflight",[python,"scripts/preflight.py","--config",args.config,"--data-dir",data,"--require-cuda","--output",run/"preflight.json"])
    stage("prefetch_backbone",[python,"scripts/prefetch_models.py","--config",args.config])
    stage("tokenizer_length_preflight",[python,"scripts/preflight.py","--config",args.config,"--data-dir",data,"--require-cuda","--lengths","--output",run/"length-preflight.json"])
    pilot=config.get("pilot",{})
    stage("training_smoke",[python,"-m","recovery_router.train","--config",args.config,"--data",data/"train.jsonl","--output",run/"smoke","--smoke-steps",pilot.get("optimizer_steps",5),"--max-seconds",pilot.get("max_seconds",600)])
    stage("train_beliefs",[python,"-m","recovery_router.train","--config",args.config,"--data",data/"train.jsonl","--output",run/"main/model","--max-seconds",config["training"]["max_seconds"]])
    stage("classical_benchmarks",[python,"scripts/run_classical.py","--data-dir",data,"--output",run/"classical","--backend","minilm"])
    systems=[("main",["--model",run/"main/model"])]
    if not args.skip_base: systems.append(("qwen_base",["--base-model",config["model"]["name"],"--revision",config["model"]["revision"]]))
    fit_splits=["calibration","tuning","policy_train"]
    test_splits=["test_base","test","transfer"]
    if args.stress: test_splits += [s+suffix for s in list(test_splits) for suffix in ["_drop_gold","_shuffle"]]
    for name,model in systems:
        for split in fit_splits:
            stage(f"{name}_predict_{split}",[python,"-m","recovery_router.predict",*model,"--data",data/(split+".jsonl"),"--output",run/name/"predictions"/(split+".jsonl"),"--batch-size",args.batch_size])
        stage(f"{name}_fit_and_seal_policies",[python,"-m","recovery_router.evaluate","fit","--data-dir",data,"--pred-dir",run/name/"predictions","--output",run/name/"policies","--config",args.config])
        for split in test_splits:
            stage(f"{name}_predict_{split}",[python,"-m","recovery_router.predict",*model,"--data",data/(split+".jsonl"),"--output",run/name/"predictions"/(split+".jsonl"),"--batch-size",args.batch_size])
        stage(f"{name}_evaluate",[python,"-m","recovery_router.evaluate","test","--data-dir",data,"--pred-dir",run/name/"predictions","--policies",run/name/"policies","--output",run/name/"results",*(["--stress"] if args.stress else [])])
    stage("sequential_online_latency",[python,"scripts/benchmark_online.py","--model",run/"main/model","--policy",run/"main/policies/rl_seed42","--calibration",run/"main/policies/calibration.json","--data-dir",data,"--catalog","clinc_base","--count","100","--warmup","3","--device","cuda","--output",run/"main/online_latency.json"])
    stage("aggregate_comparisons",[python,"scripts/aggregate_results.py","--run",run])
    return stages


def stop_child(child):
    if child.poll() is not None: return
    try:
        if os.name=="posix": os.killpg(child.pid,signal.SIGTERM)
        else: child.terminate()
    except ProcessLookupError:
        return
    try: child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            if os.name=="posix": os.killpg(child.pid,signal.SIGKILL)
            else: child.kill()
        except ProcessLookupError:
            pass
        child.wait(timeout=5)


def interrupt_run(signum,frame):
    raise KeyboardInterrupt(f"Received {signal.Signals(signum).name}; stopping the active stage")


def aggregate_after_failure(run):
    """Best-effort CPU reporting; preserve the original stage/interrupt exception."""
    log_path=Path(run)/"aggregate_after_failure.log"
    try:
        with log_path.open("w",encoding="utf-8") as log:
            result=subprocess.run([sys.executable,"scripts/aggregate_results.py","--run",str(run)],
                                  cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=30,check=False)
        return {"status":"complete" if result.returncode==0 else "failed",
                "exit_code":result.returncode,"log":str(log_path)}
    except Exception as error:
        return {"status":"failed","error_type":type(error).__name__,"error":str(error),"log":str(log_path)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/h100.json"); parser.add_argument("--data-dir",default="data/processed")
    parser.add_argument("--output",default="runs/h100-001"); parser.add_argument("--max-hours",type=float,default=9)
    parser.add_argument("--preparation-hours",type=float,default=None)
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--skip-base",action="store_true",help="Explicitly mark the unadapted Qwen comparator missing")
    parser.add_argument("--stress",action="store_true"); parser.add_argument("--resume",action="store_true")
    parser.add_argument("--dry-run",action="store_true"); parser.add_argument("--poweroff-on-exit",action="store_true",help="Linux only; explicitly request sudo -n poweroff after completion or failure")
    args=parser.parse_args(); os.chdir(ROOT)
    config=json.loads(Path(args.config).read_text(encoding="utf-8")); budget=config["budget"]
    preparation=ROOT/"docs/PREPARATION.json"
    prep_record=json.loads(preparation.read_text()) if preparation.exists() else {}
    args.preparation_hours=args.preparation_hours if args.preparation_hours is not None else prep_record.get("reserved_hours",prep_record.get("elapsed_hours",1.0))
    if args.max_hours<=0 or args.preparation_hours<0: parser.error("Time allocations must be nonnegative and run time positive")
    allowed=min(args.max_hours,budget["max_elapsed_hours"]-args.preparation_hours)
    cost_hours=(budget["max_usd"]/budget["assumed_tax_multiplier"]-budget["incidentals_usd"])/budget["gpu_usd_per_hour"]
    allowed=min(allowed,cost_hours)
    if allowed<=0: parser.error("No time or money remains under the configured assumptions")
    stages=build_stages(args,config)
    if args.dry_run:
        print(json.dumps({"allowed_gpu_run_hours":allowed,"preparation_hours":args.preparation_hours,"stages":stages},indent=2)); return
    run=Path(args.output); run.mkdir(parents=True,exist_ok=True)
    ledger_path=run/"run_ledger.json"
    if ledger_path.exists() and not args.resume: raise FileExistsError("Output already has a run ledger; choose a fresh directory or --resume")
    signature=run_signature(args,stages)
    # JSON round-trip normalizes command tuples before comparison on resume.
    signature=json.loads(json.dumps(signature))
    bootstrap=ROOT/".bootstrap-timing.json"
    billing_start=json.loads(bootstrap.read_text())["start_unix"] if bootstrap.exists() else time.time()
    ledger=json.loads(ledger_path.read_text()) if args.resume and ledger_path.exists() else {"start_unix":billing_start,"stages":[],"preparation_hours":args.preparation_hours,"signature":signature}
    if ledger.get("signature")!=signature: raise ValueError("Config, data, source or stage commands changed; resume refused. Start a newly versioned experiment.")
    deadline=ledger["start_unix"]+allowed*3600
    ledger.update(allowed_gpu_run_hours=allowed,status="running",missing_competitors=["Jev: optional API comparison not run"]+(["Unadapted Qwen: explicitly skipped"] if args.skip_base else []))
    write(ledger_path,ledger)
    completed={x["name"] for x in ledger["stages"] if x.get("exit_code")==0}
    current=None
    previous_handlers={sig:signal.signal(sig,interrupt_run) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        for name,command in stages:
            if name in completed:
                previous=next(x for x in reversed(ledger["stages"]) if x["name"]==name and x.get("exit_code")==0)
                if previous.get("output_path") and fingerprint(previous["output_path"])!=previous.get("output_fingerprint"):
                    raise ValueError(f"Completed output for {name} changed; resume refused")
                continue
            if time.time()>=deadline: raise TimeoutError("Overall time/cost envelope reached")
            if name in ("training_smoke","train_beliefs"):
                training_cap=float(command[command.index("--max-seconds")+1])
                if time.time()+training_cap+2*3600>deadline:
                    raise TimeoutError("Training would consume the reserved two hours for evaluation; reduce training cap in a new run configuration")
            log_path=run/(name+".log")
            entry={"name":name,"command":command,"start_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"start_unix":time.time(),"log":str(log_path)}
            ledger["stages"].append(entry); write(ledger_path,ledger)
            print(f"Starting {name}; log={log_path}",flush=True)
            with log_path.open("w",encoding="utf-8") as log:
                current=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=(os.name=="posix"))
                while current.poll() is None:
                    if time.time()>=deadline:
                        stop_child(current); raise TimeoutError("Overall time/cost envelope reached; child stopped")
                    time.sleep(1)
                entry.update(exit_code=current.returncode,seconds=time.time()-entry["start_unix"])
                if current.returncode==0 and "--output" in command:
                    target=command[command.index("--output")+1]
                    if Path(target).exists(): entry.update(output_path=target,output_fingerprint=fingerprint(target))
                write(ledger_path,ledger)
                if current.returncode: raise RuntimeError(f"Stage {name} failed; inspect {log_path}. No result is marked complete.")
        ledger["status"]="complete"
    except BaseException as error:
        # A repeated stop request must not interrupt bounded child cleanup/reporting.
        for sig in previous_handlers: signal.signal(sig,signal.SIG_IGN)
        ledger.update(status="incomplete",error=str(error),error_type=type(error).__name__)
        if current is not None:
            try: stop_child(current)
            except Exception as cleanup_error: ledger["cleanup_error"]=str(cleanup_error)
        ledger["failure_aggregation"]=aggregate_after_failure(run)
        raise
    finally:
        try:
            hours=(time.time()-ledger["start_unix"])/3600
            ledger.update(elapsed_gpu_run_hours=hours,estimated_compute_usd=hours*budget["gpu_usd_per_hour"],
                          estimated_total_usd=(hours*budget["gpu_usd_per_hour"]+budget["incidentals_usd"])*budget["assumed_tax_multiplier"],
                          billing_caveat="Estimates exclude machine uptime before/after this runner. Stopping a process does not power off the VM.")
            write(ledger_path,ledger)
            if args.poweroff_on_exit:
                if os.name!="posix": print("Automatic poweroff is supported only on Linux",file=sys.stderr)
                else: subprocess.run(["sudo","-n","poweroff"],check=False)
        finally:
            for sig,handler in previous_handlers.items(): signal.signal(sig,handler)


if __name__=="__main__": main()
