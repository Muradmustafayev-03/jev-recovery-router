"""Check prepared data, pinned tokenizer lengths, hardware and imports before GPU training."""
import argparse
import json
from pathlib import Path
import platform
import time
import torch
from transformers import AutoTokenizer
from recovery_router.io import read_json, read_jsonl, write_json, sha256_file
from recovery_router.model import make_view, MAX_CANDIDATES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/h100.json"); p.add_argument("--data-dir",default="data/processed")
    p.add_argument("--output",default="runs/preflight.json"); p.add_argument("--require-cuda",action="store_true")
    p.add_argument("--lengths",action="store_true",help="Scan all prepared input lengths using the pinned tokenizer")
    a=p.parse_args(); config=read_json(a.config); data=Path(a.data_dir)
    result={"python":platform.python_version(),"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"config_sha256":sha256_file(a.config),"status":"checking"}
    if a.require_cuda and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; refusing to silently train the main model on CPU")
    if torch.cuda.is_available():
        result.update(device=torch.cuda.get_device_name(0),vram_gib=torch.cuda.get_device_properties(0).total_memory/(1024**3),bf16_supported=torch.cuda.is_bf16_supported())
        if not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16 is required by the configured training precision")
    manifest=read_json(data/"manifest.json"); result["data_manifest_sha256"]=sha256_file(data/"manifest.json")
    if platform.python_version_tuple()[:2] != ('3','12'): raise RuntimeError("Use the validated Python 3.12 runtime")
    retrieval=manifest["retrieval"]; expected=config["retrieval"]
    if (retrieval["backend"],retrieval["first_k"],retrieval["expanded_k"])!=(expected["backend"],expected["initial_k"],expected["expanded_k"]):
        raise ValueError("Prepared retrieval profile differs from experiment configuration")
    for name,entry in manifest["files"].items():
        if sha256_file(data/name)!=entry["sha256"]: raise ValueError(f"Prepared file differs from manifest: {name}")
    required=["train","policy_train","calibration","tuning","test_base","test","transfer"]
    result["counts"]={name:len(read_jsonl(data/(name+".jsonl"))) for name in required}
    if not all(result["counts"].values()): raise ValueError("Missing/empty prepared split")
    if a.lengths:
        model=config["model"]
        tokenizer=AutoTokenizer.from_pretrained(model["name"],revision=model["revision"],trust_remote_code=False)
        max_length=model["max_length"]; over=[]; maximum=0; chat_maximum=0
        letters=[tokenizer.encode(chr(65+i),add_special_tokens=False) for i in range(MAX_CANDIDATES+1)]
        if any(len(ids)!=1 for ids in letters) or len({ids[0] for ids in letters})!=MAX_CANDIDATES+1:
            raise ValueError("Unadapted comparator requires distinct single-token A..U labels")
        for name in required:
            rows=read_jsonl(data/(name+".jsonl"))
            for row in rows:
                for expanded in (False,True):
                    view=make_view(row,expanded)
                    # make_view exposes serialized text, never the gold label.
                    length=len(tokenizer.encode(view["text"],add_special_tokens=True))
                    maximum=max(maximum,length)
                    if length>max_length: over.append({"id":row["id"],"split":name,"expanded":expanded,"tokens":length})
                    choices=", ".join(f"{chr(65+i)}=position {i}" for i in range(view["candidate_count"]))
                    text=view["text"]+f"\nReturn one letter only: {choices}, U=NONE_OF_SHOWN.\nANSWER:"
                    if getattr(tokenizer,"chat_template",None):
                        text=tokenizer.apply_chat_template([{"role":"user","content":text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    chat_length=len(tokenizer.encode(text,add_special_tokens=True)); chat_maximum=max(chat_maximum,chat_length)
                    if chat_length>max_length: over.append({"id":row["id"],"split":name,"expanded":expanded,"tokens":chat_length,"provider":"base_chat"})
        result.update(max_observed_input_tokens=maximum,max_base_chat_input_tokens=chat_maximum,overlong_count=len(over),overlong_examples=over[:20])
        if over:
            result["status"]="failed_input_lengths"; write_json(a.output,result)
            raise ValueError(f"{len(over)} views exceed max_length={max_length}; adjust serialization/context BEFORE sealing, never truncate candidates")
    result["status"]="passed"; write_json(a.output,result); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
