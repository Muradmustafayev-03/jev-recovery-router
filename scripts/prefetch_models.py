"""Populate the HF cache before provisioning a timed training run if desired."""
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/h100.json")
    parser.add_argument("--cache-dir")
    parser.add_argument("--tokenizer-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model = config["model"]
    revision = model.get("revision", "")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
        raise ValueError("Set an immutable 40-character model revision in the config first")
    patterns = ["config.json", "generation_config.json", "tokenizer*", "vocab.json", "merges.txt",
                "special_tokens_map.json", "added_tokens.json", "chat_template*"]
    if not args.tokenizer_only:
        patterns += ["*.safetensors", "*.safetensors.index.json"]
    path = snapshot_download(repo_id=model["name"], revision=revision,
                             allow_patterns=patterns, cache_dir=args.cache_dir)
    print(json.dumps({"model": model["name"], "revision": revision, "cached_path": path,
                      "tokenizer_only": args.tokenizer_only}))


if __name__ == "__main__":
    main()
