"""Stage a local Hugging Face artifact; never upload or access the network."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recovery_router.model import bundle_fingerprint


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise ValueError(f"Required nonempty regular artifact missing: {path}")
    return path


def _finite_nonnegative(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _model_identity_files(model):
    """Preserve every hashed adapter/tokenizer file, or reject an unsafe bundle."""
    private_name = re.compile(
        r"(^|[._-])(credentials?|secrets?|passwords?|private|env|api[._-]?keys?|"
        r"access[._-]?tokens?|refresh[._-]?tokens?|auth[._-]?tokens?|authorization)([._-]|$)", re.I)

    def visit(path):
        if (path.is_symlink() or getattr(path, "is_junction", lambda: False)()
                or not path.resolve().is_relative_to(model)):
            raise ValueError("Model identity files must not contain links or escape the model directory")
        if path.name.startswith(".") or private_name.search(path.name):
            raise ValueError("Model identity files must not contain hidden or sensitive filenames")
        if path.is_dir():
            for child in sorted(path.iterdir()):
                yield from visit(child)
        elif path.is_file():
            yield path
        else:
            raise ValueError("Model identity files must be ordinary files or directories")

    for name in ("adapter", "tokenizer"):
        folder = model / name
        if not folder.is_dir():
            raise ValueError(f"Missing model identity directory: {name}")
        yield from visit(folder)


def _report_files(root):
    """Optional public report assets only; never recurse through caches or scripts."""
    private_name = re.compile(r"(^|[._-])(credentials?|secrets?|passwords?|private|env)([._-]|$)", re.I)
    credential_value = re.compile(r"\b(?:apikey_|vck_|hf_|sk-)[A-Za-z0-9_-]{20,}|[?&]token=[A-Za-z0-9._~%-]{16,}")
    secret_fields = {"apikey", "api_key", "typesafe_api_key", "jupyter_token", "access_token", "refresh_token",
                     "authorization", "password", "secret", "client_secret", "private_key"}

    def contains_secret(value):
        if isinstance(value, dict):
            return any((str(key).lower().replace("-", "_") in secret_fields and bool(item)) or contains_secret(item)
                       for key, item in value.items())
        if isinstance(value, list):
            return any(contains_secret(item) for item in value)
        return False

    for folder, suffixes in ((root / "reports", {".md", ".json"}),
                             (root / "reports/figures", {".png", ".svg", ".json", ".md"})):
        if not folder.exists():
            continue
        if folder.is_symlink() or not folder.resolve().is_relative_to(root):
            raise ValueError("Report assets must be ordinary directories inside the project")
        for file in sorted(folder.iterdir()):
            if (not file.is_file() or file.is_symlink() or file.name.startswith(".")
                    or file.suffix.lower() not in suffixes or private_name.search(file.name)):
                continue
            if file.suffix.lower() != ".png":
                text = file.read_text(encoding="utf-8")
                if credential_value.search(text):
                    continue
                if file.suffix.lower() == ".json" and contains_secret(json.loads(text)):
                    continue
            yield file


def _validation_selection_proof(model, metadata, training, fingerprint):
    """Check recorded selection consistency, not independent model performance."""
    proof_path = _required(model.parent / "training_summary.json")
    proof = _read(proof_path)
    step = metadata["step"]
    if (proof.get("selected_step") != step or metadata.get("checkpoint_selection_eligible") is not True
            or proof.get("selected_model_fingerprint") != fingerprint):
        raise ValueError("Validation selection proof disagrees with the eligible selected model or its fingerprint")
    executed = proof.get("actual_executed_optimizer_steps")
    if (not isinstance(executed, int) or isinstance(executed, bool) or executed < step
            or training.get("actual_executed_optimizer_steps") != executed
            or proof.get("stop_reason") not in {"epochs_complete", "time_cap"}):
        raise ValueError("Validation selection proof has inconsistent executed training steps or stop reason")
    selected_nll = proof.get("selected_validation_mean_nll")
    delta = proof.get("selection_min_delta")
    if not _finite_nonnegative(selected_nll) or not _finite_nonnegative(delta):
        raise ValueError("Validation selection proof requires finite nonnegative NLL and min_delta")
    if (metadata.get("selection_min_delta") != delta
            or training.get("selected_validation_mean_nll") != selected_nll
            or metadata.get("validation_metrics", {}).get("selection_mean_nll") != selected_nll):
        raise ValueError("Validation selection metric differs between metadata, report, and proof")
    for field in ("training_data_sha256", "validation_data_sha256", "validation_views_sha256", "training_config_sha256"):
        digest = proof.get(field)
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest.lower())
                or metadata.get(field) != digest):
            raise ValueError(f"Validation selection proof has missing or inconsistent provenance: {field}")
    for field in ("selection_rule", "selection_target_oos_prior", "selection_validation_records", "selection_validation_views"):
        if field not in proof or proof[field] != metadata.get(field):
            raise ValueError(f"Validation selection proof disagrees with model selection protocol: {field}")
    curve = proof.get("validation_curve")
    if not isinstance(curve, list) or not curve:
        raise ValueError("Validation selection proof requires the full validation curve")
    best_nll, best_step, selected_row, previous_step = None, None, None, -1
    for row in curve:
        if not isinstance(row, dict):
            raise ValueError("Validation selection curve contains an invalid row")
        row_step, value = row.get("step"), row.get("selection_mean_nll")
        if (not isinstance(row_step, int) or isinstance(row_step, bool) or row_step <= previous_step
                or not _finite_nonnegative(value)):
            raise ValueError("Validation selection curve requires strictly increasing steps and finite NLL")
        expected_selected = row_step > 0 and (best_nll is None or value < best_nll - delta)
        if row.get("selected_at_this_step") is not expected_selected:
            raise ValueError("Validation selection curve does not follow the recorded min_delta rule")
        if expected_selected:
            best_nll, best_step, selected_row = value, row_step, row
        previous_step = row_step
    if (best_step != step or best_nll != selected_nll or previous_step != executed
            or selected_row.get("model_fingerprint") != fingerprint):
        raise ValueError("Validation selection proof does not select this checkpoint from its full curve")
    selected_checkpoint = proof.get("selected_checkpoint")
    if not isinstance(selected_checkpoint, str):
        raise ValueError("Validation selection proof is missing its checkpoint path")
    relative = PurePosixPath(selected_checkpoint)
    if (relative.is_absolute() or ".." in relative.parts or "\\" in selected_checkpoint
            or ":" in selected_checkpoint or not relative.parts
            or selected_row.get("bundle") != selected_checkpoint
            or training.get("selected_checkpoint") != selected_checkpoint):
        raise ValueError("Validation selection checkpoint path is unsafe or inconsistent")
    return proof_path


def _clean_previous(output):
    """Delete only files belonging to a previously generated, verified export."""
    marker = output / "export_manifest.json"
    previous = _read(_required(marker))
    if previous.get("format") != "jev-recovery-router-offline-export-v1":
        raise ValueError("--overwrite only supports this script's previous exports")
    expected = set(previous["files"]) | {"export_manifest.json"}
    actual = set()
    for file in output.rglob("*"):
        if file.is_symlink():
            raise ValueError("Refusing to overwrite an export containing symlinks")
        if file.is_file():
            actual.add(file.relative_to(output).as_posix())
    if expected != actual:
        raise ValueError("Export contains untracked/missing files; choose a new output directory")
    for relative in expected:
        file = (output / relative).resolve()
        if not file.is_relative_to(output) or file == output:
            raise ValueError("Unsafe path in previous export manifest")
        if relative != "export_manifest.json" and _sha(file) != previous["files"][relative]["sha256"]:
            raise ValueError("Previous export was modified; choose a new output directory")
    # All final paths are verified inside the explicitly selected output.
    for relative in expected:
        (output / relative).unlink()
    for directory in sorted((p for p in output.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        directory.rmdir()


def export_bundle(model, policy, calibration, output, results=None, data_dir=None,
                  project_root=None, overwrite=False, include_reports=False):
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    model, policy = Path(model).resolve(), Path(policy).resolve()
    calibration, output = Path(calibration).resolve(), Path(output).resolve()
    data = Path(data_dir or root / "data/processed").resolve()
    if output == root or any(source == output or source.is_relative_to(output) or output.is_relative_to(source)
                             for source in (model, policy, data)):
        raise ValueError("Export output must be separate from the repository root and input artifacts")
    metadata = _read(_required(model / "metadata.json"))
    training = _read(_required(model / "training_report.json"))
    step = metadata.get("step", 0)
    if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
        raise ValueError("Refusing untrained model: metadata must record positive optimizer steps")
    if training.get("stop_reason") not in {"epochs_complete", "time_cap", "validation_selected"}:
        raise ValueError("Refusing a smoke-test or unknown training run")
    if training.get("optimizer_steps_total") != step:
        raise ValueError("Training report and model step do not agree")
    cfg = metadata["model"]
    retrieval = _read(_required(data / "manifest.json")).get("retrieval", {})
    initial_k = retrieval.get("initial_k", retrieval.get("first_k", 1))
    expanded_k = retrieval.get("expanded_k", 5)
    supported = metadata.get("trained_candidate_counts")
    if (not isinstance(initial_k, int) or not isinstance(expanded_k, int)
            or not 1 <= initial_k < expanded_k <= 20
            or not isinstance(supported, list)
            or not {initial_k, expanded_k}.issubset(set(supported))):
        raise ValueError("Catalog schedule is not covered by recorded trained_candidate_counts")
    base = cfg.get("name", cfg.get("name_or_path", ""))
    revision = cfg.get("revision", "")
    if (Path(base).is_absolute() or Path(base).exists() or not isinstance(revision, str)
            or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower())):
        raise ValueError("Portable export requires a remote base-model ID and immutable 40-character revision")
    policy_metadata = _read(_required(policy / "policy.json"))
    if not policy_metadata.get("training_rows", 0) or not policy_metadata.get("logs"):
        raise ValueError("Policy metadata does not record a completed training update")
    cal = _read(_required(calibration))
    if policy_metadata.get("calibration_sha256") != _sha(calibration):
        raise ValueError("Policy metadata names a different calibration artifact")
    # Validate the entire identity tree before hashing or creating the export.
    identity_files = list(_model_identity_files(model))
    for name in ("adapter/adapter_config.json", "adapter/adapter_model.safetensors",
                 "tokenizer/tokenizer_config.json"):
        _required(model / name)
    fingerprint = bundle_fingerprint(model)
    selection_proof = None
    if training["stop_reason"] == "validation_selected":
        selection_proof = _validation_selection_proof(model, metadata, training, fingerprint)
    if policy_metadata.get("provider_fingerprint") != fingerprint:
        raise ValueError("Policy metadata is missing or names a different model provider fingerprint")
    for key in ("first_temperature", "expanded_temperature"):
        value = cal.get(key)
        if not isinstance(value, (float, int)) or not 0 < value < float("inf"):
            raise ValueError("Calibration requires two finite positive temperatures")
    result_verification = {"status": "not_supplied", "reported_status": None,
                           "provider_fingerprint_matched": None}
    if results is not None:
        supplied_results = _read(_required(results))
        result_status = supplied_results.get("status") if isinstance(supplied_results, dict) else None
        result_fingerprint = supplied_results.get("provider_fingerprint") if isinstance(supplied_results, dict) else None
        if result_status == "measured_offline_evaluation_complete":
            if result_fingerprint != fingerprint:
                raise ValueError("Completed results are missing or name a different model provider fingerprint")
            result_verification = {"status": "complete_provider_fingerprint_matched",
                                   "reported_status": result_status, "provider_fingerprint_matched": True}
        else:
            # Preserve incomplete/structural staging inputs, without promoting them
            # to measured evidence for this exported model.
            result_verification = {"status": "incomplete_or_unrecognized_staging_only",
                                   "reported_status": result_status,
                                   "provider_fingerprint_matched": result_fingerprint == fingerprint if result_fingerprint else None}

    copies = []
    def add(source, destination):
        copies.append((_required(source), destination))
    for name in ("metadata.json", "training_report.json", "head.safetensors"):
        add(model / name, "model/" + name)
    if selection_proof is not None:
        add(selection_proof, "provenance/training_summary.json")
    # Empty auxiliary files also contribute to the existing fingerprint.
    copies.extend((file, "model/" + file.relative_to(model).as_posix()) for file in identity_files)
    for name in ("policy.json", "policy.safetensors"):
        add(policy / name, "policy/" + name)
    for name in ("seal.json", "selection.json"):
        if (policy.parent / name).is_file():
            add(policy.parent / name, "provenance/" + name)
    for file in (root / "configs").glob("*.json"):
        add(file, "configs/" + file.name)
    add(calibration, "calibration.json")
    add(root / "MODEL_CARD.md", "MODEL_CARD.md")
    add(root / "pyproject.toml", "pyproject.toml")
    add(root / "LICENSE", "LICENSE")
    if (root / "NOTICE.md").exists():
        add(root / "NOTICE.md", "NOTICE.md")
    for file in (root / "src/recovery_router").rglob("*.py"):
        add(file, "src/recovery_router/" + file.relative_to(root / "src/recovery_router").as_posix())
    for file in (root / "docs").glob("*.md"):
        add(file, "docs/" + file.name)
    if include_reports:
        for file in _report_files(root):
            add(file, file.relative_to(root).as_posix())
    for name in ("manifest.json", "catalogs.json", "index.jsonl", "banking_index.jsonl"):
        add(data / name, "data/processed/" + name)
    for name in ("CLINC_LICENSE.txt", "BANKING77_LICENSE.txt"):
        add(data.parent / "downloads" / name, "data/licenses/" + name)
    if results is not None:
        add(results, "results.json")
    # Refuse absent/invalid inputs before creating or altering the destination.
    if output.exists():
        if not output.is_dir() or output.is_symlink():
            raise ValueError("Output is not an ordinary directory")
        if any(output.iterdir()):
            if not overwrite:
                raise FileExistsError("Output exists; choose a new path or use --overwrite on an unchanged prior export")
            _clean_previous(output)
    output.mkdir(parents=True, exist_ok=True)
    for source, relative in copies:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    try:
        portable_fingerprint = bundle_fingerprint(output / "model")
    except (ValueError, OSError) as error:
        raise ValueError("Exported model fingerprint could not be verified") from error
    if portable_fingerprint != fingerprint:
        raise ValueError("Exported model fingerprint differs from the original model bundle")

    card = (output / "MODEL_CARD.md").read_text(encoding="utf-8")
    card = card.replace("planned artifact, NOT TRAINED", "local artifact; evaluation claims pending")
    card = card.replace("**Status: NOT TRAINED / NOT EVALUATED.**", "**Status: TRAINING RECORDED / EVALUATION NOT INDEPENDENTLY VALIDATED.**")
    card = card.replace("It does not claim that trained weights exist, that training succeeded, or that any metric has been obtained.",
                        "This local export includes weights and recorded training metadata; it does not independently validate scientific performance.")
    card = card.replace("| NOT TRAINED |", "| TRAINING RECORDED; see export manifest |")
    card = card.replace("| NOT FITTED |", "| FITTED PARAMETERS SUPPLIED; validity not independently assessed |")
    card = card.replace("base_model: Qwen/Qwen3-1.7B", "base_model: " + base)
    (output / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    usage = '''\n\n## Loading this export\n\nThis is a custom decision-head/LoRA/actor bundle, not an `AutoModel.from_pretrained` model repository. No upload was performed. Install the included Python package with `python -m pip install .`, then run:\n\n```bash\npython -m recovery_router.inference --model model --policy policy --calibration calibration.json --data-dir data/processed --catalog clinc_base --device cpu --text "I need to replace my card"\n```\n\nUse `clinc_updated` or `banking` for the other supplied catalogs. Base Qwen and MiniLM weights are not duplicated in this export: first inference may download their pinned revisions unless cached; this export script itself performs no network requests. Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` after prefetching when fully offline inference is required.\n\nThe CLI performs the second retrieval/model pass only if the policy selects expansion. Latency excludes artifact/index loading, which is reported separately. Any supplied `results.json` is copied verbatim, not independently validated or converted into performance claims. Public catalog examples retain the separate dataset licenses under `data/licenses/`. Review the protocol and limitations before publication.\n'''
    (output / "README.md").write_text(card + usage, encoding="utf-8")
    manifest = {"format": "jev-recovery-router-offline-export-v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "publication_status": "local staging only; no upload performed",
                "model_training_steps": step, "training_stop_reason": training["stop_reason"],
                "model_fingerprint": fingerprint,
                "trained_candidate_counts": supported,
                "candidate_schedule": {"initial_k": initial_k, "expanded_k": expanded_k},
                "base_model": base, "base_revision": revision,
                "policy_method": policy_metadata.get("method"),
                "evaluation_status": "results supplied but not independently validated" if results else "NOT MEASURED in this export",
                "results_verification": result_verification,
                "report_assets_included": include_reports,
                "files": {p.relative_to(output).as_posix(): {"sha256": _sha(p), "bytes": p.stat().st_size}
                          for p in sorted(output.rglob("*")) if p.is_file()}}
    (output / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", default="dist/hf-model")
    parser.add_argument("--results")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-reports", action="store_true",
                        help="Include optional public report Markdown/JSON and direct PNG/SVG/JSON/Markdown figure assets")
    args = parser.parse_args(argv)
    manifest = export_bundle(args.model, args.policy, args.calibration, args.output,
                             args.results, args.data_dir, overwrite=args.overwrite, include_reports=args.include_reports)
    print(json.dumps({"output": str(Path(args.output).resolve()), "files": len(manifest["files"]),
                      "status": manifest["publication_status"]}))


if __name__ == "__main__":
    main()
