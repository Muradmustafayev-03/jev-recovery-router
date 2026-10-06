#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
export HF_HOME="$PWD/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_FLAVOR="${TORCH_CUDA_FLAVOR:-cu126}"
case "$TORCH_CUDA_FLAVOR" in
  cu118|cu126|cu128) ;;
  *) echo "Unsupported TORCH_CUDA_FLAVOR: $TORCH_CUDA_FLAVOR (use cu118, cu126 or cu128)" >&2; exit 2 ;;
esac
python3 -c 'import json,time; from pathlib import Path; p=Path(".bootstrap-timing.json"); p.write_text(json.dumps({"start_unix":time.time()})) if not p.exists() else None'
if command -v python3.12 >/dev/null 2>&1; then
  python3.12 -m venv .venv
elif python3 -c 'import sys; assert sys.version_info[:2] == (3,12)' 2>/dev/null; then
  python3 -m venv .venv
else
  # Install an isolated Python without changing the system interpreter.
  python3 -m pip install --target .bootstrap-tools uv==0.8.22
  export UV_PYTHON_INSTALL_DIR="$PWD/.python"
  PYTHONPATH="$PWD/.bootstrap-tools" python3 -m uv venv --python 3.12 --seed .venv
fi
source .venv/bin/activate
python -m pip install --upgrade 'pip>=24,<26' 'setuptools>=75' wheel
# Select the official runtime wheel compatible with the installed NVIDIA driver.
# This changes the CUDA runtime, while retaining BF16 and native PyTorch SDPA.
python -m pip install "torch==2.7.1+$TORCH_CUDA_FLAVOR" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA_FLAVOR"
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps --no-build-isolation -e '.[dev]'
python -m pip check
mkdir -p runs
python -m pip freeze > runs/requirements-gpu-resolved.txt
python -m pytest --basetemp .test-tmp-bootstrap -q
python scripts/preflight.py --config "${ROUTER_CONFIG:-configs/h100.json}" --require-cuda --output runs/bootstrap-preflight.json
echo "Environment ready. Start: bash ${ROUTER_RUN_SCRIPT:-scripts/run_h100.sh} --stress"
