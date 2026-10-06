#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/run_h100.sh --config configs/a100.json --output runs/a100-001 "$@"
