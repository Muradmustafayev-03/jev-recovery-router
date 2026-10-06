#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export ROUTER_CONFIG=configs/a100.json
export ROUTER_RUN_SCRIPT=scripts/run_a100.sh
exec bash scripts/bootstrap_h100.sh
