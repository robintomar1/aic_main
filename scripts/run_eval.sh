#!/usr/bin/env bash
# Usage: run_eval.sh <run_name> [entrypoint args...]
# Example: run_eval.sh cheatcode ground_truth:=true start_aic_engine:=true

set -e
NAME="${1:?usage: run_eval.sh <run_name> [entrypoint args...]}"
shift

RUN_DIR="/root/aic_results/$(date +%Y%m%d_%H%M%S)_${NAME}"
mkdir -p "$RUN_DIR"
export AIC_RESULTS_DIR="$RUN_DIR"
echo "=== writing results to $RUN_DIR ==="

/entrypoint.sh "$@"