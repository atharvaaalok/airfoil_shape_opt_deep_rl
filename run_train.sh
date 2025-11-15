#!/usr/bin/env bash
set -euo pipefail


LOG_NAME="log1"

for SEED in {0..10}; do
    echo "Running with SEED=${SEED}"

    RUN_DIR="logs/${LOG_NAME}/seed_${SEED}"

    # Create directory
    mkdir -p "$RUN_DIR"

    # Run training and save output to a log file in that directory
    uv run train.py --seed "$SEED" --log-name "$LOG_NAME" > "${RUN_DIR}/main_log.txt"
done