#!/bin/bash
# Sweep over history_length from 1 to 10
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/sweep_history.sh

for h in $(seq 1 10); do
    echo "=== Training with history_length=$h ==="
    python scripts/train.py --task point-mass-v0 --headless env.observations.policy.history_length=$h
done
