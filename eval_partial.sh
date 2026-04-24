#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="outputs/eval_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"

# Save both stdout and stderr to terminal and log file
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"

CUDA_VISIBLE_DEVICES=6
lerobot-eval --policy.path=smolvla_libero --env.type=libero --env.task=libero_object --eval.batch_size=1 --eval.n_episodes=3