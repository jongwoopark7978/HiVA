#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"


export CUDA_VISIBLE_DEVICES=6

LOG_DIR="outputs/eval_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"

# Save both stdout and stderr to terminal and log file
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"

lerobot-eval --policy.path=smolvla_libero --env.type=libero --env.task=libero_spatial,libero_object,libero_goal,libero_10 --eval.batch_size=1 --eval.n_episodes=10

#libero_spatial,libero_object,libero_goal,libero_10