#!/usr/bin/env bash
set -euo pipefail

# Smoke-test HiVA coefficient SmolVLA on bigflow.
#
# This intentionally reuses the full bigflow HiVA coefficient finetuning script
# while overriding only the launch scale and step count. It is meant to verify
# that dataset sidecar loading, SmolVLA checkpoint initialization, coefficient
# flow-matching losses, noisy duration CE, checkpoint saving, and DDP wiring are
# runnable before launching a long job.
#
# Example:
#   bash server_scripts/bigflow/finetune_bigflow_hiva_coeff_smoke.sh
#
# Override example:
#   GPU_IDS=4,5,6,7 NUM_GPUS=4 BATCH_PER_GPU=4 STEPS=2 \
#   bash server_scripts/bigflow/finetune_bigflow_hiva_coeff_smoke.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/server_scripts/common_wandb.sh"
build_run_id

export GPU_IDS="${GPU_IDS:-4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-4}"
export NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
export BATCH_PER_GPU="${BATCH_PER_GPU:-2}"
export STEPS="${STEPS:-2}"
export SAVE_FREQ="${SAVE_FREQ:-2}"
export EVAL_FREQ="${EVAL_FREQ:-0}"
export LOG_FREQ="${LOG_FREQ:-1}"
export S="${S:-1}"
export WANDB_ENABLE="${WANDB_ENABLE:-false}"
export RUN_NAME="${RUN_NAME:-smolvla_hiva_coeff_bigflow_smoke_${RUN_ID}}"

exec bash "${SCRIPT_DIR}/finetune_bigflow_ckpt_20k_hiva_coeff.sh"
