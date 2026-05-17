#!/usr/bin/env bash
set -euo pipefail

# Smoke test the decoded-action-loss MT and LP-MT bigflow runner on GPU3.
# This intentionally uses tiny steps/batch to validate wiring only.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

GPU_IDS="${GPU_IDS:-3}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
BATCH_PER_GPU="${BATCH_PER_GPU:-2}"
STEPS="${STEPS:-1}"
SAVE_FREQ="${SAVE_FREQ:-1}"
S="${S:-2}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
LOG_DIR="${LOG_DIR:-/home/jongwoopark/lerobot/outputs/train_logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/smoke_hiva_coeff_decoded_loss_mt_lpmt_gpu3_${RUN_STAMP}.log}"

mkdir -p "${LOG_DIR}"

echo "Writing smoke log to ${LOG_FILE}"
(
  set -x
  env \
    GPU_IDS="${GPU_IDS}" \
    NUM_GPUS="${NUM_GPUS}" \
    NUM_PROCESSES="${NUM_PROCESSES}" \
    BATCH_PER_GPU="${BATCH_PER_GPU}" \
    STEPS="${STEPS}" \
    SAVE_FREQ="${SAVE_FREQ}" \
    S="${S}" \
    WANDB_ENABLE="${WANDB_ENABLE}" \
    RUN_STAMP="${RUN_STAMP}_smoke" \
    bash "${SCRIPT_DIR}/run_hiva_coeff_decoded_loss_mt_lpmt_bigflow.sh"
) 2>&1 | tee "${LOG_FILE}"
