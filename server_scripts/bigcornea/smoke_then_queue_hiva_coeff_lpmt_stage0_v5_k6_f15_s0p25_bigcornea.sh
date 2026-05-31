#!/usr/bin/env bash
set -euo pipefail

# Wait for the current BigCornea all-GPU job, smoke-test the queued K=6/F=15
# S=0.25 run with batch size 1 on GPU7, then launch the real full queue.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

FULL_RUN_STAMP="${FULL_RUN_STAMP:-20260523_034355}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"
OUTER_LOG="${OUTER_LOG:-${LOG_DIR}/smoke_then_queue_hiva_coeff_lpmt_stage0_v5_k6_f15_s0p25_bigcornea_${RUN_STAMP}.outer.log}"
exec > >(tee -a "${OUTER_LOG}") 2>&1

WAIT_FOR_SESSION="${WAIT_FOR_SESSION:-hiva_lpmt_k6_resume_then_s0p125}"
SMOKE_GPU="${SMOKE_GPU:-7}"
GPU_MAX_USED_MIB="${GPU_MAX_USED_MIB:-10000}"
POLL_SECONDS="${POLL_SECONDS:-300}"

gpu_used_mib() {
  local gpu="$1"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' '
}

wait_for_tmux_session() {
  if [[ -z "${WAIT_FOR_SESSION}" ]]; then
    return 0
  fi
  while tmux has-session -t "${WAIT_FOR_SESSION}" 2>/dev/null; do
    echo "$(date): waiting for tmux session ${WAIT_FOR_SESSION} to finish..."
    sleep "${POLL_SECONDS}"
  done
}

wait_for_gpu() {
  local gpu="$1"
  while true; do
    local used
    used="$(gpu_used_mib "${gpu}")"
    echo "$(date): GPU${gpu}=${used}MiB threshold<=${GPU_MAX_USED_MIB}MiB"
    if (( used <= GPU_MAX_USED_MIB )); then
      return 0
    fi
    sleep "${POLL_SECONDS}"
  done
}

SMOKE_RUN_NAME="smoke_smolvla_hiva_coeff_lpmt_stage0_v5_k6_f15_s0p25_b1_gpu${SMOKE_GPU}_${RUN_STAMP}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${SMOKE_RUN_NAME}}"
FULL_OUTER_LOG="${FULL_OUTER_LOG:-${LOG_DIR}/queue_hiva_coeff_lpmt_stage0_v5_k6_f15_s0p25_after_current_bigcornea_${FULL_RUN_STAMP}.outer.log}"

echo "K=6/F=15/S=0.25 smoke-then-queue started at $(date)"
echo "RUN_STAMP=${RUN_STAMP}"
echo "FULL_RUN_STAMP=${FULL_RUN_STAMP}"
echo "SMOKE_GPU=${SMOKE_GPU}"
echo "SMOKE_OUTPUT_DIR=${SMOKE_OUTPUT_DIR}"
echo "OUTER_LOG=${OUTER_LOG}"
echo "FULL_OUTER_LOG=${FULL_OUTER_LOG}"

if [[ -e "${SMOKE_OUTPUT_DIR}" ]]; then
  echo "ERROR: SMOKE_OUTPUT_DIR already exists: ${SMOKE_OUTPUT_DIR}" >&2
  exit 2
fi

wait_for_tmux_session
wait_for_gpu "${SMOKE_GPU}"

echo "$(date): launching 1-step smoke test on GPU${SMOKE_GPU}"
K_VALUES=6 \
RUN_STAMP="${RUN_STAMP}" \
RUN_NAME="${SMOKE_RUN_NAME}" \
OUTPUT_DIR="${SMOKE_OUTPUT_DIR}" \
GPU_IDS="${SMOKE_GPU}" \
NUM_GPUS=1 \
NUM_PROCESSES=1 \
BATCH_PER_GPU=1 \
S=0.25 \
BASE_STEPS=20000 \
STEPS=1 \
SAVE_FREQ=1 \
SAVE_STEPS="[1]" \
SCHEDULER_WARMUP_STEPS=1 \
SCHEDULER_DECAY_STEPS=1 \
WANDB_ENABLE=false \
bash "${SCRIPT_DIR}/run_hiva_coeff_lpmt_stage0_v5_k6_k12_s0p5_bigcornea.sh"

echo "$(date): smoke test passed; launching full queued finetune"
WAIT_FOR_SESSION="" \
RUN_STAMP="${FULL_RUN_STAMP}" \
OUTER_LOG="${FULL_OUTER_LOG}" \
bash "${SCRIPT_DIR}/queue_hiva_coeff_lpmt_stage0_v5_k6_f15_s0p25_after_current_bigcornea.sh"

echo "K=6/F=15/S=0.25 smoke-then-queue finished at $(date)"
