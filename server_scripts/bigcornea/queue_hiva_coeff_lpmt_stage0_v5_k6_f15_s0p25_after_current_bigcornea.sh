#!/usr/bin/env bash
set -euo pipefail

# Queue the K=6, F=15 LP-MT stage-0 HiVA run with S=0.25 behind the current
# BigCornea all-GPU training job. This reuses the K=6/K=12 sweep launcher but
# restricts it to K=6 and writes checkpoints directly to add_disk3.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
S="${S:-0.25}"
S_TAG="${S//./p}"
RUN_NAME="${RUN_NAME:-smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k6_f15_bigcornea_b64_s${S_TAG}_${RUN_STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/nfs/bigcornea/add_disk3/jongwoopark/HiVA_train/finetuning_stage0}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"

LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"
OUTER_LOG="${OUTER_LOG:-${LOG_DIR}/queue_hiva_coeff_lpmt_stage0_v5_k6_f15_s0p25_after_current_bigcornea_${RUN_STAMP}.outer.log}"
exec > >(tee -a "${OUTER_LOG}") 2>&1

WAIT_FOR_SESSION="${WAIT_FOR_SESSION:-hiva_lpmt_k6_resume_then_s0p125}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-64}"
BASE_STEPS="${BASE_STEPS:-20000}"
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

wait_for_gpus() {
  IFS=',' read -r -a gpu_array <<< "${GPU_IDS}"
  while true; do
    local ready=1
    local parts=()
    for gpu in "${gpu_array[@]}"; do
      local used
      used="$(gpu_used_mib "${gpu}")"
      parts+=("GPU${gpu}=${used}MiB")
      if (( used > GPU_MAX_USED_MIB )); then
        ready=0
      fi
    done
    echo "$(date): ${parts[*]} threshold<=${GPU_MAX_USED_MIB}MiB"
    if (( ready )); then
      return 0
    fi
    sleep "${POLL_SECONDS}"
  done
}

echo "K=6 F=15 S=0.25 BigCornea queue started at $(date)"
echo "RUN_STAMP=${RUN_STAMP}"
echo "RUN_NAME=${RUN_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "OUTER_LOG=${OUTER_LOG}"
echo "WAIT_FOR_SESSION=${WAIT_FOR_SESSION}"
echo "GPU_IDS=${GPU_IDS} NUM_GPUS=${NUM_GPUS} BATCH_PER_GPU=${BATCH_PER_GPU} S=${S}"
echo "Expected steps: ceil(${BASE_STEPS} * 64 / (${NUM_GPUS} * ${BATCH_PER_GPU}) / ${S})"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "ERROR: OUTPUT_DIR already exists: ${OUTPUT_DIR}" >&2
  exit 2
fi

wait_for_tmux_session
wait_for_gpus

echo "$(date): launching ${RUN_NAME}"
K_VALUES=6 \
RUN_STAMP="${RUN_STAMP}" \
GPU_IDS="${GPU_IDS}" \
NUM_GPUS="${NUM_GPUS}" \
NUM_PROCESSES="${NUM_PROCESSES}" \
BATCH_PER_GPU="${BATCH_PER_GPU}" \
S="${S}" \
BASE_STEPS="${BASE_STEPS}" \
WANDB_ENABLE="${WANDB_ENABLE:-true}" \
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
bash "${SCRIPT_DIR}/run_hiva_coeff_lpmt_stage0_v5_k6_k12_s0p5_bigcornea.sh"

echo "K=6 F=15 S=0.25 BigCornea queue finished at $(date)"
