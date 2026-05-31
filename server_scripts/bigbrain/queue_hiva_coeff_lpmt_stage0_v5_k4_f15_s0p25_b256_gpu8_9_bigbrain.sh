#!/usr/bin/env bash
set -euo pipefail

# Wait for GPUs 8,9 to become free, then launch the K=4/F=15 LP-MT stage-0
# HiVA finetune with b256/GPU, S=0.25, 10k steps.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"
OUTER_LOG="${OUTER_LOG:-${LOG_DIR}/queue_hiva_coeff_lpmt_stage0_v5_k4_f15_s0p25_b256_gpu8_9_bigbrain_${RUN_STAMP}.outer.log}"
exec > >(tee -a "${OUTER_LOG}") 2>&1

GPU_IDS="${GPU_IDS:-8,9}"
GPU_MAX_USED_MIB="${GPU_MAX_USED_MIB:-10000}"
POLL_SECONDS="${POLL_SECONDS:-300}"

gpu_used_mib() {
  local gpu="$1"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' '
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

echo "K=4 F=15 S=0.25 b256 BigBrain GPU8/9 queue started at $(date)"
echo "RUN_STAMP=${RUN_STAMP}"
echo "OUTER_LOG=${OUTER_LOG}"
echo "GPU_IDS=${GPU_IDS}"
echo "GPU_MAX_USED_MIB=${GPU_MAX_USED_MIB}"
echo "POLL_SECONDS=${POLL_SECONDS}"

wait_for_gpus

echo "$(date): GPUs ${GPU_IDS} are free enough; launching K=4/F=15 run"
RUN_STAMP="${RUN_STAMP}" \
GPU_IDS="${GPU_IDS}" \
bash "${SCRIPT_DIR}/run_hiva_coeff_lpmt_stage0_v5_k4_f15_s0p25_b256_gpu8_9_bigbrain.sh"

echo "K=4 F=15 S=0.25 b256 BigBrain GPU8/9 queue finished at $(date)"

