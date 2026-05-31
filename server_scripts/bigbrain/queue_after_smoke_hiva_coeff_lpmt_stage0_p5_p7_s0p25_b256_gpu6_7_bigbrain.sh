#!/usr/bin/env bash
set -euo pipefail

# Wait for GPU5 to be free enough, smoke-test P5/P7, then wait for GPU6/7 and
# launch the requested sequential 10k-step finetunes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"
OUTER_LOG="${OUTER_LOG:-${LOG_DIR}/queue_after_smoke_hiva_coeff_lpmt_stage0_p5_p7_s0p25_b256_gpu6_7_bigbrain_${RUN_STAMP}.outer.log}"
exec > >(tee -a "${OUTER_LOG}") 2>&1

GPU5_MAX_USED_MIB="${GPU5_MAX_USED_MIB:-10000}"
GPU67_MAX_USED_MIB="${GPU67_MAX_USED_MIB:-10000}"
POLL_SECONDS="${POLL_SECONDS:-300}"

gpu_used_mib() {
  local gpu="$1"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' '
}

wait_for_gpus() {
  local max_used="$1"
  shift
  local gpus=("$@")
  while true; do
    local ready=1
    local parts=()
    for gpu in "${gpus[@]}"; do
      local used
      used="$(gpu_used_mib "${gpu}")"
      parts+=("GPU${gpu}=${used}MiB")
      if (( used > max_used )); then
        ready=0
      fi
    done
    echo "$(date): ${parts[*]} threshold<=${max_used}MiB"
    if (( ready )); then
      return 0
    fi
    sleep "${POLL_SECONDS}"
  done
}

echo "Queue-after-smoke started at $(date)"
echo "RUN_STAMP=${RUN_STAMP}"
echo "OUTER_LOG=${OUTER_LOG}"

echo "Waiting for GPU5 for smoke tests..."
wait_for_gpus "${GPU5_MAX_USED_MIB}" 5

echo "Launching GPU5 smoke tests..."
RUN_STAMP="${RUN_STAMP}" \
bash "${SCRIPT_DIR}/smoke_hiva_coeff_lpmt_stage0_p5_p7_s0p25_b256_gpu5_bigbrain.sh"

echo "Smoke tests passed. Waiting for GPU6/7 for full sequential finetunes..."
wait_for_gpus "${GPU67_MAX_USED_MIB}" 6 7

echo "Launching sequential GPU6/7 finetunes..."
RUN_STAMP="${RUN_STAMP}" \
bash "${SCRIPT_DIR}/run_hiva_coeff_lpmt_stage0_p5_p7_s0p25_b256_gpu6_7_sequential_bigbrain.sh"

echo "Queue-after-smoke finished at $(date)"
