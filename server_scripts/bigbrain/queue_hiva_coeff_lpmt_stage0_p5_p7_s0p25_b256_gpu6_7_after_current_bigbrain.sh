#!/usr/bin/env bash
set -euo pipefail

# Wait for GPU6/7 to become free, then run the requested P5 -> P7 sequential
# S=0.25, b256, 10k-step LP-MT HiVA finetunes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"
OUTER_LOG="${OUTER_LOG:-${LOG_DIR}/queue_hiva_coeff_lpmt_stage0_p5_p7_s0p25_b256_gpu6_7_after_current_bigbrain_${RUN_STAMP}.outer.log}"
exec > >(tee -a "${OUTER_LOG}") 2>&1

GPU67_MAX_USED_MIB="${GPU67_MAX_USED_MIB:-10000}"
POLL_SECONDS="${POLL_SECONDS:-300}"

gpu_used_mib() {
  local gpu="$1"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' '
}

echo "GPU6/7 queue started at $(date)"
echo "RUN_STAMP=${RUN_STAMP}"
echo "OUTER_LOG=${OUTER_LOG}"
echo "Threshold: GPU6/GPU7 <= ${GPU67_MAX_USED_MIB} MiB"

while true; do
  used6="$(gpu_used_mib 6)"
  used7="$(gpu_used_mib 7)"
  echo "$(date): GPU6=${used6}MiB GPU7=${used7}MiB"
  if (( used6 <= GPU67_MAX_USED_MIB && used7 <= GPU67_MAX_USED_MIB )); then
    break
  fi
  sleep "${POLL_SECONDS}"
done

echo "GPU6/7 are free enough. Launching sequential P5 -> P7 finetunes at $(date)"
RUN_STAMP="${RUN_STAMP}" \
bash "${SCRIPT_DIR}/run_hiva_coeff_lpmt_stage0_p5_p7_s0p25_b256_gpu6_7_sequential_bigbrain.sh"

echo "GPU6/7 queue finished at $(date)"
