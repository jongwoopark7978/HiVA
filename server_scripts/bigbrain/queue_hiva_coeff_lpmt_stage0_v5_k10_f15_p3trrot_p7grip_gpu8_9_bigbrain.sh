#!/usr/bin/env bash
set -euo pipefail

# Wait for GPUs 8,9 to become free, then launch the mixed-degree LP-MT
# finetune. This is intentionally a simple queue wrapper so it does not
# interrupt the currently running GPU8/9 job.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"
MAX_USED_MIB="${MAX_USED_MIB:-2000}"
GPU_IDS="${GPU_IDS:-8,9}"

gpu_used_mib() {
  local idx="$1"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${idx}" | tr -d ' '
}

echo "Queueing mixed-degree LP-MT finetune for GPUs ${GPU_IDS}"
echo "RUN_STAMP=${RUN_STAMP}"
echo "CHECK_INTERVAL_SECONDS=${CHECK_INTERVAL_SECONDS}"
echo "MAX_USED_MIB=${MAX_USED_MIB}"

while true; do
  used8="$(gpu_used_mib 8)"
  used9="$(gpu_used_mib 9)"
  echo "$(date): GPU8 used=${used8} MiB, GPU9 used=${used9} MiB"
  if [[ "${used8}" -le "${MAX_USED_MIB}" && "${used9}" -le "${MAX_USED_MIB}" ]]; then
    break
  fi
  sleep "${CHECK_INTERVAL_SECONDS}"
done

echo "GPUs ${GPU_IDS} are free enough; launching at $(date)"
RUN_STAMP="${RUN_STAMP}" \
GPU_IDS="${GPU_IDS}" \
bash "${SCRIPT_DIR}/run_hiva_coeff_lpmt_stage0_v5_k10_f15_p3trrot_p7grip_s0p25_bigbrain.sh"
