#!/usr/bin/env bash
set -euo pipefail

# Sequential full 50-episode evals for the requested stage-0 LP-MT K4/K8 F15
# models on GPUs 4,5,8,9. The underlying launcher assigns suites by GPU order.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"
OUTER_LOG="${OUTER_LOG:-${LOG_DIR}/eval_stage0_lpmt_k4_k8_f15_gpu4_5_8_9_50eps_${TIMESTAMP}.outer.log}"
exec > >(tee -a "${OUTER_LOG}") 2>&1

BASE_LAUNCHER="${BASE_LAUNCHER:-${SCRIPT_DIR}/eval_stage0_lpmt_k4_f15_all_ckpts_gpu4_7_50eps_bigtoken.sh}"
GPU_IDS="${GPU_IDS:-4,5,8,9}"
GPU_GROUP="${GPU_GROUP:-gpu4_5_8_9}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark}"

K4_TRAIN_DIR="${K4_TRAIN_DIR:-/nfs/bigcornea.cs.stonybrook.edu/add_disk3/jongwoopark/HiVA_train/finetuning_stage0/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k4_f15_bigbrain_b256_g2_s0p25_steps10000_20260524_190429}"
K8_TRAIN_DIR="${K8_TRAIN_DIR:-/nfs/bigcornea.cs.stonybrook.edu/add_disk3/jongwoopark/HiVA_train/finetuning_stage0/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k8_f15_bigcornea_b64_s0p25_20260524_185903}"

run_one() {
  local hiva_k="$1"
  local train_dir="$2"
  local tag="k${hiva_k}_f15"

  echo "===== $(date) starting ${tag} full all-checkpoint eval ====="
  echo "TRAIN_DIR=${train_dir}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "GPU_GROUP=${GPU_GROUP}"

  HIVA_K="${hiva_k}" \
  TRAIN_DIR="${train_dir}" \
  GPU_IDS="${GPU_IDS}" \
  GPU_GROUP="${GPU_GROUP}" \
  SIDECAR_ROOT="${SIDECAR_ROOT}" \
  TIMESTAMP="${TIMESTAMP}_${tag}" \
  bash "${BASE_LAUNCHER}"

  echo "===== $(date) finished ${tag} full all-checkpoint eval ====="
}

echo "===== requested K4/K8 full eval sequence started at $(date) ====="
echo "TIMESTAMP=${TIMESTAMP}"
echo "OUTER_LOG=${OUTER_LOG}"
echo "BASE_LAUNCHER=${BASE_LAUNCHER}"

run_one 4 "${K4_TRAIN_DIR}"
run_one 8 "${K8_TRAIN_DIR}"

echo "===== requested K4/K8 full eval sequence finished at $(date) ====="
