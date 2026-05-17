#!/usr/bin/env bash
set -euo pipefail

# Partial LIBERO eval for the v5 d4,6,10 coeff-pool LP-MT residual p4 checkpoint.
# Evaluates 10 episodes x 10 tasks x 4 suites on GPUs 4-7.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"

POLICY_PATH="${POLICY_PATH:-/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_v5_d4_6_10_coeffpool_full_ce_mean_k10_p4_f15_bigflow_b128_s2_daw0p5_rot10_20260511_083520/checkpoints/last/pretrained_model}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p4_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p4_f15_canonical_lp_mt.summary.json}"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-job1_lpmt_v5_d4_6_10_residual_coeffpool_p4_daw0p5_rot10_k10_f15_10eps_bs${EVAL_BATCH_SIZE}}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Missing required directory: ${path}" >&2
    exit 1
  fi
}

echo "===== v5 p4 residual coeff-pool DAW0.5 eval started at $(date) ====="
echo "TIMESTAMP=${TIMESTAMP}"
echo "GPU_IDS=${GPU_IDS}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "POLICY_PATH=${POLICY_PATH}"
echo "HIVA_COEFF_SIDECAR=${HIVA_COEFF_SIDECAR}"
echo "HIVA_COEFF_SUMMARY=${HIVA_COEFF_SUMMARY}"

require_dir "${POLICY_PATH}"
require_file "${HIVA_COEFF_SIDECAR}"
require_file "${HIVA_COEFF_SUMMARY}"

POLICY_PATH="${POLICY_PATH}" \
CHECKPOINT_LABEL="${CHECKPOINT_LABEL}" \
TIMESTAMP="${TIMESTAMP}_job1" \
GPU_IDS="${GPU_IDS}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR}" \
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY}" \
CHUNK_SIZE=15 \
N_ACTION_STEPS=10 \
HIVA_DURATION_EXECUTION_MAP="" \
bash "${SCRIPT_DIR}/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"

echo "===== v5 p4 residual coeff-pool DAW0.5 eval finished at $(date) ====="
