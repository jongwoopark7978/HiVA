#!/usr/bin/env bash
set -euo pipefail

# Sweep inference-time residual weight for:
#   final_action = base_action + lambda * residual_action
# on the LP-MT residual xattn scale3p0/daw1p0 checkpoint.
#
# Each lambda runs 10 episodes x 10 tasks x 4 LIBERO suites.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark}"

POLICY_PATH="${POLICY_PATH:-/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_stage1_xattn_b9_v5_d4_6_10_k10_f15_scale3p0_daw1p0_b128_s2_20260511_214458_scale3p0_daw1p0/checkpoints/last/pretrained_model}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"
LAMBDAS=(${LAMBDAS:-0 0.25 0.5 0.75})

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

lambda_label() {
  local value="$1"
  echo "${value//./p}"
}

require_dir "${POLICY_PATH}"
require_file "${HIVA_COEFF_SIDECAR}"
require_file "${HIVA_COEFF_SUMMARY}"

echo "===== residual lambda sweep started at $(date) ====="
echo "TIMESTAMP=${TIMESTAMP}"
echo "GPU_IDS=${GPU_IDS}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "POLICY_PATH=${POLICY_PATH}"
echo "HIVA_COEFF_SIDECAR=${HIVA_COEFF_SIDECAR}"
echo "HIVA_COEFF_SUMMARY=${HIVA_COEFF_SUMMARY}"
echo "LAMBDAS=${LAMBDAS[*]}"

for lambda in "${LAMBDAS[@]}"; do
  label="$(lambda_label "${lambda}")"
  checkpoint_label="lambda${label}_lpmt_stage1_xattn_b9_v5_d4_6_10_scale3p0_daw1p0_k10_f15_10eps_bs${EVAL_BATCH_SIZE}"
  echo "===== lambda=${lambda} checkpoint_label=${checkpoint_label} at $(date) ====="

  POLICY_PATH="${POLICY_PATH}" \
  CHECKPOINT_LABEL="${checkpoint_label}" \
  TIMESTAMP="${TIMESTAMP}_lambda${label}" \
  GPU_IDS="${GPU_IDS}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR}" \
  HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY}" \
  HIVA_RESIDUAL_INFERENCE_WEIGHT="${lambda}" \
  CHUNK_SIZE=15 \
  N_ACTION_STEPS=10 \
  HIVA_DURATION_EXECUTION_MAP="" \
  bash "${SCRIPT_DIR}/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"
done

echo "===== residual lambda sweep finished at $(date) ====="
