#!/usr/bin/env bash
set -euo pipefail

# Queue a partial LIBERO eval for the LP-MT residual xattn b9 scale=1.0,
# decoded-action-weight=1.5 checkpoint after the current GPU4-7 sweep finishes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
WAIT_PATTERN="${WAIT_PATTERN:-eval_lpmt_xattn_sweep_and_p5_p6_nores_10eps_bigtoken.sh}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark}"

POLICY_PATH="${POLICY_PATH:-/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_stage1_xattn_b9_v5_d4_6_10_k10_f15_scale1p0_daw1p5_b128_s2_20260511_214458_scale1p0_daw1p5/checkpoints/last/pretrained_model}"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-job1_lpmt_stage1_xattn_b9_v5_d4_6_10_scale1p0_daw1p5_b128_k10_f15_10eps_bs${EVAL_BATCH_SIZE}}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"

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

require_dir "${POLICY_PATH}"
require_file "${HIVA_COEFF_SIDECAR}"
require_file "${HIVA_COEFF_SUMMARY}"

echo "===== queued LP-MT xattn daw1p5 eval started at $(date) ====="
echo "TIMESTAMP=${TIMESTAMP}"
echo "WAIT_PATTERN=${WAIT_PATTERN}"
echo "GPU_IDS=${GPU_IDS}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "POLICY_PATH=${POLICY_PATH}"
echo "CHECKPOINT_LABEL=${CHECKPOINT_LABEL}"
echo "HIVA_COEFF_SIDECAR=${HIVA_COEFF_SIDECAR}"
echo "HIVA_COEFF_SUMMARY=${HIVA_COEFF_SUMMARY}"

while pgrep -f "${WAIT_PATTERN}" >/dev/null; do
  echo "[$(date)] Waiting for existing queued eval jobs matching: ${WAIT_PATTERN}"
  pgrep -af "${WAIT_PATTERN}" || true
  sleep 300
done

echo "[$(date)] Existing queue finished; launching ${CHECKPOINT_LABEL}"

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

echo "===== queued LP-MT xattn daw1p5 eval finished at $(date) ====="
