#!/usr/bin/env bash
set -euo pipefail

# Resume the unfinished residual-flow checkpoint evals on bigbrain.
#
# This intentionally keeps the full train directory basename in every eval
# directory so results can be traced back to the source model without a lookup.
# Checkpoints whose "last" symlink points to an already-requested numeric
# checkpoint are listed only once here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-6,7}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigbrain/add_disk0/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"

LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/eval_residual_flow_unfinished_bigbrain_gpu6_7_bs${EVAL_BATCH_SIZE}_${TIMESTAMP}.queue.log}"
RUNNER="${RUNNER:-${SCRIPT_DIR}/eval_lpmt_stage0_v5_all_ckpts_partial_gpu6_7_bs10_bigbrain.sh}"

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Missing required directory: ${path}" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

run_group() {
  local train_dir="$1"
  shift
  local ckpts=("$@")
  local model_name
  model_name="$(basename "${train_dir}")"
  local eval_dir="${REPO_ROOT}/outputs/eval/bigbrain_resume_${model_name}_10eps_bs${EVAL_BATCH_SIZE}_${TIMESTAMP}"

  require_dir "${train_dir}/checkpoints"

  echo "===== $(date) starting ${model_name} ====="
  echo "TRAIN_DIR=${train_dir}"
  echo "CKPTS=${ckpts[*]}"
  echo "EVAL_DIR=${eval_dir}"

  TRAIN_DIR="${train_dir}" \
  MODEL_TAG="${model_name}" \
  CKPTS_OVERRIDE="${ckpts[*]}" \
  GPU_IDS="${GPU_IDS}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  EVAL_CHECKPOINTS_IN_PARALLEL=1 \
  DATA_ROOT="${DATA_ROOT}" \
  HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR}" \
  HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY}" \
  SWEEP_OUTPUT_DIR="${eval_dir}" \
  TIMESTAMP="${TIMESTAMP}" \
  bash "${RUNNER}"

  echo "===== $(date) finished ${model_name} ====="
}

main() {
  exec > >(tee -a "${QUEUE_LOG}") 2>&1

  require_file "${RUNNER}"
  require_dir "${DATA_ROOT}"
  require_file "${HIVA_COEFF_SIDECAR}"
  require_file "${HIVA_COEFF_SUMMARY}"

  echo "===== residual-flow unfinished BigBrain eval queue started at $(date) ====="
  echo "TIMESTAMP=${TIMESTAMP}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "HIVA_COEFF_SIDECAR=${HIVA_COEFF_SIDECAR}"
  echo "HIVA_COEFF_SUMMARY=${HIVA_COEFF_SUMMARY}"
  echo "RUNNER=${RUNNER}"
  echo "QUEUE_LOG=${QUEUE_LOG}"

  # daw2p0: finished elsewhere: 000063, 000157, 000313.
  # last -> 000625, so "last" is omitted to avoid duplicate eval.
  run_group \
    "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_flow_stage1_v5_d4_6_10_k10_f15_tr3p0_rot3p0_grip0p5_daw2p0_b128_g8_s2_20260514_040554" \
    000469 000625

  # daw0p5: all queued checkpoints were unfinished elsewhere.
  # last -> 000625, so "last" is omitted to avoid duplicate eval.
  run_group \
    "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_flow_stage1_v5_d4_6_10_k10_f15_tr3p0_rot3p0_grip0p5_daw0p5_b128_g8_s2_20260514_040554" \
    000063 000157 000313 000469 000625

  # trrot4: finished elsewhere: 000021, 000053.
  # last -> 000209, so "last" is omitted to avoid duplicate eval.
  run_group \
    "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_residual_flow_stage1_dualexpert_v5_d4_6_10_trrot4_grip0p5_b768_g4_s2_20260514_040613" \
    000105 000157 000209

  # trrot2 was not queued in the active remote resume job.
  # last -> 000209, so "last" is omitted to avoid duplicate eval.
  run_group \
    "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_residual_flow_stage1_dualexpert_v5_d4_6_10_trrot2_grip0p5_b768_g4_s2_20260514_040613" \
    000021 000053 000105 000157 000209

  echo "===== residual-flow unfinished BigBrain eval queue finished at $(date) ====="
}

main "$@"
