#!/usr/bin/env bash
set -euo pipefail

# Resume only the incomplete BigBrain residual-flow eval chunks.
# The underlying runner skips any task directory that already has eval_info.json,
# and this wrapper also narrows suite/task ids to the known missing chunks.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"
RUNNER="${RUNNER:-${SCRIPT_DIR}/eval_lpmt_stage0_v5_all_ckpts_partial_gpu6_7_bs10_bigbrain.sh}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigbrain/add_disk0/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"

LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/resume_crashed_residual_evals_gpu6_7_bigbrain_bs${EVAL_BATCH_SIZE}_${TIMESTAMP}.queue.log}"

ALL_IDS='[0,1,2,3,4,5,6,7,8,9]'
OBJ_8_9='[8,9]'
SPATIAL_8_9='[8,9]'
EMPTY_IDS='[]'

DAW2_TRAIN="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_flow_stage1_v5_d4_6_10_k10_f15_tr3p0_rot3p0_grip0p5_daw2p0_b128_g8_s2_20260514_040554"
TRROT4_TRAIN="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_residual_flow_stage1_dualexpert_v5_d4_6_10_trrot4_grip0p5_b768_g4_s2_20260514_040613"
TRROT2_TRAIN="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_residual_flow_stage1_dualexpert_v5_d4_6_10_trrot2_grip0p5_b768_g4_s2_20260514_040613"
TRROT5_TRAIN="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_residual_flow_stage1_dualexpert_v5_d4_6_10_trrot5_grip0p5_b768_g4_s2_20260514_040613"

DAW2_EVAL="${REPO_ROOT}/outputs/eval/bigbrain_parallel_$(basename "${DAW2_TRAIN}")_10eps_bs10_20260515_003802"
TRROT4_EVAL="${REPO_ROOT}/outputs/eval/bigbrain_parallel_$(basename "${TRROT4_TRAIN}")_10eps_bs10_20260515_003802"
TRROT2_EVAL="${REPO_ROOT}/outputs/eval/bigbrain_parallel_$(basename "${TRROT2_TRAIN}")_10eps_bs10_20260515_003802"
TRROT5_EVAL="${REPO_ROOT}/outputs/eval/bigbrain_parallel_$(basename "${TRROT5_TRAIN}")_10eps_bs10_20260515_004608"

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

run_resume_chunk() {
  local label="$1"
  local gpu_id="$2"
  local train_dir="$3"
  local ckpt="$4"
  local eval_dir="$5"
  local suites_csv="$6"
  local object_ids="$7"
  local goal_ids="$8"
  local spatial_ids="$9"
  local libero10_ids="${10}"

  local model_name
  model_name="$(basename "${train_dir}")"
  local outer_log="${LOG_DIR}/resume_${label}_ckpt_${ckpt}_gpu${gpu_id}_${TIMESTAMP}.outer.log"

  require_dir "${train_dir}/checkpoints/${ckpt}/pretrained_model"
  require_dir "${eval_dir}"

  echo "===== $(date) resume ${label} ckpt_${ckpt} on GPU ${gpu_id} ====="
  echo "TRAIN_DIR=${train_dir}"
  echo "EVAL_DIR=${eval_dir}"
  echo "SUITES_CSV=${suites_csv}"
  echo "OBJECT_TASK_IDS=${object_ids}"
  echo "GOAL_TASK_IDS=${goal_ids}"
  echo "SPATIAL_TASK_IDS=${spatial_ids}"
  echo "LIBERO10_TASK_IDS=${libero10_ids}"
  echo "OUTER_LOG=${outer_log}"

  TRAIN_DIR="${train_dir}" \
  MODEL_TAG="${model_name}" \
  CKPTS_OVERRIDE="${ckpt}" \
  GPU_IDS="${gpu_id}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  EVAL_CHECKPOINTS_IN_PARALLEL=1 \
  DATA_ROOT="${DATA_ROOT}" \
  HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR}" \
  HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY}" \
  SWEEP_OUTPUT_DIR="${eval_dir}" \
  SUITES_CSV="${suites_csv}" \
  TASK_IDS_ALL="${ALL_IDS}" \
  OBJECT_TASK_IDS="${object_ids}" \
  GOAL_TASK_IDS="${goal_ids}" \
  SPATIAL_TASK_IDS="${spatial_ids}" \
  LIBERO10_TASK_IDS="${libero10_ids}" \
  TIMESTAMP="${TIMESTAMP}" \
  bash "${RUNNER}" > "${outer_log}" 2>&1
}

gpu6_queue() {
  run_resume_chunk "daw2p0" "6" "${DAW2_TRAIN}" "000469" "${DAW2_EVAL}" \
    "libero_10" "${EMPTY_IDS}" "${EMPTY_IDS}" "${EMPTY_IDS}" "${ALL_IDS}"
  run_resume_chunk "trrot2" "6" "${TRROT2_TRAIN}" "000021" "${TRROT2_EVAL}" \
    "libero_object,libero_goal,libero_spatial,libero_10" "${OBJ_8_9}" "${ALL_IDS}" "${ALL_IDS}" "${ALL_IDS}"
  run_resume_chunk "trrot5" "6" "${TRROT5_TRAIN}" "000157" "${TRROT5_EVAL}" \
    "libero_spatial,libero_10" "${EMPTY_IDS}" "${EMPTY_IDS}" "${SPATIAL_8_9}" "${ALL_IDS}"
}

gpu7_queue() {
  run_resume_chunk "trrot4" "7" "${TRROT4_TRAIN}" "000209" "${TRROT4_EVAL}" \
    "libero_spatial,libero_10" "${EMPTY_IDS}" "${EMPTY_IDS}" "${SPATIAL_8_9}" "${ALL_IDS}"
  run_resume_chunk "trrot2" "7" "${TRROT2_TRAIN}" "000209" "${TRROT2_EVAL}" \
    "libero_10" "${EMPTY_IDS}" "${EMPTY_IDS}" "${EMPTY_IDS}" "${ALL_IDS}"
  run_resume_chunk "trrot5" "7" "${TRROT5_TRAIN}" "000209" "${TRROT5_EVAL}" \
    "libero_object,libero_goal,libero_spatial,libero_10" "${OBJ_8_9}" "${ALL_IDS}" "${ALL_IDS}" "${ALL_IDS}"
}

main() {
  exec > >(tee -a "${QUEUE_LOG}") 2>&1

  require_file "${RUNNER}"
  require_dir "${DATA_ROOT}"
  require_file "${HIVA_COEFF_SIDECAR}"
  require_file "${HIVA_COEFF_SUMMARY}"

  echo "===== resume crashed residual evals started at $(date) ====="
  echo "TIMESTAMP=${TIMESTAMP}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
  echo "QUEUE_LOG=${QUEUE_LOG}"

  local pid6 pid7 status=0
  gpu6_queue &
  pid6="$!"
  gpu7_queue &
  pid7="$!"
  echo "GPU6_QUEUE_PID=${pid6}"
  echo "GPU7_QUEUE_PID=${pid7}"

  if ! wait "${pid6}"; then
    status=1
  fi
  if ! wait "${pid7}"; then
    status=1
  fi

  if [[ "${status}" -ne 0 ]]; then
    echo "One or more resume chunks failed." >&2
    exit "${status}"
  fi

  echo "===== resume crashed residual evals finished at $(date) ====="
}

main "$@"
