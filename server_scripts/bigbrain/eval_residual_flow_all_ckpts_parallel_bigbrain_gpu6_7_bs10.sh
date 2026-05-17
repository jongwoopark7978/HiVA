#!/usr/bin/env bash
set -euo pipefail

# Fan out residual-flow checkpoint evals immediately across GPUs 6 and 7.
# Each checkpoint gets its own runner process. Eval directories keep the full
# train directory basename so results remain traceable.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-6,7}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigbrain/add_disk0/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"

LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/eval_residual_flow_all_ckpts_parallel_bigbrain_gpu6_7_bs${EVAL_BATCH_SIZE}_${TIMESTAMP}.queue.log}"
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

launch_ckpt() {
  local train_dir="$1"
  local ckpt="$2"
  local gpu_id="$3"
  local model_name
  model_name="$(basename "${train_dir}")"
  local eval_dir="${REPO_ROOT}/outputs/eval/bigbrain_parallel_${model_name}_10eps_bs${EVAL_BATCH_SIZE}_${TIMESTAMP}"
  local outer_log="${LOG_DIR}/eval_bigbrain_parallel_${model_name}_ckpt_${ckpt}_gpu${gpu_id}_${TIMESTAMP}.outer.log"

  require_dir "${train_dir}/checkpoints/${ckpt}/pretrained_model"
  mkdir -p "${eval_dir}"

  echo "===== $(date) launching ${model_name} ckpt_${ckpt} on GPU ${gpu_id} ====="
  echo "TRAIN_DIR=${train_dir}"
  echo "EVAL_DIR=${eval_dir}"
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
  TIMESTAMP="${TIMESTAMP}" \
  bash "${RUNNER}" > "${outer_log}" 2>&1 &
}

main() {
  exec > >(tee -a "${QUEUE_LOG}") 2>&1

  require_file "${RUNNER}"
  require_dir "${DATA_ROOT}"
  require_file "${HIVA_COEFF_SIDECAR}"
  require_file "${HIVA_COEFF_SUMMARY}"

  local jobs=(
    "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_flow_stage1_v5_d4_6_10_k10_f15_tr3p0_rot3p0_grip0p5_daw2p0_b128_g8_s2_20260514_040554|000469"
    "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_flow_stage1_v5_d4_6_10_k10_f15_tr3p0_rot3p0_grip0p5_daw2p0_b128_g8_s2_20260514_040554|000625"
    "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_flow_stage1_v5_d4_6_10_k10_f15_tr3p0_rot3p0_grip0p5_daw0p5_b128_g8_s2_20260514_040554|000063"
    "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_flow_stage1_v5_d4_6_10_k10_f15_tr3p0_rot3p0_grip0p5_daw0p5_b128_g8_s2_20260514_040554|000157"
    "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_flow_stage1_v5_d4_6_10_k10_f15_tr3p0_rot3p0_grip0p5_daw0p5_b128_g8_s2_20260514_040554|000313"
    "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_flow_stage1_v5_d4_6_10_k10_f15_tr3p0_rot3p0_grip0p5_daw0p5_b128_g8_s2_20260514_040554|000469"
    "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_flow_stage1_v5_d4_6_10_k10_f15_tr3p0_rot3p0_grip0p5_daw0p5_b128_g8_s2_20260514_040554|000625"
    "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_residual_flow_stage1_dualexpert_v5_d4_6_10_trrot4_grip0p5_b768_g4_s2_20260514_040613|000105"
    "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_residual_flow_stage1_dualexpert_v5_d4_6_10_trrot4_grip0p5_b768_g4_s2_20260514_040613|000157"
    "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_residual_flow_stage1_dualexpert_v5_d4_6_10_trrot4_grip0p5_b768_g4_s2_20260514_040613|000209"
    "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_residual_flow_stage1_dualexpert_v5_d4_6_10_trrot2_grip0p5_b768_g4_s2_20260514_040613|000021"
    "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_residual_flow_stage1_dualexpert_v5_d4_6_10_trrot2_grip0p5_b768_g4_s2_20260514_040613|000053"
    "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_residual_flow_stage1_dualexpert_v5_d4_6_10_trrot2_grip0p5_b768_g4_s2_20260514_040613|000105"
    "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_residual_flow_stage1_dualexpert_v5_d4_6_10_trrot2_grip0p5_b768_g4_s2_20260514_040613|000157"
    "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_residual_flow_stage1_dualexpert_v5_d4_6_10_trrot2_grip0p5_b768_g4_s2_20260514_040613|000209"
  )

  echo "===== residual-flow all-checkpoint parallel BigBrain eval started at $(date) ====="
  echo "TIMESTAMP=${TIMESTAMP}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
  echo "N_JOBS=${#jobs[@]}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "HIVA_COEFF_SIDECAR=${HIVA_COEFF_SIDECAR}"
  echo "HIVA_COEFF_SUMMARY=${HIVA_COEFF_SUMMARY}"
  echo "QUEUE_LOG=${QUEUE_LOG}"

  local pids=()
  local idx job train_dir ckpt gpu_id pid status=0
  for idx in "${!jobs[@]}"; do
    job="${jobs[$idx]}"
    IFS='|' read -r train_dir ckpt <<< "${job}"
    gpu_id="${GPU_ARRAY[$((idx % ${#GPU_ARRAY[@]}))]}"
    launch_ckpt "${train_dir}" "${ckpt}" "${gpu_id}"
    pids+=("$!")
  done

  echo "LAUNCHED_PIDS=${pids[*]}"
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done

  if [[ "${status}" -ne 0 ]]; then
    echo "One or more parallel checkpoint evals failed." >&2
    exit "${status}"
  fi
  echo "===== residual-flow all-checkpoint parallel BigBrain eval finished at $(date) ====="
}

main "$@"
