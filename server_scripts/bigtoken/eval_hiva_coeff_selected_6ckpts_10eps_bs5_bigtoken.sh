#!/usr/bin/env bash
set -euo pipefail

# Partial LIBERO eval for six selected HiVA coefficient checkpoints.
# Runs two checkpoints per wave. Each checkpoint uses four GPUs, one suite per
# GPU, via eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-5}"
TASK_IDS_ALL="${TASK_IDS_ALL:-[0,1,2,3,4,5,6,7,8,9]}"
N_EPISODES="${N_EPISODES:-10}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-1}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea/add_disk2/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"

HELPER="${SCRIPT_DIR}/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"

MT_D2_15_K10_SIDECAR="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_canonical_mt.parquet"
MT_D2_15_K10_SUMMARY="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_canonical_mt.summary.json"
LPMT_D2_10_K10_F15_SIDECAR="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_10_w1_10_w3_0_k10_f15_canonical_lp_mt.parquet"
LPMT_D2_10_K10_F15_SUMMARY="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_10_w1_10_w3_0_k10_f15_canonical_lp_mt.summary.json"
LPMT_D2_6_K10_F15_SIDECAR="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_6_w1_10_w3_0_k10_f15_canonical_lp_mt.parquet"
LPMT_D2_6_K10_F15_SUMMARY="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_6_w1_10_w3_0_k10_f15_canonical_lp_mt.summary.json"

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

run_eval() {
  local wave="$1"
  local slot="$2"
  local gpu_ids="$3"
  local policy_path="$4"
  local checkpoint_label="$5"
  local sidecar="$6"
  local summary="$7"
  local chunk_size="$8"
  local n_action_steps="$9"

  require_dir "${policy_path}"
  require_file "${sidecar}"
  require_file "${summary}"

  echo "===== ${wave}/${slot}: ${checkpoint_label} ====="
  echo "GPUs=${gpu_ids}"
  echo "POLICY_PATH=${policy_path}"
  echo "HIVA_COEFF_SIDECAR=${sidecar}"
  echo "HIVA_COEFF_SUMMARY=${summary}"
  echo "CHUNK_SIZE=${chunk_size}"
  echo "N_ACTION_STEPS=${n_action_steps}"

  POLICY_PATH="${policy_path}" \
  CHECKPOINT_LABEL="${checkpoint_label}" \
  TIMESTAMP="${TIMESTAMP}_${wave}_${slot}" \
  GPU_IDS="${gpu_ids}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  TASK_IDS_ALL="${TASK_IDS_ALL}" \
  N_EPISODES="${N_EPISODES}" \
  MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS}" \
  DATA_ROOT="${DATA_ROOT}" \
  HIVA_COEFF_SIDECAR="${sidecar}" \
  HIVA_COEFF_SUMMARY="${summary}" \
  CHUNK_SIZE="${chunk_size}" \
  N_ACTION_STEPS="${n_action_steps}" \
  bash "${HELPER}"
}

run_wave() {
  local wave="$1"
  shift

  echo "===== Starting ${wave} at $(date) ====="
  run_eval "${wave}" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" &
  local pid_a="$!"
  shift 8
  run_eval "${wave}" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" &
  local pid_b="$!"

  local status=0
  if ! wait "${pid_a}"; then
    status=1
  fi
  if ! wait "${pid_b}"; then
    status=1
  fi
  if [[ "${status}" -ne 0 ]]; then
    echo "${wave} failed." >&2
    exit "${status}"
  fi
  echo "===== Finished ${wave} at $(date) ====="
}

CKPT_MT_COEFFPOOL="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_mt_d2_15_w1_10_w3_0_coeffpool_full_ce_mean_k10_bigcornea_b64_s2_baseinit_20260509_024247/checkpoints/last/pretrained_model"
CKPT_LPMT_D2_10="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_d2_10_w1_10_w3_0_residual_ffn_duration_reads_coeffs_ce_mean_k10_f15_bigcornea_b64_s2_baseinit_20260509_024247/checkpoints/last/pretrained_model"
CKPT_LPMT_D2_6="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_d2_6_w1_10_w3_0_residual_ffn_duration_reads_coeffs_ce_mean_k10_f15_bigcornea_b64_s2_baseinit_20260509_024247/checkpoints/last/pretrained_model"
CKPT_MT_DECODED_DAW0P1="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_mt_decodedloss_d2_15_residual_ffn_duration_reads_coeffs_ce_mean_k10_f15_bigflow_b128_s2_daw0p1_rot10_20260509_002956/checkpoints/last/pretrained_model"
CKPT_MT_DECODED_DAW0P3="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_mt_decodedloss_d2_15_residual_ffn_duration_reads_coeffs_ce_mean_k10_f15_bigflow_b128_s2_daw0p3_rot10_20260509_002956/checkpoints/last/pretrained_model"
CKPT_MT_DECODED_DAW0P5="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_mt_decodedloss_d2_15_residual_ffn_duration_reads_coeffs_ce_mean_k10_f15_bigflow_b128_s2_daw0p5_rot10_20260509_002956/checkpoints/last/pretrained_model"

require_file "${HELPER}"
require_dir "${DATA_ROOT}"

echo "===== Selected HiVA coefficient eval queue started at $(date) ====="
echo "TIMESTAMP=${TIMESTAMP}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "TASK_IDS_ALL=${TASK_IDS_ALL}"
echo "N_EPISODES=${N_EPISODES}"
echo "MAX_PARALLEL_TASKS=${MAX_PARALLEL_TASKS}"
echo "SIDECAR_ROOT=${SIDECAR_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"

run_wave "wave1" \
  "gpu0_3_mt_coeffpool" "0,1,2,3" "${CKPT_MT_COEFFPOOL}" "mt_d2_15_coeffpool_full_ce_mean_k10_baseinit_10eps_bs${EVAL_BATCH_SIZE}" "${MT_D2_15_K10_SIDECAR}" "${MT_D2_15_K10_SUMMARY}" 15 15 \
  "gpu4_7_lpmt_d2_10" "4,5,6,7" "${CKPT_LPMT_D2_10}" "lpmt_d2_10_residual_ffn_reads_coeffs_ce_mean_k10_f15_baseinit_10eps_bs${EVAL_BATCH_SIZE}" "${LPMT_D2_10_K10_F15_SIDECAR}" "${LPMT_D2_10_K10_F15_SUMMARY}" 15 10

run_wave "wave2" \
  "gpu0_3_lpmt_d2_6" "0,1,2,3" "${CKPT_LPMT_D2_6}" "lpmt_d2_6_residual_ffn_reads_coeffs_ce_mean_k10_f15_baseinit_10eps_bs${EVAL_BATCH_SIZE}" "${LPMT_D2_6_K10_F15_SIDECAR}" "${LPMT_D2_6_K10_F15_SUMMARY}" 15 6 \
  "gpu4_7_mt_decoded_daw0p1" "4,5,6,7" "${CKPT_MT_DECODED_DAW0P1}" "mt_decodedloss_d2_15_daw0p1_rot10_k10_f15_10eps_bs${EVAL_BATCH_SIZE}" "${MT_D2_15_K10_SIDECAR}" "${MT_D2_15_K10_SUMMARY}" 15 15

run_wave "wave3" \
  "gpu0_3_mt_decoded_daw0p3" "0,1,2,3" "${CKPT_MT_DECODED_DAW0P3}" "mt_decodedloss_d2_15_daw0p3_rot10_k10_f15_10eps_bs${EVAL_BATCH_SIZE}" "${MT_D2_15_K10_SIDECAR}" "${MT_D2_15_K10_SUMMARY}" 15 15 \
  "gpu4_7_mt_decoded_daw0p5" "4,5,6,7" "${CKPT_MT_DECODED_DAW0P5}" "mt_decodedloss_d2_15_daw0p5_rot10_k10_f15_10eps_bs${EVAL_BATCH_SIZE}" "${MT_D2_15_K10_SIDECAR}" "${MT_D2_15_K10_SUMMARY}" 15 15

echo "===== Selected HiVA coefficient eval queue finished at $(date) ====="
