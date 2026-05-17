#!/usr/bin/env bash
set -euo pipefail

# Continue the selected six-checkpoint bs4 eval after wave1's GPU4-7 slot
# finished early. This avoids the original wave barrier by running two
# independent GPU slots.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
TASK_IDS_ALL="${TASK_IDS_ALL:-[0,1,2,3,4,5,6,7,8,9]}"
N_EPISODES="${N_EPISODES:-10}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-1}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea/add_disk2/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
HELPER="${SCRIPT_DIR}/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"

WAVE1_MT_SUMMARY="${WAVE1_MT_SUMMARY:-/home/jongwoopark/lerobot/outputs/eval/full_bigtoken_mt_d2_15_coeffpool_full_ce_mean_k10_baseinit_10eps_bs4_20260509_195343_wave1_gpu0_3_mt_coeffpool/overlay_eval_summary.json}"

MT_D2_15_K10_SIDECAR="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_canonical_mt.parquet"
MT_D2_15_K10_SUMMARY="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_canonical_mt.summary.json"
LPMT_D2_6_K10_F15_SIDECAR="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_6_w1_10_w3_0_k10_f15_canonical_lp_mt.parquet"
LPMT_D2_6_K10_F15_SUMMARY="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_6_w1_10_w3_0_k10_f15_canonical_lp_mt.summary.json"

CKPT_LPMT_D2_6="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_d2_6_w1_10_w3_0_residual_ffn_duration_reads_coeffs_ce_mean_k10_f15_bigcornea_b64_s2_baseinit_20260509_024247/checkpoints/last/pretrained_model"
CKPT_MT_DECODED_DAW0P1="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_mt_decodedloss_d2_15_residual_ffn_duration_reads_coeffs_ce_mean_k10_f15_bigflow_b128_s2_daw0p1_rot10_20260509_002956/checkpoints/last/pretrained_model"
CKPT_MT_DECODED_DAW0P3="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_mt_decodedloss_d2_15_residual_ffn_duration_reads_coeffs_ce_mean_k10_f15_bigflow_b128_s2_daw0p3_rot10_20260509_002956/checkpoints/last/pretrained_model"
CKPT_MT_DECODED_DAW0P5="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_mt_decodedloss_d2_15_residual_ffn_duration_reads_coeffs_ce_mean_k10_f15_bigflow_b128_s2_daw0p5_rot10_20260509_002956/checkpoints/last/pretrained_model"

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
  local gpu_ids="$1"
  local policy_path="$2"
  local checkpoint_label="$3"
  local sidecar="$4"
  local summary="$5"
  local chunk_size="$6"
  local n_action_steps="$7"

  require_dir "${policy_path}"
  require_file "${sidecar}"
  require_file "${summary}"

  echo "===== Starting ${checkpoint_label} on GPUs ${gpu_ids} at $(date) ====="
  POLICY_PATH="${policy_path}" \
  CHECKPOINT_LABEL="${checkpoint_label}" \
  TIMESTAMP="${TIMESTAMP}" \
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
  echo "===== Finished ${checkpoint_label} on GPUs ${gpu_ids} at $(date) ====="
}

wait_for_wave1_mt() {
  echo "Waiting for wave1 MT coeff-pool summary: ${WAVE1_MT_SUMMARY}"
  while [[ ! -f "${WAVE1_MT_SUMMARY}" ]]; do
    if ! pgrep -f "hiva_coeff_bigtoken_mt_d2_15_coeffpool_full_ce_mean_k10_baseinit_10eps_bs4.*20260509_195343_wave1_gpu0_3_mt_coeffpool" >/dev/null; then
      echo "Wave1 MT process is gone but summary is missing: ${WAVE1_MT_SUMMARY}" >&2
      exit 1
    fi
    sleep 60
  done
  echo "Wave1 MT summary exists at $(date)."
}

slot_gpu4_7() {
  run_eval "4,5,6,7" "${CKPT_LPMT_D2_6}" \
    "lpmt_d2_6_residual_ffn_reads_coeffs_ce_mean_k10_f15_baseinit_10eps_bs${EVAL_BATCH_SIZE}_slot4_7" \
    "${LPMT_D2_6_K10_F15_SIDECAR}" "${LPMT_D2_6_K10_F15_SUMMARY}" 15 6

  run_eval "4,5,6,7" "${CKPT_MT_DECODED_DAW0P3}" \
    "mt_decodedloss_d2_15_daw0p3_rot10_k10_f15_10eps_bs${EVAL_BATCH_SIZE}_slot4_7" \
    "${MT_D2_15_K10_SIDECAR}" "${MT_D2_15_K10_SUMMARY}" 15 15
}

slot_gpu0_3() {
  wait_for_wave1_mt

  run_eval "0,1,2,3" "${CKPT_MT_DECODED_DAW0P1}" \
    "mt_decodedloss_d2_15_daw0p1_rot10_k10_f15_10eps_bs${EVAL_BATCH_SIZE}_slot0_3" \
    "${MT_D2_15_K10_SIDECAR}" "${MT_D2_15_K10_SUMMARY}" 15 15

  run_eval "0,1,2,3" "${CKPT_MT_DECODED_DAW0P5}" \
    "mt_decodedloss_d2_15_daw0p5_rot10_k10_f15_10eps_bs${EVAL_BATCH_SIZE}_slot0_3" \
    "${MT_D2_15_K10_SIDECAR}" "${MT_D2_15_K10_SUMMARY}" 15 15
}

require_file "${HELPER}"
require_dir "${DATA_ROOT}"

echo "===== Remaining selected HiVA eval slots started at $(date) ====="
echo "TIMESTAMP=${TIMESTAMP}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "TASK_IDS_ALL=${TASK_IDS_ALL}"
echo "N_EPISODES=${N_EPISODES}"
echo "SIDECAR_ROOT=${SIDECAR_ROOT}"

slot_gpu4_7 &
pid_47="$!"
slot_gpu0_3 &
pid_03="$!"

status=0
if ! wait "${pid_47}"; then
  status=1
fi
if ! wait "${pid_03}"; then
  status=1
fi

if [[ "${status}" -ne 0 ]]; then
  echo "Remaining selected HiVA eval slots failed." >&2
  exit "${status}"
fi

echo "===== Remaining selected HiVA eval slots finished at $(date) ====="
