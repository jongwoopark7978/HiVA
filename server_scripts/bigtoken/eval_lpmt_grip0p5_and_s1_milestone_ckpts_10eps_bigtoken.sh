#!/usr/bin/env bash
set -euo pipefail

# Sequential partial LIBERO evals:
# 1) grip0p5 LP-MT xattn checkpoint
# 2) all saved milestone checkpoints for grip0p3 S=1 model
# 3) all saved milestone checkpoints for grip0 S=1 model
#
# Milestone checkpoint evals are grouped under one parent eval directory per model.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark}"

SIDECAR_K10_F15="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet"
SUMMARY_K10_F15="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json"

GRIP0P5_POLICY="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_stage1_xattn_job1_gripsweep_v5_d4_6_10_tr3_rot3_grip0p5_daw1_betas_0p1_0p05_0p1_b1024_g3_s2_20260512_202224/checkpoints/last/pretrained_model"

MILESTONE_GRIP0P3="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_stage1_xattn_s1_milestone_v5_d4_6_10_tr3_rot3_grip0p3_daw1_b1024_g4_s1_20260513_000034"
MILESTONE_GRIP0="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_stage1_xattn_s1_milestone_v5_d4_6_10_tr3_rot3_grip0_daw1_b1024_g4_s1_20260513_001728"
MILESTONE_CKPTS=(000063 000079 000094 000110 last)

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

run_eval() {
  local policy_path="$1"
  local checkpoint_label="$2"
  local timestamp_suffix="$3"
  local base_output_dir="${4:-}"
  local suites_csv="${5:-libero_object,libero_goal,libero_spatial,libero_10}"

  if [[ -n "${base_output_dir}" && -f "${base_output_dir}/overlay_eval_summary.json" ]]; then
    echo "===== Skipping completed ${checkpoint_label}: ${base_output_dir} ====="
    return
  fi

  require_dir "${policy_path}"
  require_file "${SIDECAR_K10_F15}"
  require_file "${SUMMARY_K10_F15}"

  echo "===== ${checkpoint_label} at $(date) ====="
  echo "POLICY_PATH=${policy_path}"
  echo "HIVA_COEFF_SIDECAR=${SIDECAR_K10_F15}"
  echo "HIVA_COEFF_SUMMARY=${SUMMARY_K10_F15}"
  if [[ -n "${base_output_dir}" ]]; then
    echo "BASE_OUTPUT_DIR=${base_output_dir}"
  fi
  echo "SUITES_CSV=${suites_csv}"

  if [[ -n "${base_output_dir}" ]]; then
    BASE_OUTPUT_DIR="${base_output_dir}" \
    POLICY_PATH="${policy_path}" \
    CHECKPOINT_LABEL="${checkpoint_label}" \
    TIMESTAMP="${TIMESTAMP}_${timestamp_suffix}" \
    GPU_IDS="${GPU_IDS}" \
    SUITES_CSV="${suites_csv}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    HIVA_COEFF_SIDECAR="${SIDECAR_K10_F15}" \
    HIVA_COEFF_SUMMARY="${SUMMARY_K10_F15}" \
    CHUNK_SIZE=15 \
    N_ACTION_STEPS=10 \
    HIVA_DURATION_EXECUTION_MAP="" \
    bash "${SCRIPT_DIR}/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"
  else
    POLICY_PATH="${policy_path}" \
    CHECKPOINT_LABEL="${checkpoint_label}" \
    TIMESTAMP="${TIMESTAMP}_${timestamp_suffix}" \
    GPU_IDS="${GPU_IDS}" \
    SUITES_CSV="${suites_csv}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    HIVA_COEFF_SIDECAR="${SIDECAR_K10_F15}" \
    HIVA_COEFF_SUMMARY="${SUMMARY_K10_F15}" \
    CHUNK_SIZE=15 \
    N_ACTION_STEPS=10 \
    HIVA_DURATION_EXECUTION_MAP="" \
    bash "${SCRIPT_DIR}/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"
  fi
}

run_milestone_group() {
  local train_dir="$1"
  local group_label="$2"
  local group_dir="${REPO_ROOT}/outputs/eval/full_bigtoken_${group_label}_10eps_bs${EVAL_BATCH_SIZE}_${TIMESTAMP}"

  require_dir "${train_dir}/checkpoints"
  mkdir -p "${group_dir}"
  echo "===== Milestone group ${group_label} ====="
  echo "GROUP_DIR=${group_dir}"

  for ckpt in "${MILESTONE_CKPTS[@]}"; do
    local policy_path="${train_dir}/checkpoints/${ckpt}/pretrained_model"
    local ckpt_label="${group_label}_${ckpt}_10eps_bs${EVAL_BATCH_SIZE}"
    local safe_ckpt="${ckpt//[^0-9A-Za-z_-]/_}"
    local suites_csv="libero_object,libero_goal,libero_spatial,libero_10"
    if [[ "${group_label}" == milestone_grip0p3_* && "${ckpt}" == "000094" ]]; then
      local ckpt_dir="${group_dir}/${safe_ckpt}"
      if [[ ! -f "${ckpt_dir}/overlay_eval_summary.json" \
            && -f "${ckpt_dir}/libero_object_taskids__0_1_2_3_4_5_6_7_8_9_/eval_info.json" \
            && -f "${ckpt_dir}/libero_goal_taskids__0_1_2_3_4_5_6_7_8_9_/eval_info.json" \
            && -f "${ckpt_dir}/libero_spatial_taskids__0_1_2_3_4_5_6_7_8_9_/eval_info.json" \
            && ! -f "${ckpt_dir}/libero_10_taskids__0_1_2_3_4_5_6_7_8_9_/eval_info.json" ]]; then
        suites_csv="libero_10"
      fi
    fi
    run_eval "${policy_path}" "${ckpt_label}" "${group_label}_${safe_ckpt}" "${group_dir}/${safe_ckpt}" "${suites_csv}"
  done
}

echo "===== LP-MT grip0p5 + S1 milestone checkpoint evals started at $(date) ====="
echo "TIMESTAMP=${TIMESTAMP}"
echo "GPU_IDS=${GPU_IDS}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"

if [[ "${SKIP_GRIP0P5:-0}" != "1" ]]; then
  run_eval \
    "${GRIP0P5_POLICY}" \
    "job0_lpmt_stage1_xattn_v5_d4_6_10_tr3_rot3_grip0p5_daw1_k10_f15_10eps_bs${EVAL_BATCH_SIZE}" \
    "job0_grip0p5"
else
  echo "===== Skipping grip0p5 standalone eval because SKIP_GRIP0P5=1 ====="
fi

run_milestone_group \
  "${MILESTONE_GRIP0P3}" \
  "milestone_grip0p3_s1_lpmt_stage1_xattn_v5_d4_6_10_tr3_rot3_daw1_b1024_g4"

run_milestone_group \
  "${MILESTONE_GRIP0}" \
  "milestone_grip0_s1_lpmt_stage1_xattn_v5_d4_6_10_tr3_rot3_daw1_b1024_g4"

echo "===== LP-MT grip0p5 + S1 milestone checkpoint evals finished at $(date) ====="
