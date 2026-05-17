#!/usr/bin/env bash
set -euo pipefail

# Follow-up S=1 LP-MT xattn checkpoint sweeps.
# This script is intended to run after hiva_eval_s1_milestones_resume2_20260513_111255.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark}"

SIDECAR_K10_F15="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet"
SUMMARY_K10_F15="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json"

TRAIN_P5_GRIP0P3="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_stage1_xattn_s1_milestone_v5_d4_6_10_p5_tr3_rot3_grip0p3_daw1_b1024_g4_s1_20260513_105200"
TRAIN_B15_GRIP0="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_stage1_xattn_b15_v5_d4_6_10_k10_f15_tr3p0_rot3p0_grip0p0_daw1p0_trb0p1_rotb0p05_gripb0p1_b1024_g4_s1_earlyckpt_20260513_180325"

CKPTS_P5_GRIP0P3=(000063 000079 000094 000110 000125 last)
CKPTS_B15_GRIP0=(000032 000044 000050 000057 000063 000069 000075 last)

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
  local train_dir="$1"
  local group_label="$2"
  local group_dir="$3"
  local ckpt="$4"
  local policy_path="${train_dir}/checkpoints/${ckpt}/pretrained_model"
  local base_output_dir="${group_dir}/${ckpt}"
  local checkpoint_label="${group_label}_${ckpt}_10eps_bs${EVAL_BATCH_SIZE}"

  if [[ -f "${base_output_dir}/overlay_eval_summary.json" ]]; then
    echo "===== Skipping completed ${checkpoint_label}: ${base_output_dir} ====="
    return
  fi

  require_dir "${policy_path}"
  require_file "${SIDECAR_K10_F15}"
  require_file "${SUMMARY_K10_F15}"
  mkdir -p "${group_dir}"

  echo "===== Running ${checkpoint_label} at $(date) ====="
  echo "POLICY_PATH=${policy_path}"
  echo "BASE_OUTPUT_DIR=${base_output_dir}"

  BASE_OUTPUT_DIR="${base_output_dir}" \
  POLICY_PATH="${policy_path}" \
  CHECKPOINT_LABEL="${checkpoint_label}" \
  TIMESTAMP="${TIMESTAMP}_${group_label}_${ckpt}" \
  GPU_IDS="${GPU_IDS}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  HIVA_COEFF_SIDECAR="${SIDECAR_K10_F15}" \
  HIVA_COEFF_SUMMARY="${SUMMARY_K10_F15}" \
  CHUNK_SIZE=15 \
  N_ACTION_STEPS=10 \
  HIVA_DURATION_EXECUTION_MAP="" \
  bash "${SCRIPT_DIR}/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"
}

run_group() {
  local train_dir="$1"
  local group_label="$2"
  shift 2
  local ckpts=("$@")
  local group_dir="${REPO_ROOT}/outputs/eval/full_bigtoken_${group_label}_10eps_bs${EVAL_BATCH_SIZE}_${TIMESTAMP}"

  require_dir "${train_dir}/checkpoints"
  echo "===== Group ${group_label} ====="
  echo "TRAIN_DIR=${train_dir}"
  echo "GROUP_DIR=${group_dir}"
  for ckpt in "${ckpts[@]}"; do
    run_eval "${train_dir}" "${group_label}" "${group_dir}" "${ckpt}"
  done
}

echo "===== Follow-up S1 milestone evals started at $(date) ====="
echo "TIMESTAMP=${TIMESTAMP}"
echo "GPU_IDS=${GPU_IDS}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"

run_group \
  "${TRAIN_P5_GRIP0P3}" \
  "milestone_p5_grip0p3_s1_lpmt_stage1_xattn_v5_d4_6_10_tr3_rot3_daw1_b1024_g4" \
  "${CKPTS_P5_GRIP0P3[@]}"

echo "===== Skipping canceled queued eval for ${TRAIN_B15_GRIP0} ====="

echo "===== Follow-up S1 milestone evals finished at $(date) ====="
