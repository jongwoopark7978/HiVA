#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_DIR="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigcornea_b64_s0p5_20260514_182207"
MODEL_EVAL_DIR="/home/jongwoopark/lerobot/outputs/eval/lpmt_stage0_v5_all_ckpts_partial_bigbrain_bs10_20260515_000849"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
TIMESTAMP="${TIMESTAMP:-20260515_$(date +%H%M%S)_ckpt1500_1875_gpu4_7}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"

run_ckpt() {
  local ckpt="$1"
  local policy_path="${TRAIN_DIR}/checkpoints/${ckpt}/pretrained_model"
  local base_output_dir="${MODEL_EVAL_DIR}/ckpt_${ckpt}"
  local label="lpmt_stage0_s0p5_ckpt_${ckpt}_10eps_bs${EVAL_BATCH_SIZE}"

  if [[ -f "${base_output_dir}/overlay_eval_summary.json" ]]; then
    echo "===== $(date) skipping completed ${label}: ${base_output_dir} ====="
    return
  fi
  if [[ ! -d "${policy_path}" ]]; then
    echo "Missing policy path: ${policy_path}" >&2
    exit 1
  fi

  echo "===== $(date) evaluating ${label} ====="
  echo "POLICY_PATH=${policy_path}"
  echo "BASE_OUTPUT_DIR=${base_output_dir}"

  POLICY_PATH="${policy_path}" \
  CHECKPOINT_LABEL="${label}" \
  TIMESTAMP="${TIMESTAMP}_${ckpt}" \
  GPU_IDS="${GPU_IDS}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  N_EPISODES=10 \
  TASK_IDS_ALL="[0,1,2,3,4,5,6,7,8,9]" \
  MAX_PARALLEL_TASKS=1 \
  MAX_EPISODES_RENDERED=1 \
  EXPECTED_EPISODE_COUNT=400 \
  EXPECTED_VIDEO_COUNT=40 \
  DATA_ROOT="${DATA_ROOT}" \
  HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR}" \
  HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY}" \
  CHUNK_SIZE=15 \
  N_ACTION_STEPS=10 \
  NUM_STEPS=10 \
  HIVA_DURATION_EXECUTION_MAP="" \
  STAGED_LIBERO10_AFTER_SHORT=1 \
  SPLIT_LIBERO10_ACROSS_GPUS=0 \
  BASE_OUTPUT_DIR="${base_output_dir}" \
  bash "${SCRIPT_DIR}/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"
}

echo "===== LPMT stage0 s0p5 ckpt 001500/001875 eval started at $(date) ====="
echo "GPU_IDS=${GPU_IDS}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "MODEL_EVAL_DIR=${MODEL_EVAL_DIR}"

run_ckpt "001500"
run_ckpt "001875"

echo "===== LPMT stage0 s0p5 ckpt 001500/001875 eval finished at $(date) ====="
