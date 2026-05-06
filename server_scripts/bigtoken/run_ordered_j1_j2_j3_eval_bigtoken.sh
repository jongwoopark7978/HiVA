#!/usr/bin/env bash
set -euo pipefail

# Ordered bigtoken evaluation queue requested on 2026-05-05:
#   J1: normalized-duration-loss coefficient HiVA, one S4 checkpoint, 10eps.
#   J2: mean-weighted coefficient HiVA, resume S=2,4,8, 10eps.
#   J3: original SmolVLA bigcornea S4, resume missing libero_10, 50eps.
#
# The wrapper is intentionally sequential. Each child script may run LIBERO
# suites in parallel internally, but J2 does not start until J1 exits cleanly,
# and J3 does not start until J2 exits cleanly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
TASK_IDS_ALL="${TASK_IDS_ALL:-[0,1,2,3,4,5,6,7,8,9]}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-1}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-1}"

echo "===== Ordered evaluation queue started at $(date) ====="
echo "TIMESTAMP=${TIMESTAMP}"
echo "GPU_IDS=${GPU_IDS}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "TASK_IDS_ALL=${TASK_IDS_ALL}"

echo "===== J1: normalized-duration-loss coeff HiVA S4 partial eval ====="
env \
  TIMESTAMP="${TIMESTAMP}_j1_normdur" \
  GPU_IDS="${GPU_IDS}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  TASK_IDS_ALL="${TASK_IDS_ALL}" \
  N_EPISODES=10 \
  MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS}" \
  MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED}" \
  EXPECTED_EPISODE_COUNT=400 \
  EXPECTED_VIDEO_COUNT=40 \
  POLICY_PATH="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_cleaner_suffix_bigcornea_sigma0p25_w0p5_b80_s4_durw_sweep_b80_s4_20260504_175101/checkpoints/last/pretrained_model" \
  CHECKPOINT_LABEL="smolvla_hiva_coeff_cleaner_suffix_bigcornea_sigma0p25_w0p5_b80_s4_durw_sweep_b80_s4_20260504_175101_10eps_bs4" \
  bash "${SCRIPT_DIR}/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"

echo "===== J2: mean-weighted coeff HiVA S2/S4/S8 partial eval ====="
env \
  TIMESTAMP="${TIMESTAMP}_j2_mean" \
  GPU_IDS="${GPU_IDS}" \
  SUITE_GPU_IDS="${SUITE_GPU_IDS:-0,2,1,3}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  TASK_IDS_ALL="${TASK_IDS_ALL}" \
  N_EPISODES=10 \
  MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS}" \
  MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED}" \
  EXPECTED_EPISODE_COUNT=400 \
  EXPECTED_VIDEO_COUNT=40 \
  START_INDEX=1 \
  END_INDEX=3 \
  WAIT_FOR_GPU_MEMORY="${WAIT_FOR_GPU_MEMORY:-true}" \
  FREE_MEM_MIN_MIB="${FREE_MEM_MIN_MIB:-10000}" \
  MONITOR_GPU_MEMORY="${MONITOR_GPU_MEMORY:-true}" \
  MEMORY_MONITOR_INTERVAL_SECONDS="${MEMORY_MONITOR_INTERVAL_SECONDS:-2}" \
  bash "${SCRIPT_DIR}/eval_hiva_coeff_bigflow_mean_s1_s2_s4_s8_10eps_bigtoken.sh"

echo "===== J3: original SmolVLA bigcornea S4 missing libero_10 full eval ====="
env \
  TIMESTAMP="20260503_104832" \
  GPU_IDS="${J3_GPU_IDS:-0}" \
  MUJOCO_EGL_DEVICE_ID="${J3_MUJOCO_EGL_DEVICE_ID:-0}" \
  TASKS="libero_10" \
  N_EPISODES=50 \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED}" \
  POLICY_PATH="/home/jongwoopark/lerobot/outputs/train/smolvla_original_bigcornea_s4_20260502_102226/checkpoints/last/pretrained_model" \
  CHECKPOINT_LABEL="smolvla_original_bigcornea_s4_20260502_102226_50eps" \
  USE_DURATION_HEAD=false \
  N_ACTION_STEPS=1 \
  OBJECT_TASK_IDS="${TASK_IDS_ALL}" \
  GOAL_TASK_IDS="${TASK_IDS_ALL}" \
  SPATIAL_TASK_IDS="${TASK_IDS_ALL}" \
  LIBERO10_TASK_IDS="${TASK_IDS_ALL}" \
  BASE_OUTPUT_DIR="/home/jongwoopark/lerobot/outputs/eval/full_bigtoken_smolvla_original_bigcornea_s4_20260502_102226_50eps_20260503_104832" \
  WRITE_OVERLAY_SUMMARY=true \
  bash "${SCRIPT_DIR}/eval_duration_overlay_bigtoken.sh"

echo "===== Ordered evaluation queue finished at $(date) ====="
