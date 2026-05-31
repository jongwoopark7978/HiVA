#!/usr/bin/env bash
set -euo pipefail

# Sequentially finetune the archived P5 and P7 LP-MT HiVA stage-0 setups on
# GPU6,7 with S=0.25, batch/GPU=256, 10k steps, and the requested milestones.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"
OUTER_LOG="${OUTER_LOG:-${LOG_DIR}/run_hiva_coeff_lpmt_stage0_p5_p7_s0p25_b256_gpu6_7_sequential_bigbrain_${RUN_STAMP}.outer.log}"
exec > >(tee -a "${OUTER_LOG}") 2>&1

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigbrain/add_disk0/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
SAVE_STEPS="${SAVE_STEPS:-[6000,6250,7000,7500,8750,10000]}"
TOTAL_STEPS="${TOTAL_STEPS:-10000}"
GPU_IDS="${GPU_IDS:-6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
BATCH_PER_GPU="${BATCH_PER_GPU:-256}"
BATCH_SIZE="${BATCH_SIZE:-${BATCH_PER_GPU}}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"

run_one() {
  local degree="$1"
  local label="$2"
  local run_name="$3"
  local log_name="$4"

  echo "================================================================"
  echo "Starting ${label} at $(date)"
  echo "RUN_NAME=${run_name}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "TOTAL_STEPS=${TOTAL_STEPS}"
  echo "SAVE_STEPS=${SAVE_STEPS}"
  echo "================================================================"

  RUN_STAMP="${RUN_STAMP}" \
  RUN_NAME="${run_name}" \
  QUEUE_LOG="${LOG_DIR}/${log_name}_${RUN_STAMP}.log" \
  GPU_IDS="${GPU_IDS}" \
  NUM_PROCESSES="${NUM_PROCESSES}" \
  MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT}" \
  BATCH_PER_GPU="${BATCH_PER_GPU}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  S="0.25" \
  TOTAL_STEPS="${TOTAL_STEPS}" \
  SAVE_STEPS="${SAVE_STEPS}" \
  SCHEDULER_WARMUP_STEPS="300" \
  SCHEDULER_DECAY_STEPS="${TOTAL_STEPS}" \
  HIVA_DEGREE="${degree}" \
  HIVA_DEGREE_TR="${degree}" \
  HIVA_DEGREE_ROT="${degree}" \
  HIVA_DEGREE_GRIP="${degree}" \
  SIDECAR_ROOT="${SIDECAR_ROOT}" \
  SIDECAR="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p${degree}_f15_canonical_lp_mt.parquet" \
  SIDECAR_SUMMARY="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p${degree}_f15_canonical_lp_mt.summary.json" \
  DATA_ROOT="${DATA_ROOT}" \
  DATA_REPO_ID="local/libero_lerobot_v3_lerobotkeys" \
  WANDB_ENABLE="${WANDB_ENABLE:-true}" \
  WANDB_PROJECT="${WANDB_PROJECT:-lerobot}" \
  WANDB_MODE="${WANDB_MODE:-online}" \
  WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}" \
  WANDB_NOTES="${label}; S=0.25; batch_per_gpu=256; GPUs=${GPU_IDS}; steps=${TOTAL_STEPS}; save_steps=${SAVE_STEPS}" \
  bash "${SCRIPT_DIR}/run_hiva_coeff_lpmt_stage0_p5_s0p5_5000_bigbrain.sh" --total-steps "${TOTAL_STEPS}"

  echo "Finished ${label} at $(date)"
}

P5_RUN_NAME="${P5_RUN_NAME:-smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_p5_f15_bigbrain_b256_g2_s0p25_steps10000_${RUN_STAMP}}"
P7_RUN_NAME="${P7_RUN_NAME:-smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_p7_f15_bigbrain_b256_g2_s0p25_steps10000_${RUN_STAMP}}"

run_one "5" "P5 LP-MT HiVA stage0 finetune" "${P5_RUN_NAME}" "hiva_coeff_lpmt_stage0_p5_s0p25_b256_g2_gpu6_7_steps10000_bigbrain"
run_one "7" "P7 LP-MT HiVA stage0 finetune" "${P7_RUN_NAME}" "hiva_coeff_lpmt_stage0_p7_s0p25_b256_g2_gpu6_7_steps10000_bigbrain"

echo "Both sequential finetunes finished at $(date)"
