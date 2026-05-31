#!/usr/bin/env bash
set -euo pipefail

# Smoke test the P5 and P7 LP-MT HiVA stage-0 commands on GPU5 using the same
# batch/GPU=256 and S=0.25 settings, but only a few optimizer steps.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"
OUTER_LOG="${OUTER_LOG:-${LOG_DIR}/smoke_hiva_coeff_lpmt_stage0_p5_p7_s0p25_b256_gpu5_bigbrain_${RUN_STAMP}.outer.log}"
exec > >(tee -a "${OUTER_LOG}") 2>&1

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigbrain/add_disk0/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
SMOKE_STEPS="${SMOKE_STEPS:-5}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-256}"
GPU_IDS="${GPU_IDS:-5}"
SAVE_STEPS="[${SMOKE_STEPS}]"

run_smoke() {
  local degree="$1"
  local label="$2"
  local run_name="smoke_smolvla_hiva_coeff_lpmt_stage0_v5_k10_p${degree}_f15_s0p25_b${SMOKE_BATCH_SIZE}_gpu${GPU_IDS//,/x}_${RUN_STAMP}"

  echo "================================================================"
  echo "Smoke testing ${label} at $(date)"
  echo "RUN_NAME=${run_name}"
  echo "================================================================"

  RUN_STAMP="${RUN_STAMP}" \
  RUN_NAME="${run_name}" \
  QUEUE_LOG="${LOG_DIR}/smoke_hiva_coeff_lpmt_stage0_p${degree}_s0p25_b256_gpu5_bigbrain_${RUN_STAMP}.log" \
  GPU_IDS="${GPU_IDS}" \
  NUM_PROCESSES="1" \
  MAIN_PROCESS_PORT="0" \
  BATCH_PER_GPU="${SMOKE_BATCH_SIZE}" \
  BATCH_SIZE="${SMOKE_BATCH_SIZE}" \
  S="0.25" \
  TOTAL_STEPS="${SMOKE_STEPS}" \
  SAVE_STEPS="${SAVE_STEPS}" \
  SCHEDULER_WARMUP_STEPS="1" \
  SCHEDULER_DECAY_STEPS="${SMOKE_STEPS}" \
  HIVA_DEGREE="${degree}" \
  HIVA_DEGREE_TR="${degree}" \
  HIVA_DEGREE_ROT="${degree}" \
  HIVA_DEGREE_GRIP="${degree}" \
  SIDECAR_ROOT="${SIDECAR_ROOT}" \
  SIDECAR="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p${degree}_f15_canonical_lp_mt.parquet" \
  SIDECAR_SUMMARY="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p${degree}_f15_canonical_lp_mt.summary.json" \
  DATA_ROOT="${DATA_ROOT}" \
  DATA_REPO_ID="local/libero_lerobot_v3_lerobotkeys" \
  WANDB_ENABLE="false" \
  WANDB_MODE="disabled" \
  WANDB_DISABLE_ARTIFACT="true" \
  WANDB_NOTES="${label} smoke; S=0.25; batch_per_gpu=${SMOKE_BATCH_SIZE}; GPU=${GPU_IDS}; steps=${SMOKE_STEPS}" \
  bash "${SCRIPT_DIR}/run_hiva_coeff_lpmt_stage0_p5_s0p5_5000_bigbrain.sh" --total-steps "${SMOKE_STEPS}"

  echo "Smoke passed for ${label} at $(date)"
}

run_smoke "5" "P5 LP-MT HiVA"
run_smoke "7" "P7 LP-MT HiVA"

echo "Both smoke tests passed at $(date)"
