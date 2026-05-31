#!/usr/bin/env bash
set -euo pipefail

# Restart from scratch: BigBrain stage-0 LP-MT HiVA coefficient finetune
# matching the P=5 no-residual/no-decoded-loss run, with per-GPU batch size
# 256 on GPUs 8,9.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"

SAVE_STEPS="${SAVE_STEPS:-[1600,1800,1900,3125,3250,3375,3500,3625,3750,3875,4000,4375,5000]}"
RUN_NAME="${RUN_NAME:-smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_p5_f15_bigbrain_b256_g2_s0p5_steps5000_restart_${RUN_STAMP}}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/hiva_coeff_lpmt_stage0_p5_s0p5_b256_g2_gpu8_9_steps5000_restart_bigbrain_${RUN_STAMP}.log}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigbrain/add_disk0/jongwoopark}"

echo "Restarting P5 stage0 LP-MT finetune on BigBrain"
echo "RUN_NAME=${RUN_NAME}"
echo "GPU_IDS=8,9"
echo "BATCH_PER_GPU=256"
echo "EFFECTIVE_BATCH=512"
echo "TOTAL_STEPS=5000"
echo "SAVE_STEPS=${SAVE_STEPS}"
echo "QUEUE_LOG=${QUEUE_LOG}"

RUN_STAMP="${RUN_STAMP}" \
RUN_NAME="${RUN_NAME}" \
QUEUE_LOG="${QUEUE_LOG}" \
GPU_IDS="8,9" \
NUM_PROCESSES="2" \
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29589}" \
BATCH_PER_GPU="256" \
BATCH_SIZE="256" \
S="0.5" \
TOTAL_STEPS="5000" \
SAVE_STEPS="${SAVE_STEPS}" \
HIVA_DEGREE="5" \
SIDECAR_ROOT="${SIDECAR_ROOT}" \
SIDECAR="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p5_f15_canonical_lp_mt.parquet" \
SIDECAR_SUMMARY="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p5_f15_canonical_lp_mt.summary.json" \
DATA_ROOT="${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys" \
DATA_REPO_ID="local/libero_lerobot_v3_lerobotkeys" \
WANDB_ENABLE="${WANDB_ENABLE:-true}" \
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}" \
WANDB_MODE="${WANDB_MODE:-online}" \
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}" \
WANDB_NOTES="${WANDB_NOTES:-Restart from scratch. P5 stage0 LP-MT BigBrain run; GPUs=8,9; batch_per_gpu=256; effective_batch=512; steps=5000; save_steps=${SAVE_STEPS}}" \
bash "${SCRIPT_DIR}/run_hiva_coeff_lpmt_stage0_p5_s0p5_5000_bigbrain.sh" --total-steps 5000
