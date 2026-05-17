#!/usr/bin/env bash
set -euo pipefail

# BigBrain stage-0 LP-MT HiVA coefficient finetune matching the archived
# BestS0p5 degree-3 k10_f15 run, with per-GPU batch size 256 on GPUs 6,7.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"

SAVE_STEPS="${SAVE_STEPS:-[2500,2750,3000,3125,3250,3375,3500,3625,3750,3875,4000,4375,5000]}"
RUN_NAME="${RUN_NAME:-smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigbrain_b256_g2_s0p5_steps5000_${RUN_STAMP}}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/hiva_coeff_lpmt_stage0_s0p5_b256_g2_steps5000_bigbrain_${RUN_STAMP}.log}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigbrain/add_disk0/jongwoopark}"

echo "Launching BestS0p5-style stage0 LP-MT finetune on BigBrain"
echo "RUN_NAME=${RUN_NAME}"
echo "GPU_IDS=6,7"
echo "BATCH_PER_GPU=256"
echo "EFFECTIVE_BATCH=512"
echo "TOTAL_STEPS=5000"
echo "SAVE_STEPS=${SAVE_STEPS}"
echo "QUEUE_LOG=${QUEUE_LOG}"

RUN_STAMP="${RUN_STAMP}" \
RUN_NAME="${RUN_NAME}" \
QUEUE_LOG="${QUEUE_LOG}" \
GPU_IDS="6,7" \
NUM_PROCESSES="2" \
BATCH_PER_GPU="256" \
BATCH_SIZE="256" \
S="0.5" \
TOTAL_STEPS="5000" \
SAVE_STEPS="${SAVE_STEPS}" \
HIVA_DEGREE="3" \
SIDECAR_ROOT="${SIDECAR_ROOT}" \
SIDECAR="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet" \
SIDECAR_SUMMARY="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json" \
DATA_ROOT="${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys" \
DATA_REPO_ID="local/libero_lerobot_v3_lerobotkeys" \
WANDB_ENABLE="${WANDB_ENABLE:-true}" \
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}" \
WANDB_MODE="${WANDB_MODE:-online}" \
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}" \
WANDB_NOTES="${WANDB_NOTES:-BestS0p5-style degree3 k10_f15 BigBrain run; GPUs=6,7; batch_per_gpu=256; effective_batch=512; steps=5000}" \
bash "${SCRIPT_DIR}/run_hiva_coeff_lpmt_stage0_p5_s0p5_5000_bigbrain.sh" --total-steps 5000
