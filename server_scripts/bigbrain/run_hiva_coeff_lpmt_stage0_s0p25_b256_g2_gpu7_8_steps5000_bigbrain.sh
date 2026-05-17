#!/usr/bin/env bash
set -euo pipefail

# BigBrain stage-0 LP-MT HiVA coefficient finetune matching the b256 S=0.5
# degree-3 k10_f15 run, but with S=0.25 on GPUs 7,8.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"

SAVE_STEPS="${SAVE_STEPS:-[2500,3000,3125,3500,3750,4375,5000]}"
RUN_NAME="${RUN_NAME:-smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigbrain_b256_g2_s0p25_steps5000_${RUN_STAMP}}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/hiva_coeff_lpmt_stage0_s0p25_b256_g2_gpu7_8_steps5000_bigbrain_${RUN_STAMP}.log}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigbrain/add_disk0/jongwoopark}"
REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk3/jongwoopark/HiVA_train/finetuning_stage0/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigbrain_b256_g2_s0p5_steps5000_20260515_233203/checkpoints/005000/pretrained_model}"

echo "Launching BestS0p5-style stage0 LP-MT finetune on BigBrain with S=0.25"
echo "REFERENCE_CHECKPOINT=${REFERENCE_CHECKPOINT}"
echo "RUN_NAME=${RUN_NAME}"
echo "GPU_IDS=7,8"
echo "BATCH_PER_GPU=256"
echo "EFFECTIVE_BATCH=512"
echo "TOTAL_STEPS=5000"
echo "SAVE_STEPS=${SAVE_STEPS}"
echo "QUEUE_LOG=${QUEUE_LOG}"

if [[ ! -e "${REFERENCE_CHECKPOINT}" ]]; then
  echo "ERROR: reference checkpoint does not exist: ${REFERENCE_CHECKPOINT}" >&2
  exit 2
fi

RUN_STAMP="${RUN_STAMP}" \
RUN_NAME="${RUN_NAME}" \
QUEUE_LOG="${QUEUE_LOG}" \
GPU_IDS="7,8" \
NUM_PROCESSES="2" \
BATCH_PER_GPU="256" \
BATCH_SIZE="256" \
S="0.25" \
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
WANDB_NOTES="${WANDB_NOTES:-BestS0p5-style degree3 k10_f15 BigBrain run with S=0.25; GPUs=7,8; batch_per_gpu=256; effective_batch=512; steps=5000; save_steps=${SAVE_STEPS}}" \
bash "${SCRIPT_DIR}/run_hiva_coeff_lpmt_stage0_p5_s0p5_5000_bigbrain.sh" --total-steps 5000
