#!/usr/bin/env bash
set -euo pipefail

# Sequential MT HiVA decoded-action-loss weight sweep on bigflow.
#
# Reference run:
#   smolvla_hiva_coeff_mt_decodedloss_d2_15_residual_ffn_duration_reads_coeffs_ce_mean_k10_f15_bigcornea_b64_s2_rot10_20260508_142308
#
# This sweep keeps the reference MT architecture/loss setup and changes:
#   - GPUs: 4,5,6,7
#   - BATCH_PER_GPU: 128
#   - HIVA_DECODED_ACTION_LOSS_WEIGHT: 0.1, 0.3, 0.5, 2.0
#
# Directory-name tag:
#   "daw" = decoded action loss weight.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

GPU_IDS="${GPU_IDS:-4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-128}"
S="${S:-2}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF

COMMON_ENV=(
  GPU_IDS="${GPU_IDS}"
  NUM_GPUS="${NUM_GPUS}"
  NUM_PROCESSES="${NUM_PROCESSES}"
  BATCH_PER_GPU="${BATCH_PER_GPU}"
  S="${S}"
  WANDB_ENABLE="${WANDB_ENABLE}"
  WANDB_PROJECT="${WANDB_PROJECT}"
  HIVA_DURATION_CLASSES="[2,15]"
  HIVA_DMAX=15
  HIVA_K=10
  HIVA_DEGREE=3
  HIVA_BASIS_MODE=canonical_mt
  HIVA_FIT_HORIZON=15
  HIVA_DURATION_LOSS=ce_mean
  HIVA_SUFFIX_ATTENTION=duration_reads_coeffs
  HIVA_DURATION_HEAD_TYPE=residual_ffn
  HIVA_DURATION_NOISY_LOSS_WEIGHT=1.0
  HIVA_DURATION_CLEAN_LOSS_WEIGHT=0.0
  HIVA_DURATION_NOISY_SIGMA=0.25
  HIVA_DECODED_TR_LOSS_WEIGHT=1.0
  HIVA_DECODED_ROT_LOSS_WEIGHT=10.0
  HIVA_DECODED_GRIP_LOSS_WEIGHT=1.0
  HIVA_DECODED_PREFIX_WEIGHT=1.0
  HIVA_DECODED_POST_DURATION_EXEC_WEIGHT=0.5
  HIVA_DECODED_PREVIEW_WEIGHT=0.1
  HIVA_DECODED_TERMINAL_WEIGHT=0.0
  HIVA_DECODED_LOSS_BETA=0.1
  POLICY_CHUNK_SIZE=15
  POLICY_N_ACTION_STEPS=15
  SIDECAR="/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_canonical_mt.parquet"
  SIDECAR_SUMMARY="/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_canonical_mt.summary.json"
)

run_weight() {
  local weight="$1"
  local tag="$2"

  echo "===== Starting MT decoded-loss sweep daw=${weight}, tag=${tag}, S=${S}, batch=${BATCH_PER_GPU} at $(date) ====="
  env "${COMMON_ENV[@]}" \
    HIVA_DECODED_ACTION_LOSS_WEIGHT="${weight}" \
    RUN_NAME="smolvla_hiva_coeff_mt_decodedloss_d2_15_residual_ffn_duration_reads_coeffs_ce_mean_k10_f15_bigflow_b${BATCH_PER_GPU}_s${S}_daw${tag}_rot10_${RUN_STAMP}" \
    bash "${SCRIPT_DIR}/finetune_bigflow_ckpt_20k_hiva_coeff_mt.sh"
  echo "===== Finished MT decoded-loss sweep daw=${weight}, tag=${tag} at $(date) ====="
}

run_weight 0.1 0p1
run_weight 0.3 0p3
run_weight 0.5 0p5
run_weight 2.0 2
