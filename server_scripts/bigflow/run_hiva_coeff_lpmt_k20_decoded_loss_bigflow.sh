#!/usr/bin/env bash
set -euo pipefail

# Sequential K=20 LP-MT HiVA coefficient finetuning with decoded action loss on bigflow.
#
# Defaults requested for both jobs:
#   duration={2,15}, Dmax=15, K=20
#   hiva_duration_loss=ce_mean
#   hiva_suffix_attention=duration_reads_coeffs
#   hiva_duration_head_type=residual_ffn
#   BATCH_PER_GPU=160
#   decoded loss weights:
#     action=1.0, tr=1.0, rot=10.0, grip=1.0
#     prefix=1.0, post-duration-exec=0.5, preview=0.1, terminal=0.0
#
# Job 1: preview/fit horizon=30
# Job 2: preview/fit horizon=50
#
# Example:
#   GPU_IDS=4,5,6,7 NUM_GPUS=4 BATCH_PER_GPU=160 S=2 WANDB_ENABLE=true \
#   bash server_scripts/bigflow/run_hiva_coeff_lpmt_k20_decoded_loss_bigflow.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

GPU_IDS="${GPU_IDS:-4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-160}"
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
  HIVA_K=20
  HIVA_DEGREE=3
  HIVA_BASIS_MODE=canonical_lp_mt
  HIVA_DURATION_LOSS=ce_mean
  HIVA_SUFFIX_ATTENTION=duration_reads_coeffs
  HIVA_DURATION_HEAD_TYPE=residual_ffn
  HIVA_DURATION_NOISY_LOSS_WEIGHT=1.0
  HIVA_DURATION_CLEAN_LOSS_WEIGHT=0.0
  HIVA_DURATION_NOISY_SIGMA=0.25
  HIVA_DECODED_ACTION_LOSS_WEIGHT="${HIVA_DECODED_ACTION_LOSS_WEIGHT:-1.0}"
  HIVA_DECODED_TR_LOSS_WEIGHT="${HIVA_DECODED_TR_LOSS_WEIGHT:-1.0}"
  HIVA_DECODED_ROT_LOSS_WEIGHT="${HIVA_DECODED_ROT_LOSS_WEIGHT:-10.0}"
  HIVA_DECODED_GRIP_LOSS_WEIGHT="${HIVA_DECODED_GRIP_LOSS_WEIGHT:-1.0}"
  HIVA_DECODED_PREFIX_WEIGHT="${HIVA_DECODED_PREFIX_WEIGHT:-1.0}"
  HIVA_DECODED_POST_DURATION_EXEC_WEIGHT="${HIVA_DECODED_POST_DURATION_EXEC_WEIGHT:-0.5}"
  HIVA_DECODED_PREVIEW_WEIGHT="${HIVA_DECODED_PREVIEW_WEIGHT:-0.1}"
  HIVA_DECODED_TERMINAL_WEIGHT="${HIVA_DECODED_TERMINAL_WEIGHT:-0.0}"
  HIVA_DECODED_LOSS_BETA="${HIVA_DECODED_LOSS_BETA:-0.1}"
)

run_lpmt() {
  local fit_horizon="$1"

  echo "===== Starting K20 decoded-loss LP-MT HiVA f${fit_horizon} S=${S} at $(date) ====="
  env "${COMMON_ENV[@]}" \
    HIVA_FIT_HORIZON="${fit_horizon}" \
    POLICY_CHUNK_SIZE="${fit_horizon}" \
    POLICY_N_ACTION_STEPS=15 \
    SIDECAR="/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k20_f${fit_horizon}_canonical_lp_mt.parquet" \
    SIDECAR_SUMMARY="/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k20_f${fit_horizon}_canonical_lp_mt.summary.json" \
    RUN_NAME="smolvla_hiva_coeff_lpmt_decodedloss_d2_15_residual_ffn_duration_reads_coeffs_ce_mean_k20_f${fit_horizon}_rotw10_bigflow_b${BATCH_PER_GPU}_s${S}_${RUN_STAMP}" \
    bash "${SCRIPT_DIR}/finetune_bigflow_ckpt_20k_hiva_coeff_lp_mt.sh"
  echo "===== Finished K20 decoded-loss LP-MT HiVA f${fit_horizon} at $(date) ====="
}

run_lpmt 30
run_lpmt 50
