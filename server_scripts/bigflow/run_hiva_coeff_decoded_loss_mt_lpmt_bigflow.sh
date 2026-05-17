#!/usr/bin/env bash
set -euo pipefail

# Sequential decoded-action-loss HiVA coefficient jobs on bigflow.
#
# The decoded loss compares decoded raw LIBERO actions from predicted coefficients
# against the target action chunk with prefix/post-duration/preview weighting:
#   prefix=1.0, post-duration executable=0.5, preview=0.1, terminal synthetic tail=0.0.
#
# Job 1: MT / no-extra-preview equivalent
#   duration={2,15}, Dmax=15, fit/preview horizon=15, K=10
#
# Job 2: LP-MT
#   duration={2,15}, Dmax=15, fit/preview horizon=30, K=10
#
# Example:
#   GPU_IDS=4,5,6,7 NUM_GPUS=4 BATCH_PER_GPU=160 S=2 WANDB_ENABLE=true \
#   bash server_scripts/bigflow/run_hiva_coeff_decoded_loss_mt_lpmt_bigflow.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

GPU_IDS="${GPU_IDS:-4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-160}"
S="${S:-2}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"

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
  HIVA_DURATION_LOSS=ce_mean
  HIVA_SUFFIX_ATTENTION=duration_reads_coeffs
  HIVA_DURATION_HEAD_TYPE=residual_ffn
  HIVA_DURATION_NOISY_LOSS_WEIGHT=1.0
  HIVA_DURATION_CLEAN_LOSS_WEIGHT=0.0
  HIVA_DURATION_NOISY_SIGMA=0.25
  HIVA_DECODED_ACTION_LOSS_WEIGHT="${HIVA_DECODED_ACTION_LOSS_WEIGHT:-1.0}"
  HIVA_DECODED_TR_LOSS_WEIGHT="${HIVA_DECODED_TR_LOSS_WEIGHT:-1.0}"
  HIVA_DECODED_ROT_LOSS_WEIGHT="${HIVA_DECODED_ROT_LOSS_WEIGHT:-1.0}"
  HIVA_DECODED_GRIP_LOSS_WEIGHT="${HIVA_DECODED_GRIP_LOSS_WEIGHT:-1.0}"
  HIVA_DECODED_PREFIX_WEIGHT="${HIVA_DECODED_PREFIX_WEIGHT:-1.0}"
  HIVA_DECODED_POST_DURATION_EXEC_WEIGHT="${HIVA_DECODED_POST_DURATION_EXEC_WEIGHT:-0.5}"
  HIVA_DECODED_PREVIEW_WEIGHT="${HIVA_DECODED_PREVIEW_WEIGHT:-0.1}"
  HIVA_DECODED_TERMINAL_WEIGHT="${HIVA_DECODED_TERMINAL_WEIGHT:-0.0}"
  HIVA_DECODED_LOSS_BETA="${HIVA_DECODED_LOSS_BETA:-0.1}"
)

run_mt_d2_15() {
  echo "===== Starting decoded-loss MT HiVA d2_15 k10 f15 S=${S} at $(date) ====="
  env "${COMMON_ENV[@]}" \
    HIVA_BASIS_MODE=canonical_mt \
    HIVA_FIT_HORIZON=15 \
    POLICY_CHUNK_SIZE=15 \
    POLICY_N_ACTION_STEPS=15 \
    SIDECAR="/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_canonical_mt.parquet" \
    SIDECAR_SUMMARY="/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_canonical_mt.summary.json" \
    RUN_NAME="smolvla_hiva_coeff_mt_decodedloss_d2_15_residual_ffn_duration_reads_coeffs_ce_mean_k10_f15_bigflow_b${BATCH_PER_GPU}_s${S}_${RUN_STAMP}" \
    bash "${SCRIPT_DIR}/finetune_bigflow_ckpt_20k_hiva_coeff_mt.sh"
  echo "===== Finished decoded-loss MT HiVA d2_15 k10 f15 at $(date) ====="
}

run_lpmt_d2_15_f30() {
  echo "===== Starting decoded-loss LP-MT HiVA d2_15 k10 f30 S=${S} at $(date) ====="
  env "${COMMON_ENV[@]}" \
    HIVA_BASIS_MODE=canonical_lp_mt \
    HIVA_FIT_HORIZON=30 \
    POLICY_CHUNK_SIZE=30 \
    POLICY_N_ACTION_STEPS=15 \
    SIDECAR="/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_f30_canonical_lp_mt.parquet" \
    SIDECAR_SUMMARY="/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_f30_canonical_lp_mt.summary.json" \
    RUN_NAME="smolvla_hiva_coeff_lpmt_decodedloss_d2_15_residual_ffn_duration_reads_coeffs_ce_mean_k10_f30_bigflow_b${BATCH_PER_GPU}_s${S}_${RUN_STAMP}" \
    bash "${SCRIPT_DIR}/finetune_bigflow_ckpt_20k_hiva_coeff_lp_mt.sh"
  echo "===== Finished decoded-loss LP-MT HiVA d2_15 k10 f30 at $(date) ====="
}

run_mt_d2_15
run_lpmt_d2_15_f30
