#!/usr/bin/env bash
set -euo pipefail

# Sequential MT HiVA coefficient runs for the balanced two-duration sidecars.
#
# Defaults:
#   GPU_IDS=4,5,6,7
#   BATCH_PER_GPU=160
#   S=2
#   hiva_duration_loss=ce_mean
#   hiva_suffix_attention=duration_reads_coeffs
#   hiva_duration_head_type=residual_ffn
#   hiva_duration_clean_loss_weight=0.0
#
# The first run uses duration={6,15}; the second uses duration={2,15}.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

GPU_IDS="${GPU_IDS:-4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-160}"
S="${S:-2}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"
HIVA_DURATION_LOSS="${HIVA_DURATION_LOSS:-ce_mean}"
HIVA_SUFFIX_ATTENTION="${HIVA_SUFFIX_ATTENTION:-duration_reads_coeffs}"
HIVA_DURATION_HEAD_TYPE="${HIVA_DURATION_HEAD_TYPE:-residual_ffn}"
HIVA_DURATION_CLEAN_LOSS_WEIGHT="${HIVA_DURATION_CLEAN_LOSS_WEIGHT:-0.0}"

COMMON_ENV=(
  GPU_IDS="${GPU_IDS}"
  NUM_GPUS="${NUM_GPUS}"
  NUM_PROCESSES="${NUM_PROCESSES}"
  BATCH_PER_GPU="${BATCH_PER_GPU}"
  S="${S}"
  WANDB_ENABLE="${WANDB_ENABLE}"
  WANDB_PROJECT="${WANDB_PROJECT}"
  HIVA_K=10
  HIVA_DMAX=15
  HIVA_BASIS_MODE=canonical_mt
  HIVA_DURATION_LOSS="${HIVA_DURATION_LOSS}"
  HIVA_SUFFIX_ATTENTION="${HIVA_SUFFIX_ATTENTION}"
  HIVA_DURATION_HEAD_TYPE="${HIVA_DURATION_HEAD_TYPE}"
  HIVA_DURATION_NOISY_LOSS_WEIGHT=1.0
  HIVA_DURATION_CLEAN_LOSS_WEIGHT="${HIVA_DURATION_CLEAN_LOSS_WEIGHT}"
  HIVA_DURATION_NOISY_SIGMA=0.25
)

run_one() {
  local tag="$1"
  local duration_classes="$2"
  local sidecar="$3"
  local summary="$4"

  echo "===== Starting MT HiVA ${tag} at $(date) ====="
  env "${COMMON_ENV[@]}" \
    HIVA_DURATION_CLASSES="${duration_classes}" \
    SIDECAR="${sidecar}" \
    SIDECAR_SUMMARY="${summary}" \
    RUN_NAME="smolvla_hiva_coeff_mt_${tag}_${HIVA_DURATION_HEAD_TYPE}_${HIVA_SUFFIX_ATTENTION}_${HIVA_DURATION_LOSS}_k10_bigflow_b${BATCH_PER_GPU}_s${S}_${RUN_STAMP}" \
    bash "${SCRIPT_DIR}/finetune_bigflow_ckpt_20k_hiva_coeff_mt.sh"
  echo "===== Finished MT HiVA ${tag} at $(date) ====="
}

run_one \
  "d6_15_w1_10_w3_0" \
  "[6,15]" \
  "/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d6_15_w1_10_w3_0_k10_canonical_mt.parquet" \
  "/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d6_15_w1_10_w3_0_k10_canonical_mt.summary.json"

run_one \
  "d2_15_w1_10_w3_0" \
  "[2,15]" \
  "/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_canonical_mt.parquet" \
  "/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_canonical_mt.summary.json"
