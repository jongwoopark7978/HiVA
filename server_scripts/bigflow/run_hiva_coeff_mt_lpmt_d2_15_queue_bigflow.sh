#!/usr/bin/env bash
set -euo pipefail

# Sequential full finetuning queue on bigflow GPUs 4-7:
#   1. MT, duration={2,15}, S=2
#   2. LP-MT, duration={2,15}, executable Dmax=15, preview horizon=20, K=10, S=2
#   3. MT, duration={2,15}, S=1
#
# Defaults match the recent MT/LP-MT recipe:
#   BATCH_PER_GPU=160
#   hiva_duration_loss=ce_mean
#   hiva_suffix_attention=full
#   hiva_duration_head_type=residual_ffn

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

GPU_IDS="${GPU_IDS:-4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-160}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"

COMMON_ENV=(
  GPU_IDS="${GPU_IDS}"
  NUM_GPUS="${NUM_GPUS}"
  NUM_PROCESSES="${NUM_PROCESSES}"
  BATCH_PER_GPU="${BATCH_PER_GPU}"
  WANDB_ENABLE="${WANDB_ENABLE}"
  WANDB_PROJECT="${WANDB_PROJECT}"
  HIVA_DURATION_CLASSES="[2,15]"
  HIVA_DMAX=15
  HIVA_K=10
  HIVA_DEGREE=3
  HIVA_DURATION_LOSS=ce_mean
  HIVA_SUFFIX_ATTENTION=full
  HIVA_DURATION_HEAD_TYPE=residual_ffn
  HIVA_DURATION_NOISY_LOSS_WEIGHT=1.0
  HIVA_DURATION_CLEAN_LOSS_WEIGHT=0.0
  HIVA_DURATION_NOISY_SIGMA=0.25
)

run_mt() {
  local tag="$1"
  local s="$2"

  echo "===== Starting MT HiVA ${tag} S=${s} at $(date) ====="
  env "${COMMON_ENV[@]}" \
    S="${s}" \
    HIVA_BASIS_MODE=canonical_mt \
    HIVA_FIT_HORIZON=15 \
    POLICY_CHUNK_SIZE=15 \
    POLICY_N_ACTION_STEPS=15 \
    SIDECAR="/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_canonical_mt.parquet" \
    SIDECAR_SUMMARY="/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_canonical_mt.summary.json" \
    RUN_NAME="smolvla_hiva_coeff_mt_d2_15_w1_10_w3_0_residual_ffn_full_ce_mean_k10_bigflow_b${BATCH_PER_GPU}_s${s}_${RUN_STAMP}" \
    bash "${SCRIPT_DIR}/finetune_bigflow_ckpt_20k_hiva_coeff_mt.sh"
  echo "===== Finished MT HiVA ${tag} S=${s} at $(date) ====="
}

run_lpmt() {
  local tag="$1"
  local s="$2"

  echo "===== Starting LP-MT HiVA ${tag} S=${s} at $(date) ====="
  env "${COMMON_ENV[@]}" \
    S="${s}" \
    HIVA_BASIS_MODE=canonical_lp_mt \
    HIVA_FIT_HORIZON=20 \
    POLICY_CHUNK_SIZE=20 \
    POLICY_N_ACTION_STEPS=15 \
    SIDECAR="/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_f20_canonical_lp_mt.parquet" \
    SIDECAR_SUMMARY="/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_f20_canonical_lp_mt.summary.json" \
    RUN_NAME="smolvla_hiva_coeff_lpmt_d2_15_w1_10_w3_0_residual_ffn_full_ce_mean_k10_f20_bigflow_b${BATCH_PER_GPU}_s${s}_${RUN_STAMP}" \
    bash "${SCRIPT_DIR}/finetune_bigflow_ckpt_20k_hiva_coeff_lp_mt.sh"
  echo "===== Finished LP-MT HiVA ${tag} S=${s} at $(date) ====="
}

run_mt "job1_d2_15" 2
run_lpmt "job2_d2_15_f20" 2
run_mt "job3_d2_15" 1
