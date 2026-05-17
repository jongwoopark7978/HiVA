#!/usr/bin/env bash
set -euo pipefail

# Sequential S=1 stage-1 xattn residual runs for the LP-MT HiVA P5 and P7
# coefficient bases. Hyperparameters match the last S=1 milestone job, with
# hiva_residual_scale_grip=0.3, tr/rot scales 3.0, DAW=1.0, and betas
# tr=0.1, rot=0.05, grip=0.1.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

RUN_GROUP_STAMP="${RUN_GROUP_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/stage1_xattn_s1_milestone_p5_p7_bigflow_${RUN_GROUP_STAMP}.queue.log}"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

echo "[$(date)] Queue log: ${QUEUE_LOG}"

COMMON_ENV=(
  GPU_IDS=4,5,6,7
  NUM_GPUS=4
  NUM_PROCESSES=4
  BATCH_PER_GPU=1024
  BATCH_SIZE=1024
  S=1
  WANDB_ENABLE=true
  HIVA_RESIDUAL_SCALE_TR=3.0
  HIVA_RESIDUAL_SCALE_ROT=3.0
  HIVA_RESIDUAL_SCALE_GRIP=0.3
  HIVA_DECODED_ACTION_LOSS_WEIGHT=1.0
  HIVA_DECODED_TR_LOSS_BETA=0.1
  HIVA_DECODED_ROT_LOSS_BETA=0.05
  HIVA_DECODED_GRIP_LOSS_BETA=0.1
)

run_one() {
  local degree="$1"
  local init_base="$2"
  local sidecar="$3"
  local summary="$4"

  local run_stamp="${RUN_GROUP_STAMP}_p${degree}"
  local run_name="smolvla_hiva_coeff_lpmt_stage1_xattn_s1_milestone_v5_d4_6_10_p${degree}_tr3_rot3_grip0p3_daw1_b1024_g4_s1_${RUN_GROUP_STAMP}"

  echo "===== $(date) launching P${degree} stage-1 xattn S=1 ====="
  echo "INIT_HIVA_BASE=${init_base}"
  echo "SIDECAR=${sidecar}"
  echo "SIDECAR_SUMMARY=${summary}"
  echo "RUN_NAME=${run_name}"

  env "${COMMON_ENV[@]}" \
    RUN_STAMP="${run_stamp}" \
    RUN_NAME="${run_name}" \
    INIT_HIVA_BASE="${init_base}" \
    SIDECAR="${sidecar}" \
    SIDECAR_SUMMARY="${summary}" \
    HIVA_DEGREE="${degree}" \
    bash "${SCRIPT_DIR}/run_hiva_residual_stage1_xattn_s1_milestone_ckpts_bigflow.sh"

  echo "===== $(date) finished P${degree} stage-1 xattn S=1 ====="
}

run_one 5 \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_v5_d4_6_10_coeffpool_full_ce_mean_k10_p5_f15_bigflow_b128_s2_nores_nodl_20260511_184643/checkpoints/last/pretrained_model" \
  "/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p5_f15_canonical_lp_mt.parquet" \
  "/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p5_f15_canonical_lp_mt.summary.json"

run_one 7 \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_v5_d4_6_10_coeffpool_full_ce_mean_k10_p7_f15_bigflow_b128_s2_nores_nodl_20260511_184643/checkpoints/last/pretrained_model" \
  "/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p7_f15_canonical_lp_mt.parquet" \
  "/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p7_f15_canonical_lp_mt.summary.json"

echo "All P5/P7 S=1 milestone jobs completed at $(date)"
