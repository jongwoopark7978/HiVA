#!/usr/bin/env bash
set -euo pipefail

# Re-run the stage-1 LP-MT HiVA residual xattn finetune with the same
# hyperparameters and initial checkpoint as:
#
#   smolvla_hiva_coeff_lpmt_residual_stage1_xattn_b15_v5_d4_6_10_k10_f15_
#   tr3p0_rot3p0_grip0p0_daw1p0_trb0p1_rotb0p05_gripb0p1_b128_s1_
#   20260512_235751
#
# Differences for this queue:
#   - GPUs 4,5,6,7
#   - batch size 1024 per GPU
#   - residual transformer blocks: 15, then 18
#   - save checkpoints at 0.25x, 0.35x, 0.40x, 0.45x, 0.50x, 0.55x, 0.60x
#     of the original scaled total step count.
#   - stop after the 0.60x checkpoint. For this 4 GPU x 1024 batch setup, the
#     original scaled full run would be 125 optimizer steps, so this queue runs
#     75 steps while keeping warmup/decay based on the 125-step schedule.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

RUN_GROUP_STAMP="${RUN_GROUP_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/stage1_xattn_early_ckpt_b15_b18_bigflow_${RUN_GROUP_STAMP}.queue.log}"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

echo "[$(date)] Queue log: ${QUEUE_LOG}"

COMMON_ENV=(
  GPU_IDS=4,5,6,7
  NUM_GPUS=4
  NUM_PROCESSES=4
  BATCH_PER_GPU=1024
  BATCH_SIZE=1024
  S=1
  STEPS=75
  SCHEDULER_WARMUP_STEPS=4
  SCHEDULER_DECAY_STEPS=125
  SAVE_FREQ=0
  SAVE_STEPS=[32,44,50,57,63,69,75]
  WANDB_ENABLE=true
  INIT_HIVA_BASE=/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/HiVA_train/BestS2_smolvla_hiva_coeff_lpmt_coeffpool_job1_v5_d4_6_10_full_ce_mean_k10_f15_bigcornea_b64_s2_20260510_041334/checkpoints/last/pretrained_model
  SIDECAR=/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet
  SIDECAR_SUMMARY=/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json
  HIVA_DEGREE=3
  HIVA_RESIDUAL_SCALE_TR=3.0
  HIVA_RESIDUAL_SCALE_ROT=3.0
  HIVA_RESIDUAL_SCALE_GRIP=0.0
  HIVA_DECODED_ACTION_LOSS_WEIGHT=1.0
  HIVA_DECODED_TR_LOSS_BETA=0.1
  HIVA_DECODED_ROT_LOSS_BETA=0.05
  HIVA_DECODED_GRIP_LOSS_BETA=0.1
)

run_one() {
  local blocks="$1"
  local run_stamp="${RUN_GROUP_STAMP}_b${blocks}"
  local run_name="smolvla_hiva_coeff_lpmt_residual_stage1_xattn_b${blocks}_v5_d4_6_10_k10_f15_tr3p0_rot3p0_grip0p0_daw1p0_trb0p1_rotb0p05_gripb0p1_b1024_g4_s1_earlyckpt_${RUN_GROUP_STAMP}"

  echo "===== $(date) launching residual stage-1 xattn b${blocks} early-checkpoint run ====="
  echo "RUN_NAME=${run_name}"
  echo "SAVE_STEPS=[32,44,50,57,63,69,75]"

  env "${COMMON_ENV[@]}" \
    RUN_STAMP="${run_stamp}" \
    RUN_NAME="${run_name}" \
    HIVA_RESIDUAL_NUM_BLOCKS="${blocks}" \
    bash "${SCRIPT_DIR}/run_hiva_residual_stage1_xattn_bigflow.sh"

  echo "===== $(date) finished residual stage-1 xattn b${blocks} early-checkpoint run ====="
}

run_one 15
run_one 18

echo "All early-checkpoint b15/b18 stage-1 jobs completed at $(date)"
