#!/usr/bin/env bash
set -euo pipefail

# Follow-on b18 stage-1 residual xattn run.
#
# The currently running b15 queue was intentionally shortened to 0.60x
# (75/125 steps). This script waits for that queue to finish, removes the
# placeholder that blocks the obsolete queued b18 launch, then runs b18 through
# the full 1.0x schedule with checkpoints at:
#   0.25x, 0.35x, 0.40x, 0.45x, 0.50x, 0.55x, 0.60x,
#   0.625x, 0.75x, 0.875x, 1.0x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

RUN_STAMP="${RUN_STAMP:-20260513_180325_b18_full}"
WAIT_TMUX_SESSION="${WAIT_TMUX_SESSION:-hiva_stage1_earlyckpt_stop60_b15_b18_gpu4567_20260513_180325}"
RUN_NAME="${RUN_NAME:-smolvla_hiva_coeff_lpmt_residual_stage1_xattn_b18_v5_d4_6_10_k10_f15_tr3p0_rot3p0_grip0p0_daw1p0_trb0p1_rotb0p05_gripb0p1_b1024_g4_s1_earlyckpt_20260513_180325}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_NAME}}"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/stage1_xattn_b18_full_milestones_after_b15_bigflow_${RUN_STAMP}.queue.log}"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

echo "[$(date)] Queue log: ${QUEUE_LOG}"
echo "[$(date)] Waiting for ${WAIT_TMUX_SESSION} to finish before launching b18."
while tmux has-session -t "${WAIT_TMUX_SESSION}" 2>/dev/null; do
  sleep 300
done

if [[ -e "${OUTPUT_DIR}/BLOCKED_OBSOLETE_QUEUE.txt" ]]; then
  echo "[$(date)] Removing obsolete b18 placeholder: ${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}"
elif [[ -e "${OUTPUT_DIR}" ]]; then
  echo "ERROR: output directory already exists and is not the obsolete placeholder: ${OUTPUT_DIR}" >&2
  exit 3
fi

echo "[$(date)] Launching b18 full-milestone run."
env \
  GPU_IDS=4,5,6,7 \
  NUM_GPUS=4 \
  NUM_PROCESSES=4 \
  BATCH_PER_GPU=1024 \
  BATCH_SIZE=1024 \
  S=1 \
  STEPS=125 \
  SCHEDULER_WARMUP_STEPS=4 \
  SCHEDULER_DECAY_STEPS=125 \
  SAVE_FREQ=0 \
  SAVE_STEPS=[32,44,50,57,63,69,75,79,94,110,125] \
  WANDB_ENABLE=true \
  RUN_STAMP="${RUN_STAMP}" \
  RUN_NAME="${RUN_NAME}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  INIT_HIVA_BASE=/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/HiVA_train/BestS2_smolvla_hiva_coeff_lpmt_coeffpool_job1_v5_d4_6_10_full_ce_mean_k10_f15_bigcornea_b64_s2_20260510_041334/checkpoints/last/pretrained_model \
  SIDECAR=/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet \
  SIDECAR_SUMMARY=/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json \
  HIVA_DEGREE=3 \
  HIVA_RESIDUAL_NUM_BLOCKS=18 \
  HIVA_RESIDUAL_SCALE_TR=3.0 \
  HIVA_RESIDUAL_SCALE_ROT=3.0 \
  HIVA_RESIDUAL_SCALE_GRIP=0.0 \
  HIVA_DECODED_ACTION_LOSS_WEIGHT=1.0 \
  HIVA_DECODED_TR_LOSS_BETA=0.1 \
  HIVA_DECODED_ROT_LOSS_BETA=0.05 \
  HIVA_DECODED_GRIP_LOSS_BETA=0.1 \
  bash "${SCRIPT_DIR}/run_hiva_residual_stage1_xattn_bigflow.sh"

echo "[$(date)] Finished b18 full-milestone run."
