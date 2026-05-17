#!/usr/bin/env bash
set -euo pipefail

# Queue one additional S=1 milestone stage-1 xattn run after the current
# resume+S=1 queue finishes. This uses the same hyperparameters as the queued
# S=1 job, except hiva_residual_scale_grip=0.3 instead of 0.0.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

WAIT_TMUX_SESSION="${WAIT_TMUX_SESSION:-hiva_resume_grip0p5_then_s1_20260512_225256}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/stage1_xattn_s1_milestone_grip0p3_after_current_${RUN_STAMP}.queue.log}"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

echo "[$(date)] Queue log: ${QUEUE_LOG}"
echo "[$(date)] Waiting for tmux session to finish: ${WAIT_TMUX_SESSION}"
while tmux has-session -t "${WAIT_TMUX_SESSION}" 2>/dev/null; do
  sleep 300
done

echo "[$(date)] Previous queue is finished. Launching S=1 grip=0.3 milestone run."

export GPU_IDS="${GPU_IDS:-4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-4}"
export NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
export BATCH_PER_GPU="${BATCH_PER_GPU:-1024}"
export BATCH_SIZE="${BATCH_SIZE:-${BATCH_PER_GPU}}"
export S="${S:-1}"
export WANDB_ENABLE="${WANDB_ENABLE:-true}"

export HIVA_RESIDUAL_SCALE_TR="${HIVA_RESIDUAL_SCALE_TR:-3.0}"
export HIVA_RESIDUAL_SCALE_ROT="${HIVA_RESIDUAL_SCALE_ROT:-3.0}"
export HIVA_RESIDUAL_SCALE_GRIP="${HIVA_RESIDUAL_SCALE_GRIP:-0.3}"
export HIVA_DECODED_ACTION_LOSS_WEIGHT="${HIVA_DECODED_ACTION_LOSS_WEIGHT:-1.0}"
export HIVA_DECODED_TR_LOSS_BETA="${HIVA_DECODED_TR_LOSS_BETA:-0.1}"
export HIVA_DECODED_ROT_LOSS_BETA="${HIVA_DECODED_ROT_LOSS_BETA:-0.05}"
export HIVA_DECODED_GRIP_LOSS_BETA="${HIVA_DECODED_GRIP_LOSS_BETA:-0.1}"

GRIP_TAG="${HIVA_RESIDUAL_SCALE_GRIP//./p}"
export RUN_NAME="${RUN_NAME:-smolvla_hiva_coeff_lpmt_stage1_xattn_s1_milestone_v5_d4_6_10_tr3_rot3_grip${GRIP_TAG}_daw1_b${BATCH_PER_GPU}_g${NUM_GPUS}_s${S}_${RUN_STAMP}}"

exec bash "${SCRIPT_DIR}/run_hiva_residual_stage1_xattn_s1_milestone_ckpts_bigflow.sh"
