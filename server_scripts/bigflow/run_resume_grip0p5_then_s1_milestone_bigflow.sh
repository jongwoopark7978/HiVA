#!/usr/bin/env bash
set -euo pipefail

# Queue:
# 1. Resume the interrupted stage-1 xattn grip=0.5 run from checkpoint 000042 on GPUs 5-7.
# 2. After it finishes successfully, launch the S=1 milestone-checkpoint run on GPUs 4-7.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${CONDA_ENV_BIN}/accelerate}"
LEROBOT_TRAIN_BIN="${LEROBOT_TRAIN_BIN:-${CONDA_ENV_BIN}/lerobot-train}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/resume_grip0p5_then_s1_milestone_bigflow_${RUN_STAMP}.queue.log}"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

RESUME_CKPT="${RESUME_CKPT:-/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_stage1_xattn_job1_gripsweep_v5_d4_6_10_tr3_rot3_grip0p5_daw1_betas_0p1_0p05_0p1_b1024_g3_s2_20260512_202224/checkpoints/000042}"
RESUME_CONFIG="${RESUME_CONFIG:-${RESUME_CKPT}/pretrained_model/train_config.json}"

if [[ ! -f "${RESUME_CONFIG}" ]]; then
  echo "ERROR: missing resume train config: ${RESUME_CONFIG}" >&2
  exit 2
fi
if [[ ! -d "${RESUME_CKPT}/training_state" ]]; then
  echo "ERROR: missing resume training_state: ${RESUME_CKPT}/training_state" >&2
  exit 2
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
source "/home/jongwoopark/lerobot/server_scripts/common_hf_cache.sh"
setup_hf_datasets_cache
export PATH="${CONDA_ENV_BIN}:${PATH}"
mkdir -p "${HF_DATASETS_CACHE}"

echo "[$(date)] Queue log: ${QUEUE_LOG}"
echo "[$(date)] Step 1/2: resume grip=0.5 from ${RESUME_CKPT} on GPUs 5,6,7"

export CUDA_VISIBLE_DEVICES=5,6,7
export HIVA_TRAIN_RESIDUAL_ONLY=true
export PYTHONPATH="${SCRIPT_DIR}/hiva_residual_stage1_sitecustomize:${REPO_ROOT}/src:${PYTHONPATH:-}"

"${ACCELERATE_BIN}" launch \
  --num_processes=3 \
  --mixed_precision=bf16 \
  "${LEROBOT_TRAIN_BIN}" \
  --config_path="${RESUME_CONFIG}" \
  --resume=true

echo "[$(date)] Step 1/2 complete."
echo "[$(date)] Step 2/2: launch S=1 milestone run on GPUs 4,5,6,7"

export GPU_IDS=4,5,6,7
export NUM_GPUS=4
export NUM_PROCESSES=4
export BATCH_PER_GPU="${BATCH_PER_GPU:-1024}"
export BATCH_SIZE="${BATCH_SIZE:-${BATCH_PER_GPU}}"
export S=1
export WANDB_ENABLE="${WANDB_ENABLE:-true}"
export HIVA_RESIDUAL_SCALE_TR=3.0
export HIVA_RESIDUAL_SCALE_ROT=3.0
export HIVA_RESIDUAL_SCALE_GRIP=0.0
export HIVA_DECODED_ACTION_LOSS_WEIGHT=1.0
export HIVA_DECODED_TR_LOSS_BETA=0.1
export HIVA_DECODED_ROT_LOSS_BETA=0.05
export HIVA_DECODED_GRIP_LOSS_BETA=0.1

exec bash "${SCRIPT_DIR}/run_hiva_residual_stage1_xattn_s1_milestone_ckpts_bigflow.sh"
