#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

POLICY_PATH="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_duration_token_bigflow_smoke_20260429_190734/checkpoints/last/pretrained_model"

LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="${LOG_DIR}/eval_duration_smoke_${TIMESTAMP}.log"

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "Logging to ${MASTER_LOG}"
echo "Using duration policy: ${POLICY_PATH}"

GPU_ID="${GPU_ID:-5}"
TASK_NAME="${TASK_NAME:-libero_spatial}"
TASK_IDS="${TASK_IDS:-[0]}"
N_EPISODES="${N_EPISODES:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
RUN_NAME="duration_smoke_${TASK_NAME}_taskids_${TASK_IDS//[^0-9A-Za-z_-]/_}_${TIMESTAMP}"
OUTPUT_DIR="${REPO_ROOT}/outputs/eval/${RUN_NAME}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# LIBERO env camera names -> policy expected camera names.
RENAME_MAP='{"observation.images.image":"observation.images.agentview","observation.images.image2":"observation.images.wrist"}'

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "TASK_NAME=${TASK_NAME}"
echo "TASK_IDS=${TASK_IDS}"
echo "N_EPISODES=${N_EPISODES}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

lerobot-eval \
  --policy.path="${POLICY_PATH}" \
  --policy.device=cuda \
  --policy.n_action_steps=8 \
  --policy.use_duration_head=true \
  --policy.duration_train_reuse_prefix_cache=true \
  --env.type=libero \
  --env.task="${TASK_NAME}" \
  --env.task_ids="${TASK_IDS}" \
  --env.control_mode=relative \
  --eval.batch_size="${EVAL_BATCH_SIZE}" \
  --eval.n_episodes="${N_EPISODES}" \
  --rename_map="${RENAME_MAP}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${RUN_NAME}"

echo "Duration smoke eval finished."
