#!/usr/bin/env bash
set -euo pipefail

# Smoke evaluation for a saved smolvla_hiva_coeff checkpoint on bigflow.
#
# This mirrors eval_duration_smoke_bigflow.sh, but loads the coefficient HiVA
# policy and its {6,10,15}, K=6 coefficient sidecar metadata. Evaluation is a
# single-process workload; exposing GPUs 4-7 makes GPU 4 cuda:0 for the policy
# while leaving the same idle GPU set available to the process.
#
# Example:
#   bash server_scripts/bigflow/eval_hiva_coeff_smoke_bigflow.sh
#
# Override example:
#   POLICY_PATH=/home/jongwoopark/lerobot/outputs/train/<run>/checkpoints/last/pretrained_model \
#   GPU_IDS=4,5,6,7 TASK_NAME=libero_spatial TASK_IDS='[0]' N_EPISODES=1 \
#   bash server_scripts/bigflow/eval_hiva_coeff_smoke_bigflow.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

POLICY_PATH="${POLICY_PATH:-/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_bigflow_smoke_20260504_010038/checkpoints/000002/pretrained_model}"

LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="${LOG_DIR}/eval_hiva_coeff_smoke_bigflow_${TIMESTAMP}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

GPU_IDS="${GPU_IDS:-${GPU_ID:-4,5,6,7}}"
TASK_NAME="${TASK_NAME:-libero_spatial}"
TASK_IDS="${TASK_IDS:-[0]}"
N_EPISODES="${N_EPISODES:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
RUN_NAME="${RUN_NAME:-hiva_coeff_smoke_${TASK_NAME}_taskids_${TASK_IDS//[^0-9A-Za-z_-]/_}_${TIMESTAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/${RUN_NAME}}"

SIDECAR="${SIDECAR:-/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d6_10_15_k6_all_episodes.parquet}"
SIDECAR_SUMMARY="${SIDECAR_SUMMARY:-/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d6_10_15_k6_all_episodes.summary.json}"
HIVA_BASIS_MODE="${HIVA_BASIS_MODE:-duration_specific}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

RENAME_MAP='{"observation.images.image":"observation.images.agentview","observation.images.image2":"observation.images.wrist"}'

echo "Logging to ${MASTER_LOG}"
echo "POLICY_PATH=${POLICY_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "TASK_NAME=${TASK_NAME}"
echo "TASK_IDS=${TASK_IDS}"
echo "SIDECAR=${SIDECAR}"
echo "SIDECAR_SUMMARY=${SIDECAR_SUMMARY}"
echo "HIVA_BASIS_MODE=${HIVA_BASIS_MODE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

lerobot-eval \
  --policy.path="${POLICY_PATH}" \
  --policy.device=cuda \
  --policy.n_action_steps=15 \
  --policy.hiva_coeff_sidecar_path="${SIDECAR}" \
  --policy.hiva_coeff_sidecar_summary_path="${SIDECAR_SUMMARY}" \
  --policy.hiva_basis_mode="${HIVA_BASIS_MODE}" \
  --env.type=libero \
  --env.task="${TASK_NAME}" \
  --env.task_ids="${TASK_IDS}" \
  --env.control_mode=relative \
  --eval.batch_size="${EVAL_BATCH_SIZE}" \
  --eval.n_episodes="${N_EPISODES}" \
  --rename_map="${RENAME_MAP}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${RUN_NAME}"

echo "HiVA coefficient smoke eval finished."
