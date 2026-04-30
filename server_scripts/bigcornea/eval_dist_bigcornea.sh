#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

POLICY_PATH="${POLICY_PATH:-/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_duration_token_bigflow_smoke_20260429_190734/checkpoints/last/pretrained_model}"

LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="${LOG_DIR}/eval_dist_bigcornea_${TIMESTAMP}.log"

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "Logging to ${MASTER_LOG}"
echo "Using duration policy: ${POLICY_PATH}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

N_EPISODES="${N_EPISODES:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
RENAME_MAP='{"observation.images.image":"observation.images.agentview","observation.images.image2":"observation.images.wrist"}'

run_eval() {
    local GPU_ID=$1
    local TASK_NAME=$2
    local TASK_LOG="${LOG_DIR}/eval_bigcornea_${TASK_NAME}_${TIMESTAMP}.log"
    local RUN_NAME="duration_bigcornea_eval_${TASK_NAME}_${TIMESTAMP}"
    local OUTPUT_DIR="${REPO_ROOT}/outputs/eval/${RUN_NAME}"

    echo "Starting ${TASK_NAME} on GPU ${GPU_ID}"
    echo "Task log: ${TASK_LOG}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    lerobot-eval \
        --policy.path="${POLICY_PATH}" \
        --policy.device=cuda \
        --policy.n_action_steps=8 \
        --policy.use_duration_head=true \
        --policy.duration_train_reuse_prefix_cache=true \
        --env.type=libero \
        --env.task="${TASK_NAME}" \
        --env.control_mode=relative \
        --eval.batch_size="${EVAL_BATCH_SIZE}" \
        --eval.n_episodes="${N_EPISODES}" \
        --rename_map="${RENAME_MAP}" \
        --output_dir="${OUTPUT_DIR}" \
        --job_name="${RUN_NAME}" \
        > "${TASK_LOG}" 2>&1 &
}

run_eval 0 libero_spatial
run_eval 1 libero_object
run_eval 2 libero_goal
run_eval 3 libero_10

echo "All bigcornea duration evaluation jobs launched. Waiting for completion..."
wait
echo "All bigcornea duration evaluations finished."
