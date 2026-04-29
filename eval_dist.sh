#!/usr/bin/env bash
set -euo pipefail

# POLICY_PATH="/home/jongwoopark/lerobot/outputs/train/smolvla_libero_from_official_20260425_004040/checkpoints/last/pretrained_model"
# POLICY_PATH="/home/jongwoopark/lerobot/outputs/train/smolvla_libero_multitask_20260425_165516/checkpoints/last/pretrained_model"
POLICY_PATH="/home/jongwoopark/lerobot/outputs/train/smolvla_libero_from_official_20260425_160656/checkpoints/last/pretrained_model"

LOG_DIR="outputs/eval_logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="${LOG_DIR}/eval_${TIMESTAMP}.log"

# Redirect launcher stdout/stderr to both terminal and master log
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "Logging to ${MASTER_LOG}"
echo "Using policy: ${POLICY_PATH}"

# LIBERO env camera names -> policy expected camera names
RENAME_MAP='{"observation.images.image":"observation.images.agentview","observation.images.image2":"observation.images.wrist"}'

run_eval() {
    local GPU_ID=$1
    local TASK_NAME=$2
    local TASK_LOG="${LOG_DIR}/eval_${TASK_NAME}_${TIMESTAMP}.log"

    echo "Starting ${TASK_NAME} on GPU ${GPU_ID}"
    echo "Task log: ${TASK_LOG}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    lerobot-eval \
        --policy.path="${POLICY_PATH}" \
        --env.type=libero \
        --env.task="${TASK_NAME}" \
        --eval.batch_size=1 \
        --eval.n_episodes=10 \
        --policy.n_action_steps=1 \
        --rename_map="${RENAME_MAP}" \
        > "${TASK_LOG}" 2>&1 &
}

# One task per GPU
run_eval 0 libero_spatial
run_eval 1 libero_object
run_eval 2 libero_goal
run_eval 3 libero_10

echo "All evaluation jobs launched. Waiting for completion..."
wait
echo "All evaluations finished."