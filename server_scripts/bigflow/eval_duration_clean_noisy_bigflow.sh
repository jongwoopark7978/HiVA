#!/usr/bin/env bash
set -euo pipefail

# Evaluation launcher for the clean-plus-noisy duration-query ablation.
#
# Inference intentionally keeps the stable duration decision used by the simple
# baseline: after normal action denoising, the final generated chunk is evaluated
# once with the duration query at t=0. We do not average duration logits over
# near-clean denoising steps in this script.
#
# Example:
#   POLICY_PATH=/home/jongwoopark/lerobot/outputs/train/<run>/checkpoints/last/pretrained_model \
#   bash server_scripts/bigflow/eval_duration_clean_noisy_bigflow.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

: "${POLICY_PATH:?Set POLICY_PATH to the clean-plus-noisy checkpoint pretrained_model directory.}"

LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="${LOG_DIR}/eval_duration_clean_noisy_bigflow_${TIMESTAMP}.log"

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "Logging to ${MASTER_LOG}"
echo "Using clean-plus-noisy duration policy: ${POLICY_PATH}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
N_EPISODES="${N_EPISODES:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
RENAME_MAP='{"observation.images.image":"observation.images.agentview","observation.images.image2":"observation.images.wrist"}'
TASKS=(${TASKS:-libero_spatial libero_object libero_goal libero_10})

run_eval() {
    local gpu_id="$1"
    local task_name="$2"
    local task_log="${LOG_DIR}/eval_clean_noisy_bigflow_${task_name}_${TIMESTAMP}.log"
    local run_name="duration_clean_noisy_bigflow_eval_${task_name}_${TIMESTAMP}"
    local output_dir="${REPO_ROOT}/outputs/eval/${run_name}"

    echo "Starting ${task_name} on GPU ${gpu_id}"
    echo "Task log: ${task_log}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    lerobot-eval \
        --policy.path="${POLICY_PATH}" \
        --policy.device=cuda \
        --policy.n_action_steps=8 \
        --policy.use_duration_head=true \
        --policy.duration_train_reuse_prefix_cache=true \
        --env.type=libero \
        --env.task="${task_name}" \
        --env.control_mode=relative \
        --eval.batch_size="${EVAL_BATCH_SIZE}" \
        --eval.n_episodes="${N_EPISODES}" \
        --rename_map="${RENAME_MAP}" \
        --output_dir="${output_dir}" \
        --job_name="${run_name}" \
        > "${task_log}" 2>&1 &
}

for idx in "${!TASKS[@]}"; do
    gpu_id="${GPU_ARRAY[$((idx % ${#GPU_ARRAY[@]}))]}"
    run_eval "${gpu_id}" "${TASKS[$idx]}"
done

echo "All clean-plus-noisy duration evaluation jobs launched. Waiting for completion..."
wait
echo "All clean-plus-noisy duration evaluations finished."
