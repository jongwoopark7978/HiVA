#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/server_scripts/common_wandb.sh"

LOG_DIR="outputs/train_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/finetune_$(date +%Y%m%d_%H%M%S).log"

# Save both stdout and stderr to terminal and log file
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"

# Use 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# use 2 gpus
# export CUDA_VISIBLE_DEVICES=2,3
# bigbrain
DATA_ROOT="/nfs/bigbrain/add_disk0/jongwoopark/libero_lerobot_v3_lerobotkeys"
# bigflow
# DATA_ROOT="/nfs/bigflow/add_disk0/jongwoopark/libero_lerobot_v3_lerobotkeys"
# bigquery
#DATA_ROOT="/nfs/bigquery.cs.stonybrook.edu/add_disk0/cristinam/libero/libero_lerobot_v3_lerobotkeys"
DATA_REPO_ID="local/libero_lerobot_v3_lerobotkeys"
TASKS="libero_spatial,libero_object,libero_goal,libero_10"

# Keep HF datasets cache on fast local disk
# Create a unique temporary HF datasets cache for this run
export HF_DATASETS_CACHE="$(mktemp -d /tmp/jongwoo_hf_datasets_cache_XXXXXX)"
echo "Using HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"

# Clean it up automatically when the script exits
cleanup() {
    if [ -n "${HF_DATASETS_CACHE:-}" ] && [ -d "${HF_DATASETS_CACHE}" ]; then
        echo "Removing temporary cache: ${HF_DATASETS_CACHE}"
        rm -rf "${HF_DATASETS_CACHE}"
    fi
}
trap cleanup EXIT

# export HF_DATASETS_CACHE="/nfs/bigflow/add_disk0/jongwoopark/jongwoopark_hf_datasets_cache"
# mkdir -p "${HF_DATASETS_CACHE}"

RUN_NAME="smolvla_libero_multitask_$(date +%Y%m%d_%H%M%S)"
build_wandb_args
print_wandb_config

#regular
#accelerate launch --multi_gpu --num_processes=4 $(which lerobot-train) \
#for speed up
#accelerate launch --multi_gpu --num_processes=4 --mixed_precision=bf16 "$(which lerobot-train)" \

# pairs
# steps=5000, warmup_steps=50, decay_steps=5000, decay_lr=2.5e-6 with bs320 * 4 gpus
# --eval_freq=1000

# 100k recommended.
# # actual run with bs320
# accelerate launch --multi_gpu --num_processes=4 --mixed_precision=bf16 "$(which lerobot-train)" \
#   --policy.type=smolvla \
#   --policy.push_to_hub=false \
#   --policy.load_vlm_weights=true \
#   --policy.train_expert_only=true \
#   --policy.freeze_vision_encoder=true \
#   --batch_size=320 \
#   --steps=5000 \
#   --policy.scheduler_warmup_steps=50 \
#   --policy.scheduler_decay_steps=5000 \
#   --policy.scheduler_decay_lr=2.5e-6 \
#   --policy.device=cuda \
#   --policy.num_steps=10 \
#   --policy.n_action_steps=1 \
#   --dataset.repo_id="${DATA_REPO_ID}" \
#   --dataset.root="${DATA_ROOT}" \
#   --env.type=libero \
#   --env.task="${TASKS}" \
#   --output_dir=outputs/train/${RUN_NAME} \
#   --job_name="${RUN_NAME}" \
#   --eval.batch_size=1 \
#   --eval.n_episodes=1 \
#   --eval_freq="${EVAL_FREQ:-0}"



# 20k recommended.
# actual run with bs320
accelerate launch --multi_gpu --num_processes=4 --mixed_precision=bf16 "$(which lerobot-train)" \
  --policy.type=smolvla \
  --policy.push_to_hub=false \
  --policy.load_vlm_weights=true \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --batch_size=320 \
  --steps=1000 \
  --policy.scheduler_warmup_steps=10 \
  --policy.scheduler_decay_steps=1000 \
  --policy.scheduler_decay_lr=2.5e-6 \
  --policy.device=cuda \
  --policy.num_steps=10 \
  --policy.n_action_steps=1 \
  --dataset.repo_id="${DATA_REPO_ID}" \
  --dataset.root="${DATA_ROOT}" \
  --env.type=libero \
  --env.task="${TASKS}" \
  --output_dir=outputs/train/${RUN_NAME} \
  --job_name="${RUN_NAME}" \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval_freq="${EVAL_FREQ:-0}" \
  "${WANDB_ARGS[@]}"


# 20k recommended.
# steps, warmup_steps, decay_steps can be scaled down/up proportionally.
# run with bs128 and 2 gpus -> 4 times larger. 20k / 4 = 5k steps

# accelerate launch --multi_gpu --num_processes=2 --mixed_precision=bf16 "$(which lerobot-train)" \
#   --policy.type=smolvla \
#   --policy.push_to_hub=false \
#   --policy.load_vlm_weights=true \
#   --policy.train_expert_only=true \
#   --policy.freeze_vision_encoder=true \
#   --batch_size=128 \
#   --steps=5000 \
#   --policy.scheduler_warmup_steps=50 \
#   --policy.scheduler_decay_steps=5000 \
#   --policy.scheduler_decay_lr=2.5e-6 \
#   --policy.device=cuda \
#   --policy.num_steps=10 \
#   --policy.n_action_steps=1 \
#   --dataset.repo_id="${DATA_REPO_ID}" \
#   --dataset.root="${DATA_ROOT}" \
#   --env.type=libero \
#   --env.task="${TASKS}" \
#   --output_dir=outputs/train/${RUN_NAME} \
#   --job_name=${RUN_NAME} \
#   --eval.batch_size=1 \
#   --eval.n_episodes=1 \
#   --eval_freq="${EVAL_FREQ:-0}"





# test pairs
# steps=100, warmup_steps=10, decay_steps=100, decay_lr=2.5e-6 with bs320 * 4 gpus
# --eval_freq=5

# test run with bs320
# accelerate launch --multi_gpu --num_processes=4 --mixed_precision=bf16 "$(which lerobot-train)" \
#   --policy.type=smolvla \
#   --policy.push_to_hub=false \
#   --policy.load_vlm_weights=true \
#   --policy.train_expert_only=true \
#   --policy.freeze_vision_encoder=true \
#   --batch_size=320 \
#   --steps=100 \
#   --policy.scheduler_warmup_steps=10 \
#   --policy.scheduler_decay_steps=100 \
#   --policy.scheduler_decay_lr=2.5e-6 \
#   --policy.device=cuda \
#   --policy.num_steps=10 \
#   --policy.n_action_steps=1 \
#   --dataset.repo_id="${DATA_REPO_ID}" \
#   --dataset.root="${DATA_ROOT}" \
#   --env.type=libero \
#   --env.task="${TASKS}" \
#   --output_dir=outputs/train/${RUN_NAME} \
#   --job_name=${RUN_NAME} \
#   --eval.batch_size=1 \
#   --eval.n_episodes=1 \
#   --eval_freq=5


# test run with eval with bs192
# accelerate launch --multi_gpu --num_processes=2 --mixed_precision=bf16 "$(which lerobot-train)" \
#   --policy.type=smolvla \
#   --policy.push_to_hub=false \
#   --policy.load_vlm_weights=true \
#   --policy.train_expert_only=true \
#   --policy.freeze_vision_encoder=true \
#   --batch_size=192 \
#   --steps=100 \
#   --policy.scheduler_warmup_steps=10 \
#   --policy.scheduler_decay_steps=100 \
#   --policy.scheduler_decay_lr=2.5e-6 \
#   --policy.device=cuda \
#   --policy.num_steps=10 \
#   --policy.n_action_steps=1 \
#   --dataset.repo_id="${DATA_REPO_ID}" \
#   --dataset.root="${DATA_ROOT}" \
#   --env.type=libero \
#   --env.task="${TASKS}" \
#   --output_dir=outputs/train/${RUN_NAME} \
#   --job_name=${RUN_NAME} \
#   --eval.batch_size=1 \
#   --eval.n_episodes=1 \
#   --eval_freq=5


# test run with eval with bs128
# accelerate launch --multi_gpu --num_processes=2 --mixed_precision=bf16 "$(which lerobot-train)" \
#   --policy.type=smolvla \
#   --policy.push_to_hub=false \
#   --policy.load_vlm_weights=true \
#   --policy.train_expert_only=true \
#   --policy.freeze_vision_encoder=true \
#   --batch_size=128 \
#   --steps=100 \
#   --policy.scheduler_warmup_steps=10 \
#   --policy.scheduler_decay_steps=100 \
#   --policy.scheduler_decay_lr=2.5e-6 \
#   --policy.device=cuda \
#   --policy.num_steps=10 \
#   --policy.n_action_steps=1 \
#   --dataset.repo_id="${DATA_REPO_ID}" \
#   --dataset.root="${DATA_ROOT}" \
#   --env.type=libero \
#   --env.task="${TASKS}" \
#   --output_dir=outputs/train/${RUN_NAME} \
#   --job_name=${RUN_NAME} \
#   --eval.batch_size=1 \
#   --eval.n_episodes=1 \
#   --eval_freq=5



# bs	mem
# 64	34
# 192	56
# 320	80
# 384	94
