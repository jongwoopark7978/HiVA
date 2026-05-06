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

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MUJOCO_GL=egl

# bigbrain
# DATA_ROOT="/nfs/bigbrain/add_disk0/jongwoopark/libero_lerobot_v3_lerobotkeys"
# bigflow
# DATA_ROOT="/nfs/bigflow/add_disk0/jongwoopark/libero_lerobot_v3_lerobotkeys"
# bigsplat
DATA_ROOT="/nfs/bigflow.cs.stonybrook.edu/add_disk0/jongwoopark/libero_lerobot_v3_lerobotkeys"

# bigquery
#DATA_ROOT="/nfs/bigquery.cs.stonybrook.edu/add_disk0/cristinam/libero/libero_lerobot_v3_lerobotkeys"
DATA_REPO_ID="local/libero_lerobot_v3_lerobotkeys"
TASKS="libero_spatial,libero_object,libero_goal,libero_10"

# Keep HF datasets cache on fast local disk
# Create a unique temporary HF datasets cache for this run

# Since dataset creation is expensive, a persistent cache is usually better on a large shared filesystem.
export HF_DATASETS_CACHE="/nfs/bigquery.cs.stonybrook.edu/add_disk0/jongwoopark/tmp/jongwoo_hf_datasets_cache"
mkdir -p "${HF_DATASETS_CACHE}"

build_run_id
RUN_NAME="${RUN_NAME:-smolvla_libero_multitask_${RUN_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${RUN_NAME}}"
RESUME="${RESUME:-false}"
guard_train_output_dir "${OUTPUT_DIR}" "${RESUME}"
build_wandb_args
print_wandb_config

# 20k recommended.
# steps, warmup_steps, decay_steps can be scaled down/up proportionally.
# run with bs64 and 4 gpus -> 4 times larger. 20k / 4 = 5k steps

accelerate launch --multi_gpu --num_processes=4 --mixed_precision=bf16 "$(which lerobot-train)" \
  --policy.type=smolvla \
  --policy.expert_width_multiplier=0.5 \
  --policy.num_vlm_layers=0 \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Instruct \
  --policy.push_to_hub=false \
  --policy.load_vlm_weights=true \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --batch_size=64 \
  --steps=5000 \
  --policy.scheduler_warmup_steps=50 \
  --policy.scheduler_decay_steps=5000 \
  --policy.scheduler_decay_lr=2.5e-6 \
  --policy.device=cuda \
  --policy.num_steps=10 \
  --policy.n_action_steps=1 \
  --dataset.repo_id="${DATA_REPO_ID}" \
  --dataset.root="${DATA_ROOT}" \
  --env.type=libero \
  --env.task="${TASKS}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${RUN_NAME}" \
  --resume="${RESUME}" \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval_freq="${EVAL_FREQ:-0}" \
  "${WANDB_ARGS[@]}"


# bs	mem
# 64	42
# 128 50
# 192	56
# 320	80
# 384	94
