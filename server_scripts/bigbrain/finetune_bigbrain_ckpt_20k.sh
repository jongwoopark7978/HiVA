#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

LOG_DIR="outputs/train_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/finetune_$(date +%Y%m%d_%H%M%S).log"

# Save both stdout and stderr to terminal and log file
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"

# Use 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MUJOCO_GL=egl

# bigbrain
DATA_ROOT="/nfs/bigbrain/add_disk0/jongwoopark/libero_lerobot_v3_lerobotkeys"
# bigflow
# DATA_ROOT="/nfs/bigflow/add_disk0/jongwoopark/libero_lerobot_v3_lerobotkeys"
# bigquery
#DATA_ROOT="/nfs/bigquery.cs.stonybrook.edu/add_disk0/cristinam/libero/libero_lerobot_v3_lerobotkeys"
DATA_REPO_ID="local/libero_lerobot_v3_lerobotkeys"
TASKS="libero_spatial,libero_object,libero_goal,libero_10"

# Reuse your local downloaded checkpoint. If you prefer Hub loading, replace with:
PRETRAINED="/home/jongwoopark/lerobot/smolvla_libero"


# Keep HF datasets cache on fast local disk
# Create a unique temporary HF datasets cache for this run
export HF_DATASETS_CACHE="/tmp/jongwoo_hf_datasets_cache"
mkdir -p "${HF_DATASETS_CACHE}"


# export HF_DATASETS_CACHE="/nfs/bigflow/add_disk0/jongwoopark/jongwoopark_hf_datasets_cache"
# mkdir -p "${HF_DATASETS_CACHE}"

RUN_NAME="smolvla_libero_from_official_$(date +%Y%m%d_%H%M%S)"

# 20k recommended.
# actual run with bs128
accelerate launch --multi_gpu --num_processes=4 --mixed_precision=bf16 "$(which lerobot-train)" \
  --policy.type=smolvla \
  --policy.pretrained_path="${PRETRAINED}" \
  --policy.expert_width_multiplier=0.5 \
  --policy.num_vlm_layers=0 \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Instruct \
  --policy.push_to_hub=false \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --batch_size=192 \
  --steps=1700 \
  --policy.scheduler_warmup_steps=17 \
  --policy.scheduler_decay_steps=1700 \
  --policy.scheduler_decay_lr=2.5e-6 \
  --policy.device=cuda \
  --policy.num_steps=10 \
  --policy.n_action_steps=1 \
  --dataset.repo_id="${DATA_REPO_ID}" \
  --dataset.root="${DATA_ROOT}" \
  --rename_map='{"observation.images.agentview":"observation.images.image","observation.images.wrist":"observation.images.image2"}' \
  --env.type=libero \
  --env.control_mode=relative \
  --env.task="${TASKS}" \
  --output_dir="outputs/train/${RUN_NAME}" \
  --job_name="${RUN_NAME}" \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval_freq="${EVAL_FREQ:-0}" \



# 20k recommended.
# actual run with bs320
# accelerate launch --multi_gpu --num_processes=4 --mixed_precision=bf16 "$(which lerobot-train)" \
#   --policy.type=smolvla \
#   --policy.pretrained_path="${PRETRAINED}" \
#   --policy.expert_width_multiplier=0.5 \
#   --policy.num_vlm_layers=0 \
#   --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Instruct \
#   --policy.push_to_hub=false \
#   --policy.train_expert_only=true \
#   --policy.freeze_vision_encoder=true \
#   --batch_size=320 \
#   --steps=1000 \
#   --policy.scheduler_warmup_steps=10 \
#   --policy.scheduler_decay_steps=1000 \
#   --policy.scheduler_decay_lr=2.5e-6 \
#   --policy.device=cuda \
#   --policy.num_steps=10 \
#   --policy.n_action_steps=1 \
#   --dataset.repo_id="${DATA_REPO_ID}" \
#   --dataset.root="${DATA_ROOT}" \
#   --rename_map='{"observation.images.agentview":"observation.images.image","observation.images.wrist":"observation.images.image2"}' \
#   --env.type=libero \
#   --env.control_mode=relative \
#   --env.task="${TASKS}" \
#   --output_dir="outputs/train/${RUN_NAME}" \
#   --job_name="${RUN_NAME}" \
#   --eval.batch_size=1 \
#   --eval.n_episodes=1 \
#   --eval_freq="${EVAL_FREQ:-0}" \

# bs	mem  time
# 128	63
# 192	82   
# 256   OOM
# 320	OOM
# 384	OOM

