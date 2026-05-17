#!/usr/bin/env bash
set -euo pipefail

# Test cleaner-suffix HiVA coefficient SmolVLA finetuning on bigcornea 24GB GPUs.
#
# Default launch tests batch size 64 on one GPU:
#   WANDB_ENABLE=true \
#   WANDB_PROJECT=lerobot \
#   bash server_scripts/bigcornea/finetune_bigcornea_ckpt_20k_hiva_coeff_cleaner_suffix_b64.sh
#
# Batch-size sweep examples:
#   BATCH_PER_GPU=48 bash server_scripts/bigcornea/finetune_bigcornea_ckpt_20k_hiva_coeff_cleaner_suffix_b64.sh
#   BATCH_PER_GPU=32 bash server_scripts/bigcornea/finetune_bigcornea_ckpt_20k_hiva_coeff_cleaner_suffix_b64.sh
#
# Multi-GPU override:
#   GPU_IDS=0,1,2,3 NUM_GPUS=4 BATCH_PER_GPU=64 \
#   bash server_scripts/bigcornea/finetune_bigcornea_ckpt_20k_hiva_coeff_cleaner_suffix_b64.sh
#   BATCH_PER_GPU  peak GPU memory
#   64             19 GB
#   80             23 GB
#   96             OOM



SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/server_scripts/common_wandb.sh"

LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/finetune_bigcornea_hiva_coeff_cleaner_suffix_b64_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

GPU_IDS="${GPU_IDS:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-64}"
S="${S:-2}"
RESUME="${RESUME:-false}"

BASE_NUM_GPUS="${BASE_NUM_GPUS:-1}"
BASE_BATCH_PER_GPU="${BASE_BATCH_PER_GPU:-64}"
BASE_STEPS="${BASE_STEPS:-20000}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-2.5e-6}"
EVAL_FREQ="${EVAL_FREQ:-0}"

HIVA_TR_LOSS_WEIGHT="${HIVA_TR_LOSS_WEIGHT:-1.0}"
HIVA_ROT_LOSS_WEIGHT="${HIVA_ROT_LOSS_WEIGHT:-1.0}"
HIVA_GRIP_LOSS_WEIGHT="${HIVA_GRIP_LOSS_WEIGHT:-1.0}"
HIVA_DURATION_NOISY_LOSS_WEIGHT="${HIVA_DURATION_NOISY_LOSS_WEIGHT:-1.0}"
HIVA_DURATION_CLEAN_LOSS_WEIGHT="${HIVA_DURATION_CLEAN_LOSS_WEIGHT:-0.0}"
HIVA_DURATION_NOISY_SIGMA="${HIVA_DURATION_NOISY_SIGMA:-0.25}"
HIVA_DURATION_LOSS="${HIVA_DURATION_LOSS:-ce_mean}"
HIVA_SUFFIX_ATTENTION="${HIVA_SUFFIX_ATTENTION:-duration_prefix}"
HIVA_BASIS_MODE="${HIVA_BASIS_MODE:-duration_specific}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-/nfs/bigcornea/add_disk2/jongwoopark/libero_lerobot_v3_lerobotkeys}"
DATA_REPO_ID="${DATA_REPO_ID:-local/libero_lerobot_v3_lerobotkeys}"
TASKS="${TASKS:-libero_spatial,libero_object,libero_goal,libero_10}"
SIDECAR="${SIDECAR:-/nfs/bigcornea/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d6_10_15_k6_all_episodes.parquet}"
SIDECAR_SUMMARY="${SIDECAR_SUMMARY:-/nfs/bigcornea/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d6_10_15_k6_all_episodes.summary.json}"
INIT_SMOLVLA="${INIT_SMOLVLA:-/home/jongwoopark/lerobot/smolvla_libero}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${CONDA_ENV_BIN}/accelerate}"
LEROBOT_TRAIN_BIN="${LEROBOT_TRAIN_BIN:-${CONDA_ENV_BIN}/lerobot-train}"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/jongwoo_hf_datasets_cache}"
mkdir -p "${HF_DATASETS_CACHE}"

build_run_id
RUN_NAME="${RUN_NAME:-smolvla_hiva_coeff_cleaner_suffix_bigcornea_b${BATCH_PER_GPU}_s${S}_${RUN_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_NAME}}"
guard_train_output_dir "${OUTPUT_DIR}" "${RESUME}"
build_wandb_args

GLOBAL_BATCH_SIZE=$((NUM_GPUS * BATCH_PER_GPU))
BASE_GLOBAL_BATCH_SIZE=$((BASE_NUM_GPUS * BASE_BATCH_PER_GPU))

STEPS="${STEPS:-$(python - <<PY
import math
steps = math.ceil(int("${BASE_STEPS}") * int("${BASE_GLOBAL_BATCH_SIZE}") / int("${GLOBAL_BATCH_SIZE}") / float("${S}"))
print(max(1, steps))
PY
)}"
BATCH_SIZE="${BATCH_SIZE:-${BATCH_PER_GPU}}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-$(python - <<PY
import math
print(max(1, math.ceil(int("${STEPS}") * float("${WARMUP_RATIO}"))))
PY
)}"
SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-${STEPS}}"
SAVE_FREQ="${SAVE_FREQ:-$(python - <<PY
steps = int("${STEPS}")
print(max(1, (steps + 1) // 2))
PY
)}"

echo "Logging to ${LOG_FILE}"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "BATCH_PER_GPU=${BATCH_PER_GPU}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "S=${S}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "SIDECAR=${SIDECAR}"
echo "SIDECAR_SUMMARY=${SIDECAR_SUMMARY}"
echo "INIT_SMOLVLA=${INIT_SMOLVLA}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "STEPS=${STEPS}"
echo "SAVE_FREQ=${SAVE_FREQ}"
echo "EVAL_FREQ=${EVAL_FREQ}"
echo "RESUME=${RESUME}"
echo "HIVA_TR_LOSS_WEIGHT=${HIVA_TR_LOSS_WEIGHT}"
echo "HIVA_ROT_LOSS_WEIGHT=${HIVA_ROT_LOSS_WEIGHT}"
echo "HIVA_GRIP_LOSS_WEIGHT=${HIVA_GRIP_LOSS_WEIGHT}"
echo "HIVA_DURATION_NOISY_LOSS_WEIGHT=${HIVA_DURATION_NOISY_LOSS_WEIGHT}"
echo "HIVA_DURATION_CLEAN_LOSS_WEIGHT=${HIVA_DURATION_CLEAN_LOSS_WEIGHT}"
echo "HIVA_DURATION_NOISY_SIGMA=${HIVA_DURATION_NOISY_SIGMA}"
echo "HIVA_DURATION_LOSS=${HIVA_DURATION_LOSS}"
echo "HIVA_SUFFIX_ATTENTION=${HIVA_SUFFIX_ATTENTION}"
echo "HIVA_BASIS_MODE=${HIVA_BASIS_MODE}"
print_wandb_config

"${ACCELERATE_BIN}" launch \
  --num_processes="${NUM_PROCESSES}" \
  --mixed_precision=bf16 \
  "${LEROBOT_TRAIN_BIN}" \
  --policy.type=smolvla_hiva_coeff \
  --policy.push_to_hub=false \
  --policy.load_vlm_weights=false \
  --policy.init_smolvla_checkpoint_path="${INIT_SMOLVLA}" \
  --policy.hiva_coeff_sidecar_path="${SIDECAR}" \
  --policy.hiva_coeff_sidecar_summary_path="${SIDECAR_SUMMARY}" \
  --policy.hiva_duration_classes='[6,10,15]' \
  --policy.hiva_dmax=15 \
  --policy.hiva_k=6 \
  --policy.hiva_degree=3 \
  --policy.hiva_tr_loss_weight="${HIVA_TR_LOSS_WEIGHT}" \
  --policy.hiva_rot_loss_weight="${HIVA_ROT_LOSS_WEIGHT}" \
  --policy.hiva_grip_loss_weight="${HIVA_GRIP_LOSS_WEIGHT}" \
  --policy.hiva_duration_noisy_loss_weight="${HIVA_DURATION_NOISY_LOSS_WEIGHT}" \
  --policy.hiva_duration_clean_loss_weight="${HIVA_DURATION_CLEAN_LOSS_WEIGHT}" \
  --policy.hiva_duration_noisy_sigma="${HIVA_DURATION_NOISY_SIGMA}" \
  --policy.hiva_duration_loss="${HIVA_DURATION_LOSS}" \
  --policy.hiva_suffix_attention="${HIVA_SUFFIX_ATTENTION}" \
  --policy.hiva_basis_mode="${HIVA_BASIS_MODE}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --log_freq=1 \
  --save_checkpoint=true \
  --save_freq="${SAVE_FREQ}" \
  --num_workers=0 \
  --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP_STEPS}" \
  --policy.scheduler_decay_steps="${SCHEDULER_DECAY_STEPS}" \
  --policy.scheduler_decay_lr="${SCHEDULER_DECAY_LR}" \
  --policy.device=cuda \
  --policy.num_steps=10 \
  --policy.chunk_size=15 \
  --policy.n_action_steps=15 \
  --dataset.repo_id="${DATA_REPO_ID}" \
  --dataset.root="${DATA_ROOT}" \
  --rename_map='{"observation.images.agentview":"observation.images.image","observation.images.wrist":"observation.images.image2"}' \
  --env.type=libero \
  --env.control_mode=relative \
  --env.task="${TASKS}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${RUN_NAME}" \
  --resume="${RESUME}" \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval_freq="${EVAL_FREQ}" \
  "${WANDB_ARGS[@]}"
