#!/usr/bin/env bash
set -euo pipefail

# Canonical max-target (MT) HiVA coefficient SmolVLA finetuning with continuous
# duration flow matching on bigcornea.
#
# Defaults target the bigcornea d2/15 MT coefficient sidecar under:
#   /nfs/bigcornea/add_disk2/jongwoopark
#
# Example smoke:
#   GPU_IDS=0 NUM_GPUS=1 NUM_PROCESSES=1 BATCH_PER_GPU=2 STEPS=2 SAVE_FREQ=1 \
#   HIVA_DURATION_CONT_NORM=bounded HIVA_SUFFIX_ATTENTION=duration_reads_coeffs WANDB_ENABLE=false \
#   bash server_scripts/bigcornea/finetune_bigcornea_ckpt_20k_hiva_coeff_mt_continuous.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/server_scripts/common_wandb.sh"

LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/finetune_bigcornea_hiva_coeff_mt_continuous_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
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
HIVA_DURATION_PREDICTION_TYPE="${HIVA_DURATION_PREDICTION_TYPE:-continuous_fm}"
HIVA_DURATION_FM_LOSS_WEIGHT="${HIVA_DURATION_FM_LOSS_WEIGHT:-1.0}"
HIVA_DURATION_CONT_NORM="${HIVA_DURATION_CONT_NORM:-bounded}"
HIVA_DURATION_MEAN="${HIVA_DURATION_MEAN:-}"
HIVA_DURATION_STD="${HIVA_DURATION_STD:-}"
HIVA_DURATION_NOISY_LOSS_WEIGHT="${HIVA_DURATION_NOISY_LOSS_WEIGHT:-1.0}"
HIVA_DURATION_CLEAN_LOSS_WEIGHT="${HIVA_DURATION_CLEAN_LOSS_WEIGHT:-0.0}"
HIVA_DURATION_NOISY_SIGMA="${HIVA_DURATION_NOISY_SIGMA:-0.25}"
HIVA_DURATION_LOSS="${HIVA_DURATION_LOSS:-ce_mean}"
HIVA_SUFFIX_ATTENTION="${HIVA_SUFFIX_ATTENTION:-duration_reads_coeffs}"
if [[ "${HIVA_SUFFIX_ATTENTION}" == "duration_reads_coefficient" ]]; then
  HIVA_SUFFIX_ATTENTION="duration_reads_coeffs"
fi
HIVA_BASIS_MODE="${HIVA_BASIS_MODE:-canonical_mt}"
HIVA_DURATION_CLASSES="${HIVA_DURATION_CLASSES:-[2,15]}"
HIVA_K="${HIVA_K:-10}"
HIVA_DMAX="${HIVA_DMAX:-15}"
HIVA_FIT_HORIZON="${HIVA_FIT_HORIZON:-${HIVA_DMAX}}"
HIVA_DEGREE="${HIVA_DEGREE:-3}"
POLICY_CHUNK_SIZE="${POLICY_CHUNK_SIZE:-${HIVA_FIT_HORIZON}}"
POLICY_N_ACTION_STEPS="${POLICY_N_ACTION_STEPS:-${HIVA_DMAX}}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-/nfs/bigcornea/add_disk2/jongwoopark/libero_lerobot_v3_lerobotkeys}"
DATA_REPO_ID="${DATA_REPO_ID:-local/libero_lerobot_v3_lerobotkeys}"
TASKS="${TASKS:-libero_spatial,libero_object,libero_goal,libero_10}"
SIDECAR="${SIDECAR:-/nfs/bigcornea/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k${HIVA_K}_canonical_mt.parquet}"
SIDECAR_SUMMARY="${SIDECAR_SUMMARY:-/nfs/bigcornea/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k${HIVA_K}_canonical_mt.summary.json}"
INIT_SMOLVLA="${INIT_SMOLVLA:-/home/jongwoopark/lerobot/smolvla_libero}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${CONDA_ENV_BIN}/accelerate}"
LEROBOT_TRAIN_BIN="${LEROBOT_TRAIN_BIN:-${CONDA_ENV_BIN}/lerobot-train}"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/jongwoo_hf_datasets_cache}"
mkdir -p "${HF_DATASETS_CACHE}"

if [[ ! -f "${SIDECAR}" ]]; then
  echo "ERROR: missing MT coefficient sidecar: ${SIDECAR}" >&2
  exit 2
fi
if [[ ! -f "${SIDECAR_SUMMARY}" ]]; then
  echo "ERROR: missing MT coefficient sidecar summary: ${SIDECAR_SUMMARY}" >&2
  exit 2
fi

build_run_id
DEFAULT_RUN_NAME="smolvla_hiva_coeff_mt_contdur_d2_15_${HIVA_DURATION_CONT_NORM}_${HIVA_SUFFIX_ATTENTION}_k${HIVA_K}_bigcornea_b${BATCH_PER_GPU}_s${S}_${RUN_ID}"
DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME//./p}"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
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
echo "HIVA_DURATION_PREDICTION_TYPE=${HIVA_DURATION_PREDICTION_TYPE}"
echo "HIVA_DURATION_FM_LOSS_WEIGHT=${HIVA_DURATION_FM_LOSS_WEIGHT}"
echo "HIVA_DURATION_CONT_NORM=${HIVA_DURATION_CONT_NORM}"
echo "HIVA_DURATION_MEAN=${HIVA_DURATION_MEAN}"
echo "HIVA_DURATION_STD=${HIVA_DURATION_STD}"
echo "HIVA_SUFFIX_ATTENTION=${HIVA_SUFFIX_ATTENTION}"
echo "HIVA_BASIS_MODE=${HIVA_BASIS_MODE}"
echo "HIVA_DURATION_CLASSES=${HIVA_DURATION_CLASSES}"
echo "HIVA_K=${HIVA_K}"
echo "HIVA_DMAX=${HIVA_DMAX}"
echo "HIVA_FIT_HORIZON=${HIVA_FIT_HORIZON}"
echo "HIVA_SUFFIX_LEN=$((1 + 3 * HIVA_K))"
echo "POLICY_CHUNK_SIZE=${POLICY_CHUNK_SIZE}"
echo "POLICY_N_ACTION_STEPS=${POLICY_N_ACTION_STEPS}"
print_wandb_config

HIVA_DURATION_EXTRA_ARGS=()
if [[ -n "${HIVA_DURATION_MEAN}" ]]; then
  HIVA_DURATION_EXTRA_ARGS+=(--policy.hiva_duration_mean="${HIVA_DURATION_MEAN}")
fi
if [[ -n "${HIVA_DURATION_STD}" ]]; then
  HIVA_DURATION_EXTRA_ARGS+=(--policy.hiva_duration_std="${HIVA_DURATION_STD}")
fi

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
  --policy.hiva_duration_classes="${HIVA_DURATION_CLASSES}" \
  --policy.hiva_dmax="${HIVA_DMAX}" \
  --policy.hiva_fit_horizon="${HIVA_FIT_HORIZON}" \
  --policy.hiva_k="${HIVA_K}" \
  --policy.hiva_degree="${HIVA_DEGREE}" \
  --policy.hiva_tr_loss_weight="${HIVA_TR_LOSS_WEIGHT}" \
  --policy.hiva_rot_loss_weight="${HIVA_ROT_LOSS_WEIGHT}" \
  --policy.hiva_grip_loss_weight="${HIVA_GRIP_LOSS_WEIGHT}" \
  --policy.hiva_duration_noisy_loss_weight="${HIVA_DURATION_NOISY_LOSS_WEIGHT}" \
  --policy.hiva_duration_clean_loss_weight="${HIVA_DURATION_CLEAN_LOSS_WEIGHT}" \
  --policy.hiva_duration_noisy_sigma="${HIVA_DURATION_NOISY_SIGMA}" \
  --policy.hiva_duration_loss="${HIVA_DURATION_LOSS}" \
  --policy.hiva_duration_prediction_type="${HIVA_DURATION_PREDICTION_TYPE}" \
  --policy.hiva_duration_fm_loss_weight="${HIVA_DURATION_FM_LOSS_WEIGHT}" \
  --policy.hiva_duration_cont_norm="${HIVA_DURATION_CONT_NORM}" \
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
  --policy.chunk_size="${POLICY_CHUNK_SIZE}" \
  --policy.n_action_steps="${POLICY_N_ACTION_STEPS}" \
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
  "${HIVA_DURATION_EXTRA_ARGS[@]}" \
  "${WANDB_ARGS[@]}"
