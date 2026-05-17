#!/usr/bin/env bash
set -euo pipefail

# Bigcornea HiVA coefficient SmolVLA finetuning shared by MT and LP-MT wrappers.
#
# This common launcher is intentionally not HP-named. The caller sets MT or LP-MT
# defaults, then this script builds the accelerate command with a Bash arg array.
#
# Example launch on bigcornea:
#   GPU_IDS=4,5,6,7 \
#   NUM_GPUS=4 \
#   BATCH_PER_GPU=160 \
#   S=2 \
#   WANDB_ENABLE=true \
#   WANDB_PROJECT=lerobot \
#   bash server_scripts/bigcornea/finetune_bigcornea_ckpt_20k_hiva_coeff_mt.sh
#
# Smoke-test override:
#   GPU_IDS=3 NUM_GPUS=1 NUM_PROCESSES=1 BATCH_PER_GPU=2 STEPS=1 SAVE_FREQ=1 \
#   WANDB_ENABLE=false \
#   bash server_scripts/bigcornea/finetune_bigcornea_ckpt_20k_hiva_coeff_mt.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/server_scripts/common_wandb.sh"

LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/finetune_bigcornea_hiva_coeff_common_$(date +%Y%m%d_%H%M%S).log"
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
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_N_EPISODES="${EVAL_N_EPISODES:-1}"
EVAL_TASK_IDS="${EVAL_TASK_IDS:-}"
EVAL_MAX_PARALLEL_TASKS="${EVAL_MAX_PARALLEL_TASKS:-1}"

HIVA_TR_LOSS_WEIGHT="${HIVA_TR_LOSS_WEIGHT:-1.0}"
HIVA_ROT_LOSS_WEIGHT="${HIVA_ROT_LOSS_WEIGHT:-1.0}"
HIVA_GRIP_LOSS_WEIGHT="${HIVA_GRIP_LOSS_WEIGHT:-1.0}"
HIVA_DURATION_NOISY_LOSS_WEIGHT="${HIVA_DURATION_NOISY_LOSS_WEIGHT:-1.0}"
HIVA_DURATION_CLEAN_LOSS_WEIGHT="${HIVA_DURATION_CLEAN_LOSS_WEIGHT:-0.0}"
HIVA_DURATION_NOISY_SIGMA="${HIVA_DURATION_NOISY_SIGMA:-0.25}"
HIVA_DURATION_LOSS="${HIVA_DURATION_LOSS:-ce_mean}"
HIVA_DURATION_PREDICTION_TYPE="${HIVA_DURATION_PREDICTION_TYPE:-categorical}"
HIVA_DURATION_READOUT="${HIVA_DURATION_READOUT:-token}"
HIVA_DURATION_FM_LOSS_WEIGHT="${HIVA_DURATION_FM_LOSS_WEIGHT:-1.0}"
HIVA_DURATION_CONT_NORM="${HIVA_DURATION_CONT_NORM:-bounded}"
HIVA_DURATION_MEAN="${HIVA_DURATION_MEAN:-}"
HIVA_DURATION_STD="${HIVA_DURATION_STD:-}"
HIVA_SUFFIX_ATTENTION="${HIVA_SUFFIX_ATTENTION:-duration_reads_coeffs}"
HIVA_BASIS_MODE="${HIVA_BASIS_MODE:-canonical_mt}"
HIVA_DURATION_CLASSES="${HIVA_DURATION_CLASSES:-[6,10,15]}"
HIVA_K="${HIVA_K:-10}"
HIVA_DMAX="${HIVA_DMAX:-15}"
HIVA_FIT_HORIZON="${HIVA_FIT_HORIZON:-${HIVA_DMAX}}"
HIVA_DEGREE="${HIVA_DEGREE:-3}"
HIVA_DURATION_HEAD_TYPE="${HIVA_DURATION_HEAD_TYPE:-residual_ffn}"
HIVA_DURATION_FFN_HIDDEN_MULT="${HIVA_DURATION_FFN_HIDDEN_MULT:-4.0}"
HIVA_DURATION_FFN_ALPHA_INIT="${HIVA_DURATION_FFN_ALPHA_INIT:-0.1}"
HIVA_DECODED_ACTION_LOSS_WEIGHT="${HIVA_DECODED_ACTION_LOSS_WEIGHT:-0.0}"
HIVA_DECODED_TR_LOSS_WEIGHT="${HIVA_DECODED_TR_LOSS_WEIGHT:-1.0}"
HIVA_DECODED_ROT_LOSS_WEIGHT="${HIVA_DECODED_ROT_LOSS_WEIGHT:-1.0}"
HIVA_DECODED_GRIP_LOSS_WEIGHT="${HIVA_DECODED_GRIP_LOSS_WEIGHT:-1.0}"
HIVA_DECODED_PREFIX_WEIGHT="${HIVA_DECODED_PREFIX_WEIGHT:-1.0}"
HIVA_DECODED_POST_DURATION_EXEC_WEIGHT="${HIVA_DECODED_POST_DURATION_EXEC_WEIGHT:-0.5}"
HIVA_DECODED_PREVIEW_WEIGHT="${HIVA_DECODED_PREVIEW_WEIGHT:-0.1}"
HIVA_DECODED_TERMINAL_WEIGHT="${HIVA_DECODED_TERMINAL_WEIGHT:-0.0}"
HIVA_DECODED_LOSS_BETA="${HIVA_DECODED_LOSS_BETA:-0.1}"
HIVA_RESIDUAL_ENABLED="${HIVA_RESIDUAL_ENABLED:-false}"
HIVA_RESIDUAL_HORIZON="${HIVA_RESIDUAL_HORIZON:-${HIVA_FIT_HORIZON}}"
HIVA_RESIDUAL_FFN_HIDDEN_MULT="${HIVA_RESIDUAL_FFN_HIDDEN_MULT:-4.0}"
HIVA_RESIDUAL_TOKEN_TIME_HIDDEN_MULT="${HIVA_RESIDUAL_TOKEN_TIME_HIDDEN_MULT:-2.0}"
HIVA_RESIDUAL_ALPHA_INIT="${HIVA_RESIDUAL_ALPHA_INIT:-0.1}"
HIVA_RESIDUAL_ZERO_INIT="${HIVA_RESIDUAL_ZERO_INIT:-true}"
HIVA_RESIDUAL_SCALE_TR="${HIVA_RESIDUAL_SCALE_TR:-}"
HIVA_RESIDUAL_SCALE_ROT="${HIVA_RESIDUAL_SCALE_ROT:-}"
HIVA_RESIDUAL_SCALE_GRIP="${HIVA_RESIDUAL_SCALE_GRIP:-}"
POLICY_CHUNK_SIZE="${POLICY_CHUNK_SIZE:-${HIVA_FIT_HORIZON}}"
POLICY_N_ACTION_STEPS="${POLICY_N_ACTION_STEPS:-${HIVA_DMAX}}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-/nfs/bigcornea/add_disk2/jongwoopark/libero_lerobot_v3_lerobotkeys}"
DATA_REPO_ID="${DATA_REPO_ID:-local/libero_lerobot_v3_lerobotkeys}"
TASKS="${TASKS:-libero_spatial,libero_object,libero_goal,libero_10}"
SIDECAR="${SIDECAR:-/nfs/bigcornea/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d6_10_15_k${HIVA_K}_canonical_mt.parquet}"
SIDECAR_SUMMARY="${SIDECAR_SUMMARY:-/nfs/bigcornea/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d6_10_15_k${HIVA_K}_canonical_mt.summary.json}"
INIT_SMOLVLA="${INIT_SMOLVLA:-/home/jongwoopark/lerobot/smolvla_libero}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${CONDA_ENV_BIN}/accelerate}"
LEROBOT_TRAIN_BIN="${LEROBOT_TRAIN_BIN:-${CONDA_ENV_BIN}/lerobot-train}"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/jongwoo_hf_datasets_cache}"
mkdir -p "${HF_DATASETS_CACHE}"

build_run_id
HEAD_RUN_SUFFIX=""
if [[ "${HIVA_DURATION_HEAD_TYPE}" != "linear" ]]; then
  HEAD_RUN_SUFFIX="_${HIVA_DURATION_HEAD_TYPE}"
fi
RUN_NAME="${RUN_NAME:-smolvla_hiva_coeff_${HIVA_BASIS_MODE}${HEAD_RUN_SUFFIX}_k${HIVA_K}_bigcornea_b${BATCH_PER_GPU}_s${S}_${RUN_ID}}"
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
SAVE_STEPS="${SAVE_STEPS:-}"

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
echo "SAVE_STEPS=${SAVE_STEPS:-<none>}"
echo "EVAL_FREQ=${EVAL_FREQ}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "EVAL_N_EPISODES=${EVAL_N_EPISODES}"
echo "EVAL_TASK_IDS=${EVAL_TASK_IDS}"
echo "EVAL_MAX_PARALLEL_TASKS=${EVAL_MAX_PARALLEL_TASKS}"
echo "RESUME=${RESUME}"
echo "HIVA_TR_LOSS_WEIGHT=${HIVA_TR_LOSS_WEIGHT}"
echo "HIVA_ROT_LOSS_WEIGHT=${HIVA_ROT_LOSS_WEIGHT}"
echo "HIVA_GRIP_LOSS_WEIGHT=${HIVA_GRIP_LOSS_WEIGHT}"
echo "HIVA_DURATION_NOISY_LOSS_WEIGHT=${HIVA_DURATION_NOISY_LOSS_WEIGHT}"
echo "HIVA_DURATION_CLEAN_LOSS_WEIGHT=${HIVA_DURATION_CLEAN_LOSS_WEIGHT}"
echo "HIVA_DURATION_NOISY_SIGMA=${HIVA_DURATION_NOISY_SIGMA}"
echo "HIVA_DURATION_LOSS=${HIVA_DURATION_LOSS}"
echo "HIVA_DURATION_PREDICTION_TYPE=${HIVA_DURATION_PREDICTION_TYPE}"
echo "HIVA_DURATION_READOUT=${HIVA_DURATION_READOUT}"
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
if [[ "${HIVA_DURATION_PREDICTION_TYPE}" == "categorical" && "${HIVA_DURATION_READOUT}" == "coeff_modality_pool" ]]; then
  echo "HIVA_SUFFIX_LEN=$((3 * HIVA_K))"
else
  echo "HIVA_SUFFIX_LEN=$((1 + 3 * HIVA_K))"
fi
echo "HIVA_DURATION_HEAD_TYPE=${HIVA_DURATION_HEAD_TYPE}"
echo "HIVA_DURATION_FFN_HIDDEN_MULT=${HIVA_DURATION_FFN_HIDDEN_MULT}"
echo "HIVA_DURATION_FFN_ALPHA_INIT=${HIVA_DURATION_FFN_ALPHA_INIT}"
echo "HIVA_DECODED_ACTION_LOSS_WEIGHT=${HIVA_DECODED_ACTION_LOSS_WEIGHT}"
echo "HIVA_DECODED_TR_LOSS_WEIGHT=${HIVA_DECODED_TR_LOSS_WEIGHT}"
echo "HIVA_DECODED_ROT_LOSS_WEIGHT=${HIVA_DECODED_ROT_LOSS_WEIGHT}"
echo "HIVA_DECODED_GRIP_LOSS_WEIGHT=${HIVA_DECODED_GRIP_LOSS_WEIGHT}"
echo "HIVA_DECODED_PREFIX_WEIGHT=${HIVA_DECODED_PREFIX_WEIGHT}"
echo "HIVA_DECODED_POST_DURATION_EXEC_WEIGHT=${HIVA_DECODED_POST_DURATION_EXEC_WEIGHT}"
echo "HIVA_DECODED_PREVIEW_WEIGHT=${HIVA_DECODED_PREVIEW_WEIGHT}"
echo "HIVA_DECODED_TERMINAL_WEIGHT=${HIVA_DECODED_TERMINAL_WEIGHT}"
echo "HIVA_DECODED_LOSS_BETA=${HIVA_DECODED_LOSS_BETA}"
echo "HIVA_RESIDUAL_ENABLED=${HIVA_RESIDUAL_ENABLED}"
echo "HIVA_RESIDUAL_HORIZON=${HIVA_RESIDUAL_HORIZON}"
echo "HIVA_RESIDUAL_FFN_HIDDEN_MULT=${HIVA_RESIDUAL_FFN_HIDDEN_MULT}"
echo "HIVA_RESIDUAL_TOKEN_TIME_HIDDEN_MULT=${HIVA_RESIDUAL_TOKEN_TIME_HIDDEN_MULT}"
echo "HIVA_RESIDUAL_ALPHA_INIT=${HIVA_RESIDUAL_ALPHA_INIT}"
echo "HIVA_RESIDUAL_ZERO_INIT=${HIVA_RESIDUAL_ZERO_INIT}"
echo "HIVA_RESIDUAL_SCALE_TR=${HIVA_RESIDUAL_SCALE_TR}"
echo "HIVA_RESIDUAL_SCALE_ROT=${HIVA_RESIDUAL_SCALE_ROT}"
echo "HIVA_RESIDUAL_SCALE_GRIP=${HIVA_RESIDUAL_SCALE_GRIP}"
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
if [[ -n "${HIVA_RESIDUAL_SCALE_TR}" ]]; then
  HIVA_DURATION_EXTRA_ARGS+=(--policy.hiva_residual_scale_tr="${HIVA_RESIDUAL_SCALE_TR}")
fi
if [[ -n "${HIVA_RESIDUAL_SCALE_ROT}" ]]; then
  HIVA_DURATION_EXTRA_ARGS+=(--policy.hiva_residual_scale_rot="${HIVA_RESIDUAL_SCALE_ROT}")
fi
if [[ -n "${HIVA_RESIDUAL_SCALE_GRIP}" ]]; then
  HIVA_DURATION_EXTRA_ARGS+=(--policy.hiva_residual_scale_grip="${HIVA_RESIDUAL_SCALE_GRIP}")
fi
EVAL_EXTRA_ARGS=()
if [[ -n "${EVAL_TASK_IDS}" ]]; then
  EVAL_EXTRA_ARGS+=(--env.task_ids="${EVAL_TASK_IDS}")
fi
SAVE_EXTRA_ARGS=()
if [[ -n "${SAVE_STEPS}" ]]; then
  SAVE_EXTRA_ARGS+=(--save_steps="${SAVE_STEPS}")
fi

TRAIN_ARGS=(
  --policy.type=smolvla_hiva_coeff
  --policy.push_to_hub=false
  --policy.load_vlm_weights=false
  --policy.init_smolvla_checkpoint_path="${INIT_SMOLVLA}"
  --policy.hiva_coeff_sidecar_path="${SIDECAR}"
  --policy.hiva_coeff_sidecar_summary_path="${SIDECAR_SUMMARY}"
  --policy.hiva_duration_classes="${HIVA_DURATION_CLASSES}"
  --policy.hiva_dmax="${HIVA_DMAX}"
  --policy.hiva_fit_horizon="${HIVA_FIT_HORIZON}"
  --policy.hiva_k="${HIVA_K}"
  --policy.hiva_degree="${HIVA_DEGREE}"
  --policy.hiva_tr_loss_weight="${HIVA_TR_LOSS_WEIGHT}"
  --policy.hiva_rot_loss_weight="${HIVA_ROT_LOSS_WEIGHT}"
  --policy.hiva_grip_loss_weight="${HIVA_GRIP_LOSS_WEIGHT}"
  --policy.hiva_duration_noisy_loss_weight="${HIVA_DURATION_NOISY_LOSS_WEIGHT}"
  --policy.hiva_duration_clean_loss_weight="${HIVA_DURATION_CLEAN_LOSS_WEIGHT}"
  --policy.hiva_duration_noisy_sigma="${HIVA_DURATION_NOISY_SIGMA}"
  --policy.hiva_duration_loss="${HIVA_DURATION_LOSS}"
  --policy.hiva_duration_prediction_type="${HIVA_DURATION_PREDICTION_TYPE}"
  --policy.hiva_duration_readout="${HIVA_DURATION_READOUT}"
  --policy.hiva_duration_fm_loss_weight="${HIVA_DURATION_FM_LOSS_WEIGHT}"
  --policy.hiva_duration_cont_norm="${HIVA_DURATION_CONT_NORM}"
  --policy.hiva_suffix_attention="${HIVA_SUFFIX_ATTENTION}"
  --policy.hiva_basis_mode="${HIVA_BASIS_MODE}"
  --policy.hiva_duration_head_type="${HIVA_DURATION_HEAD_TYPE}"
  --policy.hiva_duration_ffn_hidden_mult="${HIVA_DURATION_FFN_HIDDEN_MULT}"
  --policy.hiva_duration_ffn_alpha_init="${HIVA_DURATION_FFN_ALPHA_INIT}"
  --policy.hiva_decoded_action_loss_weight="${HIVA_DECODED_ACTION_LOSS_WEIGHT}"
  --policy.hiva_decoded_tr_loss_weight="${HIVA_DECODED_TR_LOSS_WEIGHT}"
  --policy.hiva_decoded_rot_loss_weight="${HIVA_DECODED_ROT_LOSS_WEIGHT}"
  --policy.hiva_decoded_grip_loss_weight="${HIVA_DECODED_GRIP_LOSS_WEIGHT}"
  --policy.hiva_decoded_prefix_weight="${HIVA_DECODED_PREFIX_WEIGHT}"
  --policy.hiva_decoded_post_duration_exec_weight="${HIVA_DECODED_POST_DURATION_EXEC_WEIGHT}"
  --policy.hiva_decoded_preview_weight="${HIVA_DECODED_PREVIEW_WEIGHT}"
  --policy.hiva_decoded_terminal_weight="${HIVA_DECODED_TERMINAL_WEIGHT}"
  --policy.hiva_decoded_loss_beta="${HIVA_DECODED_LOSS_BETA}"
  --policy.hiva_residual_enabled="${HIVA_RESIDUAL_ENABLED}"
  --policy.hiva_residual_horizon="${HIVA_RESIDUAL_HORIZON}"
  --policy.hiva_residual_ffn_hidden_mult="${HIVA_RESIDUAL_FFN_HIDDEN_MULT}"
  --policy.hiva_residual_token_time_hidden_mult="${HIVA_RESIDUAL_TOKEN_TIME_HIDDEN_MULT}"
  --policy.hiva_residual_alpha_init="${HIVA_RESIDUAL_ALPHA_INIT}"
  --policy.hiva_residual_zero_init="${HIVA_RESIDUAL_ZERO_INIT}"
  --batch_size="${BATCH_SIZE}"
  --steps="${STEPS}"
  --log_freq=1
  --save_checkpoint=true
  --save_freq="${SAVE_FREQ}"
  --num_workers=0
  --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP_STEPS}"
  --policy.scheduler_decay_steps="${SCHEDULER_DECAY_STEPS}"
  --policy.scheduler_decay_lr="${SCHEDULER_DECAY_LR}"
  --policy.device=cuda
  --policy.num_steps=10
  --policy.chunk_size="${POLICY_CHUNK_SIZE}"
  --policy.n_action_steps="${POLICY_N_ACTION_STEPS}"
  --dataset.repo_id="${DATA_REPO_ID}"
  --dataset.root="${DATA_ROOT}"
  --rename_map='{"observation.images.agentview":"observation.images.image","observation.images.wrist":"observation.images.image2"}'
  --env.type=libero
  --env.control_mode=relative
  --env.task="${TASKS}"
  --env.max_parallel_tasks="${EVAL_MAX_PARALLEL_TASKS}"
  --output_dir="${OUTPUT_DIR}"
  --job_name="${RUN_NAME}"
  --resume="${RESUME}"
  --eval.batch_size="${EVAL_BATCH_SIZE}"
  --eval.n_episodes="${EVAL_N_EPISODES}"
  --eval_freq="${EVAL_FREQ}"
)
TRAIN_ARGS+=("${HIVA_DURATION_EXTRA_ARGS[@]}")
TRAIN_ARGS+=("${EVAL_EXTRA_ARGS[@]}")
TRAIN_ARGS+=("${SAVE_EXTRA_ARGS[@]}")
TRAIN_ARGS+=("${WANDB_ARGS[@]}")

"${ACCELERATE_BIN}" launch \
  --num_processes="${NUM_PROCESSES}" \
  --mixed_precision=bf16 \
  "${LEROBOT_TRAIN_BIN}" \
  "${TRAIN_ARGS[@]}"
