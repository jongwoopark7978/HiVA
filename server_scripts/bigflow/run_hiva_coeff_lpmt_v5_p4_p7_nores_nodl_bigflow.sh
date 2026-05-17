#!/usr/bin/env bash
set -euo pipefail

# Standalone bigflow LP-MT HiVA v5 P=4..7 sweep without decoded action loss
# and without residual action prediction.
#
# This script intentionally launches `lerobot-train` directly instead of calling
# another finetuning wrapper. It also snapshots itself before running so VSCode
# edits to this file cannot corrupt a long-running shell process.
#
# Default launch:
#   GPU_IDS=2,3,4,5 NUM_GPUS=4 BATCH_PER_GPU=128 S=2 \
#   bash server_scripts/bigflow/run_hiva_coeff_lpmt_v5_p4_p7_nores_nodl_bigflow.sh

if [[ "${HIVA_SCRIPT_SNAPSHOT:-0}" != "1" ]]; then
  SNAPSHOT_ROOT="${HIVA_SCRIPT_SNAPSHOT_ROOT:-/tmp/jongwoopark_hiva_script_snapshots}"
  mkdir -p "${SNAPSHOT_ROOT}"
  SNAPSHOT_PATH="${SNAPSHOT_ROOT}/$(basename "$0").$USER.$(date +%Y%m%d_%H%M%S_%N).$$.sh"
  cp "$0" "${SNAPSHOT_PATH}"
  chmod +x "${SNAPSHOT_PATH}"
  export HIVA_SCRIPT_SNAPSHOT=1
  export HIVA_ORIGINAL_SCRIPT_PATH="$(readlink -f "$0")"
  exec bash "${SNAPSHOT_PATH}" "$@"
fi

SCRIPT_PATH="${HIVA_ORIGINAL_SCRIPT_PATH:-$(readlink -f "$0")}"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"
QUEUE_LOG="${LOG_DIR}/hiva_lpmt_v5_p4_p7_nores_nodl_bigflow_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

DEGREES="${DEGREES:-4 5 6 7}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigflow/add_disk0/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
DATA_REPO_ID="${DATA_REPO_ID:-local/libero_lerobot_v3_lerobotkeys}"
TASKS="${TASKS:-libero_spatial,libero_object,libero_goal,libero_10}"
INIT_SMOLVLA="${INIT_SMOLVLA:-/home/jongwoopark/lerobot/smolvla_libero}"

GPU_IDS="${GPU_IDS:-2,3,4,5}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-128}"
BATCH_SIZE="${BATCH_SIZE:-${BATCH_PER_GPU}}"
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

HIVA_DURATION_CLASSES="${HIVA_DURATION_CLASSES:-[4,6,10]}"
HIVA_DMAX="${HIVA_DMAX:-10}"
HIVA_FIT_HORIZON="${HIVA_FIT_HORIZON:-15}"
HIVA_K="${HIVA_K:-10}"
HIVA_BASIS_MODE="${HIVA_BASIS_MODE:-canonical_lp_mt}"
HIVA_DURATION_LOSS="${HIVA_DURATION_LOSS:-ce_mean}"
HIVA_SUFFIX_ATTENTION="${HIVA_SUFFIX_ATTENTION:-full}"
HIVA_DURATION_READOUT="${HIVA_DURATION_READOUT:-coeff_modality_pool}"
HIVA_DURATION_HEAD_TYPE="${HIVA_DURATION_HEAD_TYPE:-residual_ffn}"
HIVA_DURATION_NOISY_LOSS_WEIGHT="${HIVA_DURATION_NOISY_LOSS_WEIGHT:-1.0}"
HIVA_DURATION_CLEAN_LOSS_WEIGHT="${HIVA_DURATION_CLEAN_LOSS_WEIGHT:-0.0}"
HIVA_DURATION_NOISY_SIGMA="${HIVA_DURATION_NOISY_SIGMA:-0.25}"
HIVA_DURATION_PREDICTION_TYPE="${HIVA_DURATION_PREDICTION_TYPE:-categorical}"
HIVA_DURATION_FM_LOSS_WEIGHT="${HIVA_DURATION_FM_LOSS_WEIGHT:-1.0}"
HIVA_DURATION_CONT_NORM="${HIVA_DURATION_CONT_NORM:-bounded}"
HIVA_TR_LOSS_WEIGHT="${HIVA_TR_LOSS_WEIGHT:-1.0}"
HIVA_ROT_LOSS_WEIGHT="${HIVA_ROT_LOSS_WEIGHT:-1.0}"
HIVA_GRIP_LOSS_WEIGHT="${HIVA_GRIP_LOSS_WEIGHT:-1.0}"

# Explicitly disabled for this sweep.
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

POLICY_CHUNK_SIZE="${POLICY_CHUNK_SIZE:-${HIVA_FIT_HORIZON}}"
POLICY_N_ACTION_STEPS="${POLICY_N_ACTION_STEPS:-${HIVA_DMAX}}"

WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}"
WANDB_NOTES="${WANDB_NOTES:-LP-MT v5 P4-P7 no decoded loss, no residual action}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${CONDA_ENV_BIN}/accelerate}"
LEROBOT_TRAIN_BIN="${LEROBOT_TRAIN_BIN:-${CONDA_ENV_BIN}/lerobot-train}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/jongwoo_hf_datasets_cache}"
mkdir -p "${HF_DATASETS_CACHE}"

GLOBAL_BATCH_SIZE=$((NUM_GPUS * BATCH_PER_GPU))
BASE_GLOBAL_BATCH_SIZE=$((BASE_NUM_GPUS * BASE_BATCH_PER_GPU))
STEPS="${STEPS:-$(python - <<PY
import math
steps = math.ceil(int("${BASE_STEPS}") * int("${BASE_GLOBAL_BATCH_SIZE}") / int("${GLOBAL_BATCH_SIZE}") / float("${S}"))
print(max(1, steps))
PY
)}"
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

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
read -r -a DEGREE_ARRAY <<< "${DEGREES}"

WANDB_ARGS=(
  --wandb.enable="${WANDB_ENABLE}"
  --wandb.project="${WANDB_PROJECT}"
  --wandb.disable_artifact="${WANDB_DISABLE_ARTIFACT}"
)
if [[ -n "${WANDB_ENTITY}" ]]; then
  WANDB_ARGS+=(--wandb.entity="${WANDB_ENTITY}")
fi
if [[ -n "${WANDB_MODE}" ]]; then
  WANDB_ARGS+=(--wandb.mode="${WANDB_MODE}")
fi
if [[ -n "${WANDB_NOTES}" ]]; then
  WANDB_ARGS+=(--wandb.notes="${WANDB_NOTES}")
fi

EVAL_EXTRA_ARGS=()
if [[ -n "${EVAL_TASK_IDS}" ]]; then
  EVAL_EXTRA_ARGS+=(--env.task_ids="${EVAL_TASK_IDS}")
fi

guard_output_dir() {
  local output_dir="$1"
  local resume="$2"
  if [[ "${resume}" == "true" ]]; then
    if [[ ! -e "${output_dir}/checkpoints/last" ]]; then
      echo "ERROR: RESUME=true but no last checkpoint exists under: ${output_dir}" >&2
      exit 2
    fi
    return
  fi
  if [[ -e "${output_dir}" ]]; then
    echo "ERROR: OUTPUT_DIR already exists and RESUME is not true: ${output_dir}" >&2
    exit 2
  fi
}

echo "LP-MT v5 no-residual/no-decoded-loss P sweep started at $(date)"
echo "Running from snapshot: ${BASH_SOURCE[0]}"
echo "Original script: ${SCRIPT_PATH}"
echo "Queue log: ${QUEUE_LOG}"
echo "DEGREES=${DEGREE_ARRAY[*]}"
echo "GPU_IDS=${GPU_IDS}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "BATCH_PER_GPU=${BATCH_PER_GPU}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "S=${S}"
echo "STEPS=${STEPS}"
echo "SAVE_FREQ=${SAVE_FREQ}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "WANDB_ENABLE=${WANDB_ENABLE}"
echo "HIVA_DURATION_READOUT=${HIVA_DURATION_READOUT}"
echo "HIVA_SUFFIX_ATTENTION=${HIVA_SUFFIX_ATTENTION}"
echo "HIVA_DECODED_ACTION_LOSS_WEIGHT=${HIVA_DECODED_ACTION_LOSS_WEIGHT}"
echo "HIVA_RESIDUAL_ENABLED=${HIVA_RESIDUAL_ENABLED}"

for degree in "${DEGREE_ARRAY[@]}"; do
  sidecar="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k${HIVA_K}_p${degree}_f${HIVA_FIT_HORIZON}_canonical_lp_mt.parquet"
  summary="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k${HIVA_K}_p${degree}_f${HIVA_FIT_HORIZON}_canonical_lp_mt.summary.json"
  if [[ ! -f "${sidecar}" || ! -f "${summary}" ]]; then
    echo "Missing degree-${degree} sidecar or summary:" >&2
    echo "  ${sidecar}" >&2
    echo "  ${summary}" >&2
    exit 2
  fi

  run_name="smolvla_hiva_coeff_lpmt_v5_d4_6_10_coeffpool_full_ce_mean_k${HIVA_K}_p${degree}_f${HIVA_FIT_HORIZON}_bigflow_b${BATCH_PER_GPU}_s${S}_nores_nodl_${RUN_STAMP}"
  output_dir="${REPO_ROOT}/outputs/train/${run_name}"
  guard_output_dir "${output_dir}" "${RESUME}"

  echo
  echo "======================================================================"
  echo "Launching ${run_name}"
  echo "SIDECAR=${sidecar}"
  echo "SUMMARY=${summary}"
  echo "HIVA_DEGREE=${degree}"
  echo "OUTPUT_DIR=${output_dir}"
  echo "======================================================================"

  TRAIN_ARGS=(
    --policy.type=smolvla_hiva_coeff
    --policy.push_to_hub=false
    --policy.load_vlm_weights=false
    --policy.init_smolvla_checkpoint_path="${INIT_SMOLVLA}"
    --policy.hiva_coeff_sidecar_path="${sidecar}"
    --policy.hiva_coeff_sidecar_summary_path="${summary}"
    --policy.hiva_duration_classes="${HIVA_DURATION_CLASSES}"
    --policy.hiva_dmax="${HIVA_DMAX}"
    --policy.hiva_fit_horizon="${HIVA_FIT_HORIZON}"
    --policy.hiva_k="${HIVA_K}"
    --policy.hiva_degree="${degree}"
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
    --output_dir="${output_dir}"
    --job_name="${run_name}"
    --resume="${RESUME}"
    --eval.batch_size="${EVAL_BATCH_SIZE}"
    --eval.n_episodes="${EVAL_N_EPISODES}"
    --eval_freq="${EVAL_FREQ}"
  )
  TRAIN_ARGS+=("${EVAL_EXTRA_ARGS[@]}")
  TRAIN_ARGS+=("${WANDB_ARGS[@]}")

  "${ACCELERATE_BIN}" launch \
    --num_processes="${NUM_PROCESSES}" \
    --mixed_precision=bf16 \
    "${LEROBOT_TRAIN_BIN}" \
    "${TRAIN_ARGS[@]}"

  echo "Finished ${run_name} at $(date)"
done

echo "LP-MT v5 no-residual/no-decoded-loss P sweep finished at $(date)"
