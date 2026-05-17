#!/usr/bin/env bash
set -euo pipefail

# BigBrain stage-0 LP-MT HiVA coefficient finetune with P=5 B-spline
# coefficients. Hyperparameters follow the archived BestP5 no-residual/no-
# decoded-loss run, with direct control over total training steps.
#
# Default:
#   GPUs: 6,7
#   per-GPU batch size: 160
#   total steps: 5000
#   milestone checkpoints: 0.25, 0.30, 0.375, 0.40, 0.50, 0.60, 0.625,
#                         0.70, 0.75, 0.875, 1.0 of total steps
#
# Example:
#   bash server_scripts/bigbrain/run_hiva_coeff_lpmt_stage0_p5_s0p5_5000_bigbrain.sh --total-steps 5000

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

TOTAL_STEPS="${TOTAL_STEPS:-5000}"
DRY_RUN="${DRY_RUN:-0}"
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --total-steps)
      TOTAL_STEPS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if ! [[ "${TOTAL_STEPS}" =~ ^[0-9]+$ ]] || [[ "${TOTAL_STEPS}" -le 0 ]]; then
  echo "TOTAL_STEPS must be a positive integer, got: ${TOTAL_STEPS}" >&2
  exit 2
fi

LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/hiva_coeff_lpmt_stage0_p5_s0p5_5000_bigbrain_${RUN_STAMP}.log}"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigbrain/add_disk0/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
DATA_REPO_ID="${DATA_REPO_ID:-local/libero_lerobot_v3_lerobotkeys}"
INIT_SMOLVLA="${INIT_SMOLVLA:-/home/jongwoopark/lerobot/smolvla_libero}"

GPU_IDS="${GPU_IDS:-6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
BATCH_PER_GPU="${BATCH_PER_GPU:-160}"
BATCH_SIZE="${BATCH_SIZE:-${BATCH_PER_GPU}}"
S="${S:-0.5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-$(python - <<PY
import math
print(max(1, math.ceil(int("${TOTAL_STEPS}") * float("${WARMUP_RATIO}"))))
PY
)}"
SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-${TOTAL_STEPS}}"
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-2.5e-6}"

SAVE_STEPS="${SAVE_STEPS:-$(python - <<PY
import math
steps = int("${TOTAL_STEPS}")
fractions = (0.25, 0.30, 0.375, 0.40, 0.50, 0.60, 0.625, 0.70, 0.75, 0.875, 1.0)
milestones = sorted({min(steps, max(1, math.ceil(steps * f))) for f in fractions})
print("[" + ",".join(str(s) for s in milestones) + "]")
PY
)}"

HIVA_DEGREE="${HIVA_DEGREE:-5}"
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

# Match the BestP5 no-residual/no-decoded-loss setup.
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

POLICY_CHUNK_SIZE="${POLICY_CHUNK_SIZE:-15}"
POLICY_N_ACTION_STEPS="${POLICY_N_ACTION_STEPS:-10}"
TASKS="${TASKS:-libero_spatial,libero_object,libero_goal,libero_10}"
EVAL_FREQ="${EVAL_FREQ:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_N_EPISODES="${EVAL_N_EPISODES:-1}"
EVAL_MAX_PARALLEL_TASKS="${EVAL_MAX_PARALLEL_TASKS:-1}"

WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}"
WANDB_NOTES="${WANDB_NOTES:-BigBrain stage0 P5 LP-MT S=${S}, total_steps=${TOTAL_STEPS}, b${BATCH_PER_GPU}x${NUM_PROCESSES}}"

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

SIDECAR="${SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k${HIVA_K}_p${HIVA_DEGREE}_f${HIVA_FIT_HORIZON}_canonical_lp_mt.parquet}"
SIDECAR_SUMMARY="${SIDECAR_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k${HIVA_K}_p${HIVA_DEGREE}_f${HIVA_FIT_HORIZON}_canonical_lp_mt.summary.json}"

for required_path in "${SIDECAR}" "${SIDECAR_SUMMARY}" "${DATA_ROOT}" "${INIT_SMOLVLA}" "${ACCELERATE_BIN}" "${LEROBOT_TRAIN_BIN}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "ERROR: required path does not exist: ${required_path}" >&2
    exit 2
  fi
done

S_TAG="${S//./p}"
RUN_NAME="${RUN_NAME:-smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k${HIVA_K}_p${HIVA_DEGREE}_f${HIVA_FIT_HORIZON}_bigbrain_b${BATCH_PER_GPU}_g${NUM_PROCESSES}_s${S_TAG}_steps${TOTAL_STEPS}_${RUN_STAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/${RUN_NAME}}"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "ERROR: OUTPUT_DIR already exists: ${OUTPUT_DIR}" >&2
  exit 2
fi

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

echo "Stage-0 LP-MT HiVA P5 BigBrain finetune"
echo "RUN_NAME=${RUN_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "GPU_IDS=${GPU_IDS}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "BATCH_PER_GPU=${BATCH_PER_GPU}"
echo "EFFECTIVE_BATCH=$((BATCH_SIZE * NUM_PROCESSES))"
echo "S=${S}"
echo "TOTAL_STEPS=${TOTAL_STEPS}"
echo "SCHEDULER_WARMUP_STEPS=${SCHEDULER_WARMUP_STEPS}"
echo "SCHEDULER_DECAY_STEPS=${SCHEDULER_DECAY_STEPS}"
echo "SAVE_STEPS=${SAVE_STEPS}"
echo "SIDECAR=${SIDECAR}"
echo "SIDECAR_SUMMARY=${SIDECAR_SUMMARY}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "INIT_SMOLVLA=${INIT_SMOLVLA}"
echo "QUEUE_LOG=${QUEUE_LOG}"
echo "WANDB_ENABLE=${WANDB_ENABLE}"

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
  --policy.hiva_degree_tr="${HIVA_DEGREE}"
  --policy.hiva_degree_rot="${HIVA_DEGREE}"
  --policy.hiva_degree_grip="${HIVA_DEGREE}"
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
  --steps="${TOTAL_STEPS}"
  --log_freq=1
  --save_checkpoint=true
  --save_freq=0
  --save_steps="${SAVE_STEPS}"
  --num_workers=0
  --seed=1000
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
  --resume=false
  --eval.batch_size="${EVAL_BATCH_SIZE}"
  --eval.n_episodes="${EVAL_N_EPISODES}"
  --eval_freq="${EVAL_FREQ}"
)
TRAIN_ARGS+=("${WANDB_ARGS[@]}")

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'DRY_RUN command:\nCUDA_VISIBLE_DEVICES=%q %q launch --num_processes=%q --mixed_precision=bf16 %q' \
    "${CUDA_VISIBLE_DEVICES}" "${ACCELERATE_BIN}" "${NUM_PROCESSES}" "${LEROBOT_TRAIN_BIN}"
  printf ' %q' "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
fi

"${ACCELERATE_BIN}" launch \
  --num_processes="${NUM_PROCESSES}" \
  --mixed_precision=bf16 \
  "${LEROBOT_TRAIN_BIN}" \
  "${TRAIN_ARGS[@]}"

echo "Stage-0 P5 finetune finished at $(date)"
