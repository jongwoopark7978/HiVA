#!/usr/bin/env bash
set -euo pipefail

# Standalone stage-1 residual-only finetune for LP-MT HiVA v5 on bigflow
# using the basis_xattn_transformer residual action head.
#
# Stage 1 loads a frozen non-residual LP-MT HiVA checkpoint, trains only
# model.hiva_residual_head, and uses decoded raw-action loss as the learning
# signal. The residual-only freeze hook lives in hiva_residual_stage1_sitecustomize/.

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
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_hiva_residual_stage1_xattn_bigflow_${RUN_STAMP}.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

GPU_IDS="${GPU_IDS:-5,6,7}"
NUM_GPUS="${NUM_GPUS:-3}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-1024}"
BATCH_SIZE="${BATCH_SIZE:-${BATCH_PER_GPU}}"
S="${S:-2}"
BASE_NUM_GPUS="${BASE_NUM_GPUS:-1}"
BASE_BATCH_PER_GPU="${BASE_BATCH_PER_GPU:-64}"
# Stage 1 trains only the small residual module, so use a smaller base-step
# budget while preserving the same auto-scale formula as the main HiVA runs.
# With 8 GPUs, batch 64/GPU, and S=2 this becomes 500 optimizer steps.
BASE_STEPS="${BASE_STEPS:-8000}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-2.5e-6}"
RESUME="${RESUME:-false}"
EVAL_FREQ="${EVAL_FREQ:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_N_EPISODES="${EVAL_N_EPISODES:-1}"
EVAL_TASK_IDS="${EVAL_TASK_IDS:-}"
EVAL_MAX_PARALLEL_TASKS="${EVAL_MAX_PARALLEL_TASKS:-1}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigflow/add_disk0/jongwoopark}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/train}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
DATA_REPO_ID="${DATA_REPO_ID:-local/libero_lerobot_v3_lerobotkeys}"
TASKS="${TASKS:-libero_spatial,libero_object,libero_goal,libero_10}"

# The stage-1 base checkpoint currently lives in the bigcornea archive. The
# dataset and coefficient sidecar defaults below intentionally use the local
# bigflow disk to avoid slow dataset indexing over the bigcornea mount.
INIT_HIVA_BASE="${INIT_HIVA_BASE:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/HiVA_train/BestS2_smolvla_hiva_coeff_lpmt_coeffpool_job1_v5_d4_6_10_full_ce_mean_k10_f15_bigcornea_b64_s2_20260510_041334/checkpoints/last/pretrained_model}"
SIDECAR="${SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
SIDECAR_SUMMARY="${SIDECAR_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"

HIVA_DURATION_CLASSES="${HIVA_DURATION_CLASSES:-[4,6,10]}"
HIVA_DMAX="${HIVA_DMAX:-10}"
HIVA_FIT_HORIZON="${HIVA_FIT_HORIZON:-15}"
HIVA_K="${HIVA_K:-10}"
HIVA_DEGREE="${HIVA_DEGREE:-3}"
HIVA_BASIS_MODE="${HIVA_BASIS_MODE:-canonical_lp_mt}"
HIVA_DURATION_LOSS="${HIVA_DURATION_LOSS:-ce_mean}"
HIVA_SUFFIX_ATTENTION="${HIVA_SUFFIX_ATTENTION:-full}"
HIVA_DURATION_READOUT="${HIVA_DURATION_READOUT:-coeff_modality_pool}"
HIVA_DURATION_HEAD_TYPE="${HIVA_DURATION_HEAD_TYPE:-residual_ffn}"
HIVA_DURATION_PREDICTION_TYPE="${HIVA_DURATION_PREDICTION_TYPE:-categorical}"
HIVA_DURATION_CONT_NORM="${HIVA_DURATION_CONT_NORM:-bounded}"
HIVA_DURATION_NOISY_SIGMA="${HIVA_DURATION_NOISY_SIGMA:-0.25}"

# Stage-1 residual-only loss: freeze coefficient/duration denoising, train the
# residual module from decoded raw-action supervision.
HIVA_TR_LOSS_WEIGHT="${HIVA_TR_LOSS_WEIGHT:-0.0}"
HIVA_ROT_LOSS_WEIGHT="${HIVA_ROT_LOSS_WEIGHT:-0.0}"
HIVA_GRIP_LOSS_WEIGHT="${HIVA_GRIP_LOSS_WEIGHT:-0.0}"
HIVA_DURATION_NOISY_LOSS_WEIGHT="${HIVA_DURATION_NOISY_LOSS_WEIGHT:-0.0}"
HIVA_DURATION_CLEAN_LOSS_WEIGHT="${HIVA_DURATION_CLEAN_LOSS_WEIGHT:-0.0}"
HIVA_DURATION_FM_LOSS_WEIGHT="${HIVA_DURATION_FM_LOSS_WEIGHT:-0.0}"
HIVA_DECODED_ACTION_LOSS_WEIGHT="${HIVA_DECODED_ACTION_LOSS_WEIGHT:-0.5}"
HIVA_DECODED_TR_LOSS_WEIGHT="${HIVA_DECODED_TR_LOSS_WEIGHT:-1.0}"
HIVA_DECODED_ROT_LOSS_WEIGHT="${HIVA_DECODED_ROT_LOSS_WEIGHT:-10.0}"
HIVA_DECODED_GRIP_LOSS_WEIGHT="${HIVA_DECODED_GRIP_LOSS_WEIGHT:-1.0}"
HIVA_DECODED_PREFIX_WEIGHT="${HIVA_DECODED_PREFIX_WEIGHT:-1.0}"
HIVA_DECODED_POST_DURATION_EXEC_WEIGHT="${HIVA_DECODED_POST_DURATION_EXEC_WEIGHT:-0.5}"
HIVA_DECODED_PREVIEW_WEIGHT="${HIVA_DECODED_PREVIEW_WEIGHT:-0.1}"
HIVA_DECODED_TERMINAL_WEIGHT="${HIVA_DECODED_TERMINAL_WEIGHT:-0.0}"
HIVA_DECODED_LOSS_BETA="${HIVA_DECODED_LOSS_BETA:-0.1}"
HIVA_DECODED_TR_LOSS_BETA="${HIVA_DECODED_TR_LOSS_BETA:-0.1}"
HIVA_DECODED_ROT_LOSS_BETA="${HIVA_DECODED_ROT_LOSS_BETA:-0.1}"
HIVA_DECODED_GRIP_LOSS_BETA="${HIVA_DECODED_GRIP_LOSS_BETA:-0.1}"

HIVA_RESIDUAL_ENABLED="${HIVA_RESIDUAL_ENABLED:-true}"
HIVA_RESIDUAL_MODE="${HIVA_RESIDUAL_MODE:-basis_xattn_transformer}"
HIVA_RESIDUAL_HORIZON="${HIVA_RESIDUAL_HORIZON:-${HIVA_FIT_HORIZON}}"
HIVA_RESIDUAL_FFN_HIDDEN_MULT="${HIVA_RESIDUAL_FFN_HIDDEN_MULT:-4.0}"
HIVA_RESIDUAL_TOKEN_TIME_HIDDEN_MULT="${HIVA_RESIDUAL_TOKEN_TIME_HIDDEN_MULT:-2.0}"
HIVA_RESIDUAL_ALPHA_INIT="${HIVA_RESIDUAL_ALPHA_INIT:-0.1}"
HIVA_RESIDUAL_ZERO_INIT="${HIVA_RESIDUAL_ZERO_INIT:-true}"
HIVA_RESIDUAL_SCALE_MULT="${HIVA_RESIDUAL_SCALE_MULT:-1.0}"
HIVA_RESIDUAL_NUM_BLOCKS="${HIVA_RESIDUAL_NUM_BLOCKS:-9}"
HIVA_RESIDUAL_CROSS_ATTN_HEADS="${HIVA_RESIDUAL_CROSS_ATTN_HEADS:-4}"
HIVA_RESIDUAL_ATTN_DROPOUT="${HIVA_RESIDUAL_ATTN_DROPOUT:-0.0}"
HIVA_RESIDUAL_SCALE_TR="${HIVA_RESIDUAL_SCALE_TR:-}"
HIVA_RESIDUAL_SCALE_ROT="${HIVA_RESIDUAL_SCALE_ROT:-}"
HIVA_RESIDUAL_SCALE_GRIP="${HIVA_RESIDUAL_SCALE_GRIP:-}"

POLICY_CHUNK_SIZE="${POLICY_CHUNK_SIZE:-${HIVA_FIT_HORIZON}}"
POLICY_N_ACTION_STEPS="${POLICY_N_ACTION_STEPS:-${HIVA_DMAX}}"
RUN_NAME="${RUN_NAME:-smolvla_hiva_coeff_lpmt_residual_stage1_xattn_b${HIVA_RESIDUAL_NUM_BLOCKS}_v5_d4_6_10_k${HIVA_K}_f${HIVA_FIT_HORIZON}_bigflow_scale${HIVA_RESIDUAL_SCALE_MULT//./p}_daw${HIVA_DECODED_ACTION_LOSS_WEIGHT//./p}_b${BATCH_PER_GPU}_s${S}_${RUN_STAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${CONDA_ENV_BIN}/accelerate}"
LEROBOT_TRAIN_BIN="${LEROBOT_TRAIN_BIN:-${CONDA_ENV_BIN}/lerobot-train}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/jongwoo_hf_datasets_cache}"
export HIVA_TRAIN_RESIDUAL_ONLY=true
export PATH="${CONDA_ENV_BIN}:${PATH}"
export PYTHONPATH="${SCRIPT_DIR}/hiva_residual_stage1_sitecustomize:${REPO_ROOT}/src:${PYTHONPATH:-}"
mkdir -p "${HF_DATASETS_CACHE}" "${OUTPUT_ROOT}"

for required_path in "${DATA_ROOT}" "${SIDECAR}" "${SIDECAR_SUMMARY}" "${INIT_HIVA_BASE}" "${SCRIPT_DIR}/hiva_residual_stage1_sitecustomize/sitecustomize.py"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "ERROR: required path does not exist: ${required_path}" >&2
    exit 2
  fi
done
if [[ -e "${OUTPUT_DIR}" && "${RESUME}" != "true" ]]; then
  echo "ERROR: output directory already exists and RESUME!=true: ${OUTPUT_DIR}" >&2
  exit 3
fi

GLOBAL_BATCH_SIZE=$((NUM_GPUS * BATCH_PER_GPU))
BASE_GLOBAL_BATCH_SIZE=$((BASE_NUM_GPUS * BASE_BATCH_PER_GPU))
STEPS="${STEPS:-$("${CONDA_ENV_BIN}/python" - <<PY
import math
steps = math.ceil(int("${BASE_STEPS}") * int("${BASE_GLOBAL_BATCH_SIZE}") / int("${GLOBAL_BATCH_SIZE}") / float("${S}"))
print(max(1, steps))
PY
)}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-$("${CONDA_ENV_BIN}/python" - <<PY
import math
print(max(1, math.ceil(int("${STEPS}") * float("${WARMUP_RATIO}"))))
PY
)}"
SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-${STEPS}}"
SAVE_FREQ="${SAVE_FREQ:-$("${CONDA_ENV_BIN}/python" - <<PY
steps = int("${STEPS}")
print(max(1, (steps + 1) // 2))
PY
)}"
SAVE_STEPS="${SAVE_STEPS:-}"

OPTIONAL_ARGS=()
if [[ -n "${EVAL_TASK_IDS}" ]]; then
  OPTIONAL_ARGS+=(--env.task_ids="${EVAL_TASK_IDS}")
fi
if [[ -n "${HIVA_RESIDUAL_SCALE_TR}" ]]; then
  OPTIONAL_ARGS+=(--policy.hiva_residual_scale_tr="${HIVA_RESIDUAL_SCALE_TR}")
fi
if [[ -n "${HIVA_RESIDUAL_SCALE_ROT}" ]]; then
  OPTIONAL_ARGS+=(--policy.hiva_residual_scale_rot="${HIVA_RESIDUAL_SCALE_ROT}")
fi
if [[ -n "${HIVA_RESIDUAL_SCALE_GRIP}" ]]; then
  OPTIONAL_ARGS+=(--policy.hiva_residual_scale_grip="${HIVA_RESIDUAL_SCALE_GRIP}")
fi
if [[ -n "${SAVE_STEPS}" ]]; then
  OPTIONAL_ARGS+=(--save_steps="${SAVE_STEPS}")
fi

WANDB_ARGS=()
if [[ "${WANDB_ENABLE}" == "true" ]]; then
  WANDB_ARGS+=(
    --wandb.enable=true
    --wandb.project="${WANDB_PROJECT}"
    --wandb.disable_artifact="${WANDB_DISABLE_ARTIFACT}"
    --wandb.mode="${WANDB_MODE}"
  )
else
  WANDB_ARGS+=(--wandb.enable=false)
fi

TRAIN_ARGS=(
  --policy.type=smolvla_hiva_coeff
  --policy.push_to_hub=false
  --policy.load_vlm_weights=false
  --policy.init_smolvla_checkpoint_path="${INIT_HIVA_BASE}"
  --policy.init_smolvla_strict=false
  --policy.hiva_coeff_sidecar_path="${SIDECAR}"
  --policy.hiva_coeff_sidecar_summary_path="${SIDECAR_SUMMARY}"
  --policy.hiva_duration_classes="${HIVA_DURATION_CLASSES}"
  --policy.hiva_dmax="${HIVA_DMAX}"
  --policy.hiva_fit_horizon="${HIVA_FIT_HORIZON}"
  --policy.hiva_k="${HIVA_K}"
  --policy.hiva_degree="${HIVA_DEGREE}"
  --policy.hiva_basis_mode="${HIVA_BASIS_MODE}"
  --policy.hiva_duration_loss="${HIVA_DURATION_LOSS}"
  --policy.hiva_duration_prediction_type="${HIVA_DURATION_PREDICTION_TYPE}"
  --policy.hiva_duration_readout="${HIVA_DURATION_READOUT}"
  --policy.hiva_duration_fm_loss_weight="${HIVA_DURATION_FM_LOSS_WEIGHT}"
  --policy.hiva_duration_cont_norm="${HIVA_DURATION_CONT_NORM}"
  --policy.hiva_suffix_attention="${HIVA_SUFFIX_ATTENTION}"
  --policy.hiva_duration_head_type="${HIVA_DURATION_HEAD_TYPE}"
  --policy.hiva_duration_ffn_hidden_mult=4.0
  --policy.hiva_duration_ffn_alpha_init=0.1
  --policy.hiva_tr_loss_weight="${HIVA_TR_LOSS_WEIGHT}"
  --policy.hiva_rot_loss_weight="${HIVA_ROT_LOSS_WEIGHT}"
  --policy.hiva_grip_loss_weight="${HIVA_GRIP_LOSS_WEIGHT}"
  --policy.hiva_duration_noisy_loss_weight="${HIVA_DURATION_NOISY_LOSS_WEIGHT}"
  --policy.hiva_duration_clean_loss_weight="${HIVA_DURATION_CLEAN_LOSS_WEIGHT}"
  --policy.hiva_duration_noisy_sigma="${HIVA_DURATION_NOISY_SIGMA}"
  --policy.hiva_decoded_action_loss_weight="${HIVA_DECODED_ACTION_LOSS_WEIGHT}"
  --policy.hiva_decoded_tr_loss_weight="${HIVA_DECODED_TR_LOSS_WEIGHT}"
  --policy.hiva_decoded_rot_loss_weight="${HIVA_DECODED_ROT_LOSS_WEIGHT}"
  --policy.hiva_decoded_grip_loss_weight="${HIVA_DECODED_GRIP_LOSS_WEIGHT}"
  --policy.hiva_decoded_prefix_weight="${HIVA_DECODED_PREFIX_WEIGHT}"
  --policy.hiva_decoded_post_duration_exec_weight="${HIVA_DECODED_POST_DURATION_EXEC_WEIGHT}"
  --policy.hiva_decoded_preview_weight="${HIVA_DECODED_PREVIEW_WEIGHT}"
  --policy.hiva_decoded_terminal_weight="${HIVA_DECODED_TERMINAL_WEIGHT}"
  --policy.hiva_decoded_loss_beta="${HIVA_DECODED_LOSS_BETA}"
  --policy.hiva_decoded_tr_loss_beta="${HIVA_DECODED_TR_LOSS_BETA}"
  --policy.hiva_decoded_rot_loss_beta="${HIVA_DECODED_ROT_LOSS_BETA}"
  --policy.hiva_decoded_grip_loss_beta="${HIVA_DECODED_GRIP_LOSS_BETA}"
  --policy.hiva_residual_enabled="${HIVA_RESIDUAL_ENABLED}"
  --policy.hiva_residual_mode="${HIVA_RESIDUAL_MODE}"
  --policy.hiva_residual_horizon="${HIVA_RESIDUAL_HORIZON}"
  --policy.hiva_residual_scale_mult="${HIVA_RESIDUAL_SCALE_MULT}"
  --policy.hiva_residual_ffn_hidden_mult="${HIVA_RESIDUAL_FFN_HIDDEN_MULT}"
  --policy.hiva_residual_token_time_hidden_mult="${HIVA_RESIDUAL_TOKEN_TIME_HIDDEN_MULT}"
  --policy.hiva_residual_alpha_init="${HIVA_RESIDUAL_ALPHA_INIT}"
  --policy.hiva_residual_zero_init="${HIVA_RESIDUAL_ZERO_INIT}"
  --policy.hiva_residual_num_blocks="${HIVA_RESIDUAL_NUM_BLOCKS}"
  --policy.hiva_residual_cross_attn_heads="${HIVA_RESIDUAL_CROSS_ATTN_HEADS}"
  --policy.hiva_residual_attn_dropout="${HIVA_RESIDUAL_ATTN_DROPOUT}"
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
TRAIN_ARGS+=("${OPTIONAL_ARGS[@]}")
TRAIN_ARGS+=("${WANDB_ARGS[@]}")

cat <<INFO
Stage 1 residual-only HiVA finetune
Host: $(hostname)
RUN_NAME=${RUN_NAME}
OUTPUT_DIR=${OUTPUT_DIR}
LOG_FILE=${LOG_FILE}
INIT_HIVA_BASE=${INIT_HIVA_BASE}
DATA_ROOT=${DATA_ROOT}
SIDECAR=${SIDECAR}
SIDECAR_SUMMARY=${SIDECAR_SUMMARY}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
NUM_PROCESSES=${NUM_PROCESSES}
BATCH_PER_GPU=${BATCH_PER_GPU}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}
STEPS=${STEPS}
SAVE_FREQ=${SAVE_FREQ}
SAVE_STEPS=${SAVE_STEPS:-<none>}
HIVA_DECODED_ACTION_LOSS_WEIGHT=${HIVA_DECODED_ACTION_LOSS_WEIGHT}
HIVA_DECODED_ROT_LOSS_WEIGHT=${HIVA_DECODED_ROT_LOSS_WEIGHT}
HIVA_DECODED_TR_LOSS_BETA=${HIVA_DECODED_TR_LOSS_BETA}
HIVA_DECODED_ROT_LOSS_BETA=${HIVA_DECODED_ROT_LOSS_BETA}
HIVA_DECODED_GRIP_LOSS_BETA=${HIVA_DECODED_GRIP_LOSS_BETA}
HIVA_RESIDUAL_ENABLED=${HIVA_RESIDUAL_ENABLED}
HIVA_RESIDUAL_MODE=${HIVA_RESIDUAL_MODE}
HIVA_RESIDUAL_SCALE_MULT=${HIVA_RESIDUAL_SCALE_MULT}
HIVA_TRAIN_RESIDUAL_ONLY=${HIVA_TRAIN_RESIDUAL_ONLY}
INFO

"${ACCELERATE_BIN}" launch \
  --num_processes="${NUM_PROCESSES}" \
  --mixed_precision=bf16 \
  "${LEROBOT_TRAIN_BIN}" \
  "${TRAIN_ARGS[@]}"
