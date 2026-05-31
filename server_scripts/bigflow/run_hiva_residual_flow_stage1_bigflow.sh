#!/usr/bin/env bash
set -euo pipefail

# Standalone smoke/full runner for the SmolVLA-style HiVA residual-flow branch.
# Defaults are intentionally tiny and GPU4-only for smoke testing. Override
# STEPS, BATCH_PER_GPU, GPU_IDS, NUM_GPUS, WANDB_ENABLE, etc. for longer runs.

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
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_hiva_residual_flow_stage1_bigflow_${RUN_STAMP}.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

GPU_IDS="${GPU_IDS:-4}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-2}"
BATCH_SIZE="${BATCH_SIZE:-${BATCH_PER_GPU}}"
STEPS="${STEPS:-2}"
SAVE_FREQ="${SAVE_FREQ:-${STEPS}}"
SAVE_STEPS="${SAVE_STEPS:-}"
RESUME="${RESUME:-false}"
EVAL_FREQ="${EVAL_FREQ:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_N_EPISODES="${EVAL_N_EPISODES:-1}"
EVAL_MAX_PARALLEL_TASKS="${EVAL_MAX_PARALLEL_TASKS:-1}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-1}"
SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-${STEPS}}"
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-2.5e-6}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigflow/add_disk0/jongwoopark}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/train}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
DATA_REPO_ID="${DATA_REPO_ID:-local/libero_lerobot_v3_lerobotkeys}"
TASKS="${TASKS:-libero_spatial,libero_object,libero_goal,libero_10}"

# Stage-0 HiVA checkpoint used as the default HiVA weight loading checkpoint.
INIT_HIVA_BASE="${INIT_HIVA_BASE:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/HiVA_train/BestS2_smolvla_hiva_coeff_lpmt_coeffpool_job1_v5_d4_6_10_full_ce_mean_k10_f15_bigcornea_b64_s2_20260510_041334/checkpoints/last/pretrained_model}"
# Original SmolVLA-LIBERO checkpoint used for residual-flow suffix modules and separate expert.
ORIGINAL_SMOLVLA="${ORIGINAL_SMOLVLA:-/home/jongwoopark/lerobot/smolvla_libero}"
HIVA_RESIDUAL_FLOW_INIT_SMOLVLA="${HIVA_RESIDUAL_FLOW_INIT_SMOLVLA:-${ORIGINAL_SMOLVLA}}"

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
HIVA_DURATION_NOISY_SIGMA="${HIVA_DURATION_NOISY_SIGMA:-0.25}"

# Freeze coefficient and duration objectives for residual-flow-only stage-1 training.
HIVA_TR_LOSS_WEIGHT="${HIVA_TR_LOSS_WEIGHT:-0.0}"
HIVA_ROT_LOSS_WEIGHT="${HIVA_ROT_LOSS_WEIGHT:-0.0}"
HIVA_GRIP_LOSS_WEIGHT="${HIVA_GRIP_LOSS_WEIGHT:-0.0}"
HIVA_DURATION_NOISY_LOSS_WEIGHT="${HIVA_DURATION_NOISY_LOSS_WEIGHT:-0.0}"
HIVA_DURATION_CLEAN_LOSS_WEIGHT="${HIVA_DURATION_CLEAN_LOSS_WEIGHT:-0.0}"
HIVA_DURATION_FM_LOSS_WEIGHT="${HIVA_DURATION_FM_LOSS_WEIGHT:-0.0}"
HIVA_DECODED_ACTION_LOSS_WEIGHT="${HIVA_DECODED_ACTION_LOSS_WEIGHT:-0.0}"

HIVA_RESIDUAL_ENABLED="${HIVA_RESIDUAL_ENABLED:-false}"
HIVA_RESIDUAL_FLOW_ENABLED="${HIVA_RESIDUAL_FLOW_ENABLED:-true}"
HIVA_RESIDUAL_FLOW_CONDITIONING="${HIVA_RESIDUAL_FLOW_CONDITIONING:-v1_minimal}"
HIVA_RESIDUAL_FLOW_HORIZON="${HIVA_RESIDUAL_FLOW_HORIZON:-${HIVA_FIT_HORIZON}}"
HIVA_RESIDUAL_FLOW_STEPS="${HIVA_RESIDUAL_FLOW_STEPS:-10}"
HIVA_RESIDUAL_FLOW_LOSS_WEIGHT="${HIVA_RESIDUAL_FLOW_LOSS_WEIGHT:-1.0}"
HIVA_RESIDUAL_FLOW_DECODED_LOSS_WEIGHT="${HIVA_RESIDUAL_FLOW_DECODED_LOSS_WEIGHT:-1.0}"
HIVA_RESIDUAL_FLOW_SCALE_TR="${HIVA_RESIDUAL_FLOW_SCALE_TR:-3.0}"
HIVA_RESIDUAL_FLOW_SCALE_ROT="${HIVA_RESIDUAL_FLOW_SCALE_ROT:-3.0}"
HIVA_RESIDUAL_FLOW_SCALE_GRIP="${HIVA_RESIDUAL_FLOW_SCALE_GRIP:-0.5}"
HIVA_RESIDUAL_FLOW_INFERENCE_WEIGHT="${HIVA_RESIDUAL_FLOW_INFERENCE_WEIGHT:-1.0}"
HIVA_RESIDUAL_FLOW_OUT_HEAD_INIT="${HIVA_RESIDUAL_FLOW_OUT_HEAD_INIT:-copy}"
HIVA_RESIDUAL_FLOW_USE_SEPARATE_EXPERT="${HIVA_RESIDUAL_FLOW_USE_SEPARATE_EXPERT:-true}"
HIVA_RESIDUAL_FLOW_DECODED_LOSS_BETA="${HIVA_RESIDUAL_FLOW_DECODED_LOSS_BETA:-0.1}"

POLICY_CHUNK_SIZE="${POLICY_CHUNK_SIZE:-${HIVA_FIT_HORIZON}}"
POLICY_N_ACTION_STEPS="${POLICY_N_ACTION_STEPS:-${HIVA_DMAX}}"
RUN_NAME="${RUN_NAME:-smolvla_hiva_coeff_lpmt_residual_flow_stage1_bigflow_smoke_gpu${GPU_IDS//,/}_b${BATCH_PER_GPU}_steps${STEPS}_${RUN_STAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${CONDA_ENV_BIN}/accelerate}"
LEROBOT_TRAIN_BIN="${LEROBOT_TRAIN_BIN:-${CONDA_ENV_BIN}/lerobot-train}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
source "/home/jongwoopark/lerobot/server_scripts/common_hf_cache.sh"
setup_hf_datasets_cache
export HIVA_TRAIN_RESIDUAL_FLOW_ONLY=true
export PATH="${CONDA_ENV_BIN}:${PATH}"
export PYTHONPATH="${SCRIPT_DIR}/hiva_residual_flow_stage1_sitecustomize:${REPO_ROOT}/src:${PYTHONPATH:-}"
mkdir -p "${HF_DATASETS_CACHE}" "${OUTPUT_ROOT}"

for required_path in \
  "${DATA_ROOT}" \
  "${SIDECAR}" \
  "${SIDECAR_SUMMARY}" \
  "${INIT_HIVA_BASE}" \
  "${HIVA_RESIDUAL_FLOW_INIT_SMOLVLA}" \
  "${SCRIPT_DIR}/hiva_residual_flow_stage1_sitecustomize/sitecustomize.py"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "ERROR: required path does not exist: ${required_path}" >&2
    exit 2
  fi
done
if [[ -e "${OUTPUT_DIR}" && "${RESUME}" != "true" ]]; then
  echo "ERROR: output directory already exists and RESUME!=true: ${OUTPUT_DIR}" >&2
  exit 3
fi

OPTIONAL_ARGS=()
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
  --policy.hiva_residual_flow_init_smolvla_checkpoint_path="${HIVA_RESIDUAL_FLOW_INIT_SMOLVLA}"
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
  --policy.hiva_suffix_attention="${HIVA_SUFFIX_ATTENTION}"
  --policy.hiva_duration_head_type="${HIVA_DURATION_HEAD_TYPE}"
  --policy.hiva_tr_loss_weight="${HIVA_TR_LOSS_WEIGHT}"
  --policy.hiva_rot_loss_weight="${HIVA_ROT_LOSS_WEIGHT}"
  --policy.hiva_grip_loss_weight="${HIVA_GRIP_LOSS_WEIGHT}"
  --policy.hiva_duration_noisy_loss_weight="${HIVA_DURATION_NOISY_LOSS_WEIGHT}"
  --policy.hiva_duration_clean_loss_weight="${HIVA_DURATION_CLEAN_LOSS_WEIGHT}"
  --policy.hiva_duration_fm_loss_weight="${HIVA_DURATION_FM_LOSS_WEIGHT}"
  --policy.hiva_duration_noisy_sigma="${HIVA_DURATION_NOISY_SIGMA}"
  --policy.hiva_decoded_action_loss_weight="${HIVA_DECODED_ACTION_LOSS_WEIGHT}"
  --policy.hiva_residual_enabled="${HIVA_RESIDUAL_ENABLED}"
  --policy.hiva_residual_flow_enabled="${HIVA_RESIDUAL_FLOW_ENABLED}"
  --policy.hiva_residual_flow_conditioning="${HIVA_RESIDUAL_FLOW_CONDITIONING}"
  --policy.hiva_residual_flow_horizon="${HIVA_RESIDUAL_FLOW_HORIZON}"
  --policy.hiva_residual_flow_steps="${HIVA_RESIDUAL_FLOW_STEPS}"
  --policy.hiva_residual_flow_loss_weight="${HIVA_RESIDUAL_FLOW_LOSS_WEIGHT}"
  --policy.hiva_residual_flow_decoded_loss_weight="${HIVA_RESIDUAL_FLOW_DECODED_LOSS_WEIGHT}"
  --policy.hiva_residual_flow_scale_tr="${HIVA_RESIDUAL_FLOW_SCALE_TR}"
  --policy.hiva_residual_flow_scale_rot="${HIVA_RESIDUAL_FLOW_SCALE_ROT}"
  --policy.hiva_residual_flow_scale_grip="${HIVA_RESIDUAL_FLOW_SCALE_GRIP}"
  --policy.hiva_residual_flow_inference_weight="${HIVA_RESIDUAL_FLOW_INFERENCE_WEIGHT}"
  --policy.hiva_residual_flow_out_head_init="${HIVA_RESIDUAL_FLOW_OUT_HEAD_INIT}"
  --policy.hiva_residual_flow_use_separate_expert="${HIVA_RESIDUAL_FLOW_USE_SEPARATE_EXPERT}"
  --policy.hiva_residual_flow_decoded_loss_beta="${HIVA_RESIDUAL_FLOW_DECODED_LOSS_BETA}"
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
HiVA residual-flow stage-1 run
Host: $(hostname)
RUN_NAME=${RUN_NAME}
OUTPUT_DIR=${OUTPUT_DIR}
LOG_FILE=${LOG_FILE}
INIT_HIVA_BASE=${INIT_HIVA_BASE}
ORIGINAL_SMOLVLA=${ORIGINAL_SMOLVLA}
HIVA_RESIDUAL_FLOW_INIT_SMOLVLA=${HIVA_RESIDUAL_FLOW_INIT_SMOLVLA}
HIVA_RESIDUAL_FLOW_USE_SEPARATE_EXPERT=${HIVA_RESIDUAL_FLOW_USE_SEPARATE_EXPERT}
DATA_ROOT=${DATA_ROOT}
SIDECAR=${SIDECAR}
SIDECAR_SUMMARY=${SIDECAR_SUMMARY}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
NUM_PROCESSES=${NUM_PROCESSES}
BATCH_PER_GPU=${BATCH_PER_GPU}
STEPS=${STEPS}
SAVE_FREQ=${SAVE_FREQ}
SAVE_STEPS=${SAVE_STEPS:-<none>}
HIVA_RESIDUAL_FLOW_CONDITIONING=${HIVA_RESIDUAL_FLOW_CONDITIONING}
HIVA_RESIDUAL_FLOW_HORIZON=${HIVA_RESIDUAL_FLOW_HORIZON}
HIVA_RESIDUAL_FLOW_LOSS_WEIGHT=${HIVA_RESIDUAL_FLOW_LOSS_WEIGHT}
HIVA_RESIDUAL_FLOW_DECODED_LOSS_WEIGHT=${HIVA_RESIDUAL_FLOW_DECODED_LOSS_WEIGHT}
INFO

"${ACCELERATE_BIN}" launch \
  --num_processes="${NUM_PROCESSES}" \
  --mixed_precision=bf16 \
  "${LEROBOT_TRAIN_BIN}" \
  "${TRAIN_ARGS[@]}"
