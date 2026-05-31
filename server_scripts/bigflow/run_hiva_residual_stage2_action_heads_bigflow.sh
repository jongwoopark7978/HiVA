#!/usr/bin/env bash
set -euo pipefail

# Standalone Stage-2 co-adaptation finetune for LP-MT residual HiVA v5 on bigflow.
#
# Stage 2 loads a Stage-1 residual checkpoint, freezes the VLM/action expert,
# coefficient input projections, and action-time MLPs, and trains only:
#   model.hiva_residual_head.*
#   model.hiva_tr_out_proj.*
#   model.hiva_rot_out_proj.*
#   model.hiva_grip_out_proj.*
#   model.hiva_duration_head.*
#
# This is intended to let the B-spline coefficient output heads, duration readout,
# and residual action head co-adapt while coefficient FM + duration CE anchor the
# original HiVA behavior.
#
# Required override:
#   INIT_HIVA_STAGE1=/path/to/stage1/checkpoints/.../pretrained_model

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
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_hiva_residual_stage2_action_heads_bigflow_${RUN_STAMP}.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

GPU_IDS="${GPU_IDS:-5,6,7}"
NUM_GPUS="${NUM_GPUS:-3}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-512}"
BATCH_SIZE="${BATCH_SIZE:-${BATCH_PER_GPU}}"
S="${S:-2}"
BASE_NUM_GPUS="${BASE_NUM_GPUS:-1}"
BASE_BATCH_PER_GPU="${BASE_BATCH_PER_GPU:-64}"
# Stage 2 is a small-head co-adaptation stage. With 8 GPUs, b64/GPU, S=2 this is 1250 steps.
BASE_STEPS="${BASE_STEPS:-20000}"
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

# Stage 2 should normally start from the best Stage-1 residual checkpoint.
INIT_HIVA_STAGE1="${INIT_HIVA_STAGE1:-}"
SIDECAR="${SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
SIDECAR_SUMMARY="${SIDECAR_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"

HIVA_DURATION_CLASSES="${HIVA_DURATION_CLASSES:-[4,6,10]}"
HIVA_DMAX="${HIVA_DMAX:-10}"
HIVA_FIT_HORIZON="${HIVA_FIT_HORIZON:-15}"
HIVA_K="${HIVA_K:-10}"
HIVA_DEGREE="${HIVA_DEGREE:-3}"
HIVA_DEGREE_TR="${HIVA_DEGREE_TR:-}"
HIVA_DEGREE_ROT="${HIVA_DEGREE_ROT:-}"
HIVA_DEGREE_GRIP="${HIVA_DEGREE_GRIP:-}"
HIVA_BASIS_MODE="${HIVA_BASIS_MODE:-canonical_lp_mt}"
HIVA_DURATION_LOSS="${HIVA_DURATION_LOSS:-ce_mean}"
HIVA_SUFFIX_ATTENTION="${HIVA_SUFFIX_ATTENTION:-full}"
HIVA_DURATION_READOUT="${HIVA_DURATION_READOUT:-coeff_modality_pool}"
HIVA_DURATION_HEAD_TYPE="${HIVA_DURATION_HEAD_TYPE:-residual_ffn}"
HIVA_DURATION_PREDICTION_TYPE="${HIVA_DURATION_PREDICTION_TYPE:-categorical}"
HIVA_DURATION_CONT_NORM="${HIVA_DURATION_CONT_NORM:-bounded}"
HIVA_DURATION_NOISY_SIGMA="${HIVA_DURATION_NOISY_SIGMA:-0.25}"

# Stage-2 anchor losses: restore coefficient FM and duration CE.
HIVA_TR_LOSS_WEIGHT="${HIVA_TR_LOSS_WEIGHT:-1.0}"
HIVA_ROT_LOSS_WEIGHT="${HIVA_ROT_LOSS_WEIGHT:-1.0}"
HIVA_GRIP_LOSS_WEIGHT="${HIVA_GRIP_LOSS_WEIGHT:-1.0}"
HIVA_DURATION_NOISY_LOSS_WEIGHT="${HIVA_DURATION_NOISY_LOSS_WEIGHT:-1.0}"
HIVA_DURATION_CLEAN_LOSS_WEIGHT="${HIVA_DURATION_CLEAN_LOSS_WEIGHT:-0.0}"
HIVA_DURATION_FM_LOSS_WEIGHT="${HIVA_DURATION_FM_LOSS_WEIGHT:-0.0}"

# Stage-2 decoded loss is intentionally smaller than Stage 1 by default.
HIVA_DECODED_ACTION_LOSS_WEIGHT="${HIVA_DECODED_ACTION_LOSS_WEIGHT:-0.1}"
HIVA_DECODED_TR_LOSS_WEIGHT="${HIVA_DECODED_TR_LOSS_WEIGHT:-1.0}"
HIVA_DECODED_ROT_LOSS_WEIGHT="${HIVA_DECODED_ROT_LOSS_WEIGHT:-10.0}"
HIVA_DECODED_GRIP_LOSS_WEIGHT="${HIVA_DECODED_GRIP_LOSS_WEIGHT:-1.0}"
HIVA_DECODED_PREFIX_WEIGHT="${HIVA_DECODED_PREFIX_WEIGHT:-1.0}"
HIVA_DECODED_POST_DURATION_EXEC_WEIGHT="${HIVA_DECODED_POST_DURATION_EXEC_WEIGHT:-0.5}"
HIVA_DECODED_PREVIEW_WEIGHT="${HIVA_DECODED_PREVIEW_WEIGHT:-0.1}"
HIVA_DECODED_TERMINAL_WEIGHT="${HIVA_DECODED_TERMINAL_WEIGHT:-0.0}"
HIVA_DECODED_LOSS_BETA="${HIVA_DECODED_LOSS_BETA:-0.1}"
HIVA_DECODED_TR_LOSS_BETA="${HIVA_DECODED_TR_LOSS_BETA:-}"
HIVA_DECODED_ROT_LOSS_BETA="${HIVA_DECODED_ROT_LOSS_BETA:-}"
HIVA_DECODED_GRIP_LOSS_BETA="${HIVA_DECODED_GRIP_LOSS_BETA:-}"

HIVA_RESIDUAL_ENABLED="${HIVA_RESIDUAL_ENABLED:-true}"
HIVA_RESIDUAL_MODE="${HIVA_RESIDUAL_MODE:-basis_xattn_transformer}"
HIVA_RESIDUAL_HORIZON="${HIVA_RESIDUAL_HORIZON:-${HIVA_FIT_HORIZON}}"
HIVA_RESIDUAL_FFN_HIDDEN_MULT="${HIVA_RESIDUAL_FFN_HIDDEN_MULT:-4.0}"
HIVA_RESIDUAL_TOKEN_TIME_HIDDEN_MULT="${HIVA_RESIDUAL_TOKEN_TIME_HIDDEN_MULT:-2.0}"
HIVA_RESIDUAL_ALPHA_INIT="${HIVA_RESIDUAL_ALPHA_INIT:-0.1}"
HIVA_RESIDUAL_ZERO_INIT="${HIVA_RESIDUAL_ZERO_INIT:-true}"
HIVA_RESIDUAL_NUM_BLOCKS="${HIVA_RESIDUAL_NUM_BLOCKS:-4}"
HIVA_RESIDUAL_CROSS_ATTN_HEADS="${HIVA_RESIDUAL_CROSS_ATTN_HEADS:-4}"
HIVA_RESIDUAL_ATTN_DROPOUT="${HIVA_RESIDUAL_ATTN_DROPOUT:-0.0}"
HIVA_RESIDUAL_SCALE_MULT="${HIVA_RESIDUAL_SCALE_MULT:-1.0}"
HIVA_RESIDUAL_SCALE_TR="${HIVA_RESIDUAL_SCALE_TR:-}"
HIVA_RESIDUAL_SCALE_ROT="${HIVA_RESIDUAL_SCALE_ROT:-}"
HIVA_RESIDUAL_SCALE_GRIP="${HIVA_RESIDUAL_SCALE_GRIP:-}"

POLICY_CHUNK_SIZE="${POLICY_CHUNK_SIZE:-${HIVA_FIT_HORIZON}}"
POLICY_N_ACTION_STEPS="${POLICY_N_ACTION_STEPS:-${HIVA_DMAX}}"
RUN_NAME="${RUN_NAME:-smolvla_hiva_coeff_lpmt_residual_stage2_heads_v5_d4_6_10_k${HIVA_K}_f${HIVA_FIT_HORIZON}_${HIVA_RESIDUAL_MODE}_daw${HIVA_DECODED_ACTION_LOSS_WEIGHT//./p}_b${BATCH_PER_GPU}_s${S}_${RUN_STAMP}}"
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
source "/home/jongwoopark/lerobot/server_scripts/common_hf_cache.sh"
setup_hf_datasets_cache
export PATH="${CONDA_ENV_BIN}:${PATH}"
export HIVA_TRAIN_STAGE2_ACTION_HEADS=true
export HIVA_TRAIN_RESIDUAL_ONLY=false
mkdir -p "${HF_DATASETS_CACHE}" "${OUTPUT_ROOT}"

# Standalone freeze patch. This avoids requiring a separate sitecustomize directory.
PATCH_DIR="${HIVA_STAGE2_PATCH_DIR:-/tmp/jongwoopark_hiva_stage2_sitecustomize_${RUN_STAMP}_$$}"
mkdir -p "${PATCH_DIR}"
cat > "${PATCH_DIR}/sitecustomize.py" <<'PY'
from __future__ import annotations
import os


def _truthy(value: str | None) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "y", "on"}


def _parse_csv(value: str | None) -> list[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


if _truthy(os.environ.get("HIVA_TRAIN_STAGE2_ACTION_HEADS")):
    from lerobot.policies.smolvla_hiva_coeff.modeling_smolvla_hiva_coeff import HiVACoeffSmolVLAPolicy

    _original_init = HiVACoeffSmolVLAPolicy.__init__

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        if not getattr(self.config, "hiva_residual_enabled", False):
            raise RuntimeError("HIVA_TRAIN_STAGE2_ACTION_HEADS requires hiva_residual_enabled=true.")
        if getattr(self.model, "hiva_residual_head", None) is None:
            raise RuntimeError("Stage-2 action-head training requested but model.hiva_residual_head is None.")
        trainable_substrings = _parse_csv(os.environ.get("HIVA_STAGE2_TRAINABLE_SUBSTRINGS")) or [
            "model.hiva_residual_head",
            "model.hiva_tr_out_proj",
            "model.hiva_rot_out_proj",
            "model.hiva_grip_out_proj",
            "model.hiva_duration_head",
        ]
        trainable_names = []
        frozen_names = []
        for name, param in self.named_parameters():
            trainable = any(substr in name for substr in trainable_substrings)
            param.requires_grad = trainable
            if trainable:
                trainable_names.append(name)
            else:
                frozen_names.append(name)
        trainable_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in self.parameters())
        print(
            "[HIVA stage2 action-head patch] "
            f"trainable params: {trainable_count:,} / {total_count:,}; "
            f"trainable tensors: {len(trainable_names)}; frozen tensors: {len(frozen_names)}",
            flush=True,
        )
        print("[HIVA stage2 action-head patch] trainable substrings: " + str(trainable_substrings), flush=True)
        print("[HIVA stage2 action-head patch] first trainable tensors: " + str(trainable_names[:40]), flush=True)
        if not trainable_names:
            raise RuntimeError("Stage-2 patch found zero trainable tensors. Check module names.")

    def _get_optim_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    HiVACoeffSmolVLAPolicy.__init__ = _patched_init
    HiVACoeffSmolVLAPolicy.get_optim_params = _get_optim_params
PY
export PYTHONPATH="${PATCH_DIR}:${REPO_ROOT}/src:${PYTHONPATH:-}"

if [[ -z "${INIT_HIVA_STAGE1}" ]]; then
  echo "ERROR: INIT_HIVA_STAGE1 must point to the Stage-1 residual checkpoint pretrained_model directory." >&2
  exit 2
fi
for required_path in "${DATA_ROOT}" "${SIDECAR}" "${SIDECAR_SUMMARY}" "${INIT_HIVA_STAGE1}"; do
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

OPTIONAL_ARGS=()
if [[ -n "${EVAL_TASK_IDS}" ]]; then OPTIONAL_ARGS+=(--env.task_ids="${EVAL_TASK_IDS}"); fi
if [[ -n "${HIVA_DEGREE_TR}" ]]; then OPTIONAL_ARGS+=(--policy.hiva_degree_tr="${HIVA_DEGREE_TR}"); fi
if [[ -n "${HIVA_DEGREE_ROT}" ]]; then OPTIONAL_ARGS+=(--policy.hiva_degree_rot="${HIVA_DEGREE_ROT}"); fi
if [[ -n "${HIVA_DEGREE_GRIP}" ]]; then OPTIONAL_ARGS+=(--policy.hiva_degree_grip="${HIVA_DEGREE_GRIP}"); fi
if [[ -n "${HIVA_DECODED_TR_LOSS_BETA}" ]]; then OPTIONAL_ARGS+=(--policy.hiva_decoded_tr_loss_beta="${HIVA_DECODED_TR_LOSS_BETA}"); fi
if [[ -n "${HIVA_DECODED_ROT_LOSS_BETA}" ]]; then OPTIONAL_ARGS+=(--policy.hiva_decoded_rot_loss_beta="${HIVA_DECODED_ROT_LOSS_BETA}"); fi
if [[ -n "${HIVA_DECODED_GRIP_LOSS_BETA}" ]]; then OPTIONAL_ARGS+=(--policy.hiva_decoded_grip_loss_beta="${HIVA_DECODED_GRIP_LOSS_BETA}"); fi
if [[ -n "${HIVA_RESIDUAL_SCALE_TR}" ]]; then OPTIONAL_ARGS+=(--policy.hiva_residual_scale_tr="${HIVA_RESIDUAL_SCALE_TR}"); fi
if [[ -n "${HIVA_RESIDUAL_SCALE_ROT}" ]]; then OPTIONAL_ARGS+=(--policy.hiva_residual_scale_rot="${HIVA_RESIDUAL_SCALE_ROT}"); fi
if [[ -n "${HIVA_RESIDUAL_SCALE_GRIP}" ]]; then OPTIONAL_ARGS+=(--policy.hiva_residual_scale_grip="${HIVA_RESIDUAL_SCALE_GRIP}"); fi

WANDB_ARGS=()
if [[ "${WANDB_ENABLE}" == "true" ]]; then
  WANDB_ARGS+=(--wandb.enable=true --wandb.project="${WANDB_PROJECT}" --wandb.disable_artifact="${WANDB_DISABLE_ARTIFACT}" --wandb.mode="${WANDB_MODE}")
else
  WANDB_ARGS+=(--wandb.enable=false)
fi

TRAIN_ARGS=(
  --policy.type=smolvla_hiva_coeff
  --policy.push_to_hub=false
  --policy.load_vlm_weights=false
  --policy.init_smolvla_checkpoint_path="${INIT_HIVA_STAGE1}"
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
  --policy.hiva_residual_enabled="${HIVA_RESIDUAL_ENABLED}"
  --policy.hiva_residual_mode="${HIVA_RESIDUAL_MODE}"
  --policy.hiva_residual_horizon="${HIVA_RESIDUAL_HORIZON}"
  --policy.hiva_residual_ffn_hidden_mult="${HIVA_RESIDUAL_FFN_HIDDEN_MULT}"
  --policy.hiva_residual_token_time_hidden_mult="${HIVA_RESIDUAL_TOKEN_TIME_HIDDEN_MULT}"
  --policy.hiva_residual_alpha_init="${HIVA_RESIDUAL_ALPHA_INIT}"
  --policy.hiva_residual_zero_init="${HIVA_RESIDUAL_ZERO_INIT}"
  --policy.hiva_residual_num_blocks="${HIVA_RESIDUAL_NUM_BLOCKS}"
  --policy.hiva_residual_cross_attn_heads="${HIVA_RESIDUAL_CROSS_ATTN_HEADS}"
  --policy.hiva_residual_attn_dropout="${HIVA_RESIDUAL_ATTN_DROPOUT}"
  --policy.hiva_residual_scale_mult="${HIVA_RESIDUAL_SCALE_MULT}"
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
Stage 2 residual/action-head HiVA co-adaptation finetune
Host: $(hostname)
RUN_NAME=${RUN_NAME}
OUTPUT_DIR=${OUTPUT_DIR}
LOG_FILE=${LOG_FILE}
INIT_HIVA_STAGE1=${INIT_HIVA_STAGE1}
DATA_ROOT=${DATA_ROOT}
SIDECAR=${SIDECAR}
SIDECAR_SUMMARY=${SIDECAR_SUMMARY}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
NUM_PROCESSES=${NUM_PROCESSES}
BATCH_PER_GPU=${BATCH_PER_GPU}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}
STEPS=${STEPS}
SAVE_FREQ=${SAVE_FREQ}
HIVA_DECODED_ACTION_LOSS_WEIGHT=${HIVA_DECODED_ACTION_LOSS_WEIGHT}
HIVA_TR_LOSS_WEIGHT=${HIVA_TR_LOSS_WEIGHT}
HIVA_ROT_LOSS_WEIGHT=${HIVA_ROT_LOSS_WEIGHT}
HIVA_GRIP_LOSS_WEIGHT=${HIVA_GRIP_LOSS_WEIGHT}
HIVA_DURATION_NOISY_LOSS_WEIGHT=${HIVA_DURATION_NOISY_LOSS_WEIGHT}
HIVA_RESIDUAL_MODE=${HIVA_RESIDUAL_MODE}
HIVA_RESIDUAL_SCALE_MULT=${HIVA_RESIDUAL_SCALE_MULT}
HIVA_TRAIN_STAGE2_ACTION_HEADS=${HIVA_TRAIN_STAGE2_ACTION_HEADS}
Patch dir=${PATCH_DIR}
INFO

"${ACCELERATE_BIN}" launch \
  --num_processes="${NUM_PROCESSES}" \
  --mixed_precision=bf16 \
  "${LEROBOT_TRAIN_BIN}" \
  "${TRAIN_ARGS[@]}"
