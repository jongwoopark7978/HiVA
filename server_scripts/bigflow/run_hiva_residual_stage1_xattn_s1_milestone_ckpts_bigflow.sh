#!/usr/bin/env bash
set -euo pipefail

# One-shot stage-1 xattn run on bigflow with S=1 and explicit milestone
# checkpoints at 0.5x, 0.625x, 0.75x, 0.875x, and 1.0x of the total steps.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"

GPU_IDS="${GPU_IDS:-5,6,7}"
NUM_GPUS="${NUM_GPUS:-3}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-1024}"
BASE_STEPS="${BASE_STEPS:-8000}"
BASE_NUM_GPUS="${BASE_NUM_GPUS:-1}"
BASE_BATCH_PER_GPU="${BASE_BATCH_PER_GPU:-64}"
S="${S:-1}"

GLOBAL_BATCH_SIZE=$((NUM_GPUS * BATCH_PER_GPU))
BASE_GLOBAL_BATCH_SIZE=$((BASE_NUM_GPUS * BASE_BATCH_PER_GPU))
STEPS="${STEPS:-$("${CONDA_ENV_BIN}/python" - <<PY
import math
steps = math.ceil(int("${BASE_STEPS}") * int("${BASE_GLOBAL_BATCH_SIZE}") / int("${GLOBAL_BATCH_SIZE}") / float("${S}"))
print(max(1, steps))
PY
)}"
SAVE_STEPS="${SAVE_STEPS:-$("${CONDA_ENV_BIN}/python" - <<PY
import math
steps = int("${STEPS}")
fractions = (0.5, 0.625, 0.75, 0.875, 1.0)
milestones = sorted({min(steps, max(1, math.ceil(steps * f))) for f in fractions})
print("[" + ",".join(str(s) for s in milestones) + "]")
PY
)}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-smolvla_hiva_coeff_lpmt_stage1_xattn_s1_milestone_v5_d4_6_10_tr3_rot3_grip0_daw1_b${BATCH_PER_GPU}_g${NUM_GPUS}_s${S}_${RUN_STAMP}}"

export GPU_IDS
export NUM_GPUS
export NUM_PROCESSES
export BATCH_PER_GPU
export BATCH_SIZE="${BATCH_SIZE:-${BATCH_PER_GPU}}"
export BASE_STEPS
export BASE_NUM_GPUS
export BASE_BATCH_PER_GPU
export S
export STEPS
export SAVE_FREQ="${SAVE_FREQ:-0}"
export SAVE_STEPS
export RUN_STAMP
export RUN_NAME
export WANDB_ENABLE="${WANDB_ENABLE:-true}"

export HIVA_RESIDUAL_SCALE_TR="${HIVA_RESIDUAL_SCALE_TR:-3.0}"
export HIVA_RESIDUAL_SCALE_ROT="${HIVA_RESIDUAL_SCALE_ROT:-3.0}"
export HIVA_RESIDUAL_SCALE_GRIP="${HIVA_RESIDUAL_SCALE_GRIP:-0.0}"
export HIVA_DECODED_ACTION_LOSS_WEIGHT="${HIVA_DECODED_ACTION_LOSS_WEIGHT:-1.0}"
export HIVA_DECODED_TR_LOSS_BETA="${HIVA_DECODED_TR_LOSS_BETA:-0.1}"
export HIVA_DECODED_ROT_LOSS_BETA="${HIVA_DECODED_ROT_LOSS_BETA:-0.05}"
export HIVA_DECODED_GRIP_LOSS_BETA="${HIVA_DECODED_GRIP_LOSS_BETA:-0.1}"

echo "Launching stage-1 S=${S} run with STEPS=${STEPS}, SAVE_STEPS=${SAVE_STEPS}, GPU_IDS=${GPU_IDS}"
exec bash "${SCRIPT_DIR}/run_hiva_residual_stage1_xattn_bigflow.sh"
