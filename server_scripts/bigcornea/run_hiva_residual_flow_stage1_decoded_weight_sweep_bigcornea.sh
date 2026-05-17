#!/usr/bin/env bash
set -euo pipefail

# Stage-1 residual-flow expert decoded-loss-weight sweep on bigcornea.
# Runs one job at a time on the requested GPUs and writes milestone checkpoints.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-128}"
BATCH_SIZE="${BATCH_SIZE:-${BATCH_PER_GPU}}"
S="${S:-2}"
BASE_STEPS="${BASE_STEPS:-20000}"
BASE_NUM_GPUS="${BASE_NUM_GPUS:-1}"
BASE_BATCH_PER_GPU="${BASE_BATCH_PER_GPU:-64}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"

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
fractions = (0.1, 0.25, 0.5, 0.75, 1.0)
milestones = sorted({min(steps, max(1, math.ceil(steps * f))) for f in fractions})
print("[" + ",".join(str(s) for s in milestones) + "]")
PY
)}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-$("${CONDA_ENV_BIN}/python" - <<PY
import math
print(max(1, math.ceil(int("${STEPS}") * float("${WARMUP_RATIO}"))))
PY
)}"

QUEUE_STAMP="${QUEUE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
DECODED_WEIGHTS_RAW="${HIVA_RESIDUAL_FLOW_DECODED_WEIGHT_SWEEP:-1.0 2.0 0.5}"

echo "HiVA residual-flow stage-1 decoded-loss-weight sweep"
echo "QUEUE_STAMP=${QUEUE_STAMP}"
echo "GPU_IDS=${GPU_IDS} NUM_GPUS=${NUM_GPUS} BATCH_PER_GPU=${BATCH_PER_GPU} S=${S}"
echo "BASE_STEPS=${BASE_STEPS} BASE_GLOBAL_BATCH_SIZE=${BASE_GLOBAL_BATCH_SIZE} GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "STEPS=${STEPS} SAVE_STEPS=${SAVE_STEPS} SCHEDULER_WARMUP_STEPS=${SCHEDULER_WARMUP_STEPS}"
echo "DECODED_WEIGHTS=${DECODED_WEIGHTS_RAW}"
echo "RESIDUAL_FLOW_SCALES tr=${HIVA_RESIDUAL_FLOW_SCALE_TR:-3.0} rot=${HIVA_RESIDUAL_FLOW_SCALE_ROT:-3.0} grip=${HIVA_RESIDUAL_FLOW_SCALE_GRIP:-0.5}"

run_one() {
  local decoded_weight="$1"
  local decoded_label="${decoded_weight//./p}"
  local run_stamp="${QUEUE_STAMP}_daw${decoded_label}"
  local run_name="smolvla_hiva_coeff_lpmt_residual_flow_stage1_v5_d4_6_10_k10_f15_tr3p0_rot3p0_grip0p5_daw${decoded_label}_b${BATCH_PER_GPU}_g${NUM_GPUS}_s${S}_${QUEUE_STAMP}"

  echo
  echo "===== Running residual-flow decoded loss weight ${decoded_weight} at $(date) ====="
  RUN_STAMP="${run_stamp}" \
  RUN_NAME="${run_name}" \
  GPU_IDS="${GPU_IDS}" \
  NUM_GPUS="${NUM_GPUS}" \
  NUM_PROCESSES="${NUM_PROCESSES}" \
  BATCH_PER_GPU="${BATCH_PER_GPU}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  STEPS="${STEPS}" \
  SAVE_FREQ=0 \
  SAVE_STEPS="${SAVE_STEPS}" \
  SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS}" \
  SCHEDULER_DECAY_STEPS="${STEPS}" \
  WANDB_ENABLE="${WANDB_ENABLE:-true}" \
  HIVA_RESIDUAL_FLOW_SCALE_TR="${HIVA_RESIDUAL_FLOW_SCALE_TR:-3.0}" \
  HIVA_RESIDUAL_FLOW_SCALE_ROT="${HIVA_RESIDUAL_FLOW_SCALE_ROT:-3.0}" \
  HIVA_RESIDUAL_FLOW_SCALE_GRIP="${HIVA_RESIDUAL_FLOW_SCALE_GRIP:-0.5}" \
  HIVA_RESIDUAL_FLOW_DECODED_LOSS_WEIGHT="${decoded_weight}" \
  HIVA_RESIDUAL_FLOW_USE_SEPARATE_EXPERT=true \
  bash "${SCRIPT_DIR}/run_hiva_residual_flow_stage1_bigcornea.sh"
}

for decoded_weight in ${DECODED_WEIGHTS_RAW}; do
  run_one "${decoded_weight}"
done

echo "All residual-flow decoded-loss-weight sweep jobs completed."
