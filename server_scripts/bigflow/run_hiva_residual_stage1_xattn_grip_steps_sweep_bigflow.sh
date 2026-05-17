#!/usr/bin/env bash
set -euo pipefail

# Queue wrapper for LP-MT HiVA stage-1 xattn residual finetuning on bigflow.
# It uses the standalone bigflow stage-1 script, local bigflow dataset/sidecar,
# and runs the requested sweeps sequentially on GPUs 5,6,7.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

RUN_STAMP_ROOT="${RUN_STAMP_ROOT:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"
OUTER_LOG="${OUTER_LOG:-${LOG_DIR}/hiva_stage1_xattn_grip_steps_sweep_bigflow_${RUN_STAMP_ROOT}.outer.log}"
exec > >(tee -a "${OUTER_LOG}") 2>&1

STAGE1_SCRIPT="${STAGE1_SCRIPT:-${SCRIPT_DIR}/run_hiva_residual_stage1_xattn_bigflow.sh}"

GPU_IDS="${GPU_IDS:-5,6,7}"
NUM_GPUS="${NUM_GPUS:-3}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-1024}"
BATCH_SIZE="${BATCH_SIZE:-${BATCH_PER_GPU}}"

COMMON_ENV=(
  GPU_IDS="${GPU_IDS}"
  NUM_GPUS="${NUM_GPUS}"
  NUM_PROCESSES="${NUM_PROCESSES}"
  BATCH_PER_GPU="${BATCH_PER_GPU}"
  BATCH_SIZE="${BATCH_SIZE}"
  HIVA_RESIDUAL_SCALE_TR=3.0
  HIVA_RESIDUAL_SCALE_ROT=3.0
  HIVA_DECODED_ACTION_LOSS_WEIGHT=1.0
  HIVA_DECODED_TR_LOSS_BETA=0.1
  HIVA_DECODED_ROT_LOSS_BETA=0.05
  HIVA_DECODED_GRIP_LOSS_BETA=0.1
)

fmt_token() {
  local value="$1"
  value="${value//./p}"
  value="${value//-_/m}"
  printf '%s' "${value}"
}

run_one() {
  local job="$1"
  local grip="$2"
  local s="$3"
  local stamp
  local grip_tag
  local s_tag

  stamp="$(date +%Y%m%d_%H%M%S)"
  grip_tag="$(fmt_token "${grip}")"
  s_tag="$(fmt_token "${s}")"

  echo "===== ${job}: grip=${grip}, S=${s}, batch/gpu=${BATCH_PER_GPU}, GPUs=${GPU_IDS}, stamp=${stamp} ====="

  env "${COMMON_ENV[@]}" \
    S="${s}" \
    HIVA_RESIDUAL_SCALE_GRIP="${grip}" \
    RUN_STAMP="${stamp}" \
    RUN_NAME="smolvla_hiva_coeff_lpmt_stage1_xattn_${job}_v5_d4_6_10_tr3_rot3_grip${grip_tag}_daw1_betas_0p1_0p05_0p1_b${BATCH_PER_GPU}_g${NUM_GPUS}_s${s_tag}_${stamp}" \
    bash "${STAGE1_SCRIPT}"

  echo "===== Completed ${job}: grip=${grip}, S=${s} ====="
}

echo "Stage-1 xattn sweep started"
echo "Outer log: ${OUTER_LOG}"
echo "Stage1 script: ${STAGE1_SCRIPT}"
echo "GPU_IDS=${GPU_IDS}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "BATCH_PER_GPU=${BATCH_PER_GPU}"

# Job 1: grip residual-scale sweep with the default step-reduction S=2.
run_one job1_gripsweep 0.0 2
run_one job1_gripsweep 0.3 2
run_one job1_gripsweep 0.5 2

# Job 2: longer-step sweep with grip residual disabled.
run_one job2_stepsweep 0.0 1
run_one job2_stepsweep 0.0 0.5
run_one job2_stepsweep 0.0 0.25

echo "Stage-1 xattn sweep finished"
