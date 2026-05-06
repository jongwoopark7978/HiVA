#!/usr/bin/env bash
set -euo pipefail

# Sequential cleaner-suffix coefficient HiVA sweep on bigflow GPUs 4-7.
#
# Job1:
#   S=4, lambda_duration in {1.0, 0.5, 0.1}
# Job2:
#   lambda_duration=1.0, S in {2, 1}
#
# Defaults:
#   GPU_IDS=4,5,6,7
#   NUM_GPUS=4
#   BATCH_PER_GPU=160
#   WANDB_ENABLE=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

BASE_SCRIPT="${SCRIPT_DIR}/finetune_bigflow_ckpt_20k_hiva_coeff_cleaner_suffix.sh"
SWEEP_ID="${SWEEP_ID:-$(date +%Y%m%d_%H%M%S)}"

export GPU_IDS="${GPU_IDS:-4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-4}"
export NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
export BATCH_PER_GPU="${BATCH_PER_GPU:-160}"
export EVAL_FREQ="${EVAL_FREQ:-0}"
export WANDB_ENABLE="${WANDB_ENABLE:-true}"
export WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"
export HIVA_DURATION_LOSS="${HIVA_DURATION_LOSS:-ce_mean}"
export HIVA_SUFFIX_ATTENTION="${HIVA_SUFFIX_ATTENTION:-duration_prefix}"
export HIVA_DURATION_NOISY_SIGMA="${HIVA_DURATION_NOISY_SIGMA:-0.25}"

echo "Starting bigflow cleaner-suffix coefficient HiVA sweep ${SWEEP_ID}"
echo "GPU_IDS=${GPU_IDS}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "BATCH_PER_GPU=${BATCH_PER_GPU}"
echo "WANDB_ENABLE=${WANDB_ENABLE}"
echo "HIVA_DURATION_LOSS=${HIVA_DURATION_LOSS}"
echo "HIVA_SUFFIX_ATTENTION=${HIVA_SUFFIX_ATTENTION}"

run_one() {
  local s="$1"
  local lambda_duration="$2"
  local phase="$3"
  local lambda_label="${lambda_duration/./p}"
  local run_stamp
  run_stamp="$(date +%Y%m%d_%H%M%S_%N)_pid$$"

  export S="${s}"
  export HIVA_DURATION_NOISY_LOSS_WEIGHT="${lambda_duration}"
  export RUN_ID="${run_stamp}"
  export RUN_NAME="smolvla_hiva_coeff_cleaner_suffix_bigflow_${phase}_lambda${lambda_label}_b${BATCH_PER_GPU}_s${S}_${SWEEP_ID}_${run_stamp}"
  export WANDB_NOTES="cleaner-suffix coefficient HiVA; phase=${phase}; lambda_duration=${lambda_duration}; S=${S}; batch_per_gpu=${BATCH_PER_GPU}; num_gpus=${NUM_GPUS}; suffix_attention=${HIVA_SUFFIX_ATTENTION}; duration_loss=${HIVA_DURATION_LOSS}"

  echo
  echo "======================================================================"
  echo "Launching ${RUN_NAME}"
  echo "S=${S}"
  echo "HIVA_DURATION_NOISY_LOSS_WEIGHT=${HIVA_DURATION_NOISY_LOSS_WEIGHT}"
  echo "======================================================================"
  bash "${BASE_SCRIPT}"
}

# Job1: lambda sweep at S=4.
run_one 4 1.0 "job1"
run_one 4 0.5 "job1"
run_one 4 0.1 "job1"

# Job2: step-reduction sweep at lambda=1.0.
run_one 2 1.0 "job2"
run_one 1 1.0 "job2"

echo "Completed bigflow cleaner-suffix coefficient HiVA sweep ${SWEEP_ID}"
