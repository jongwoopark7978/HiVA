#!/usr/bin/env bash
set -euo pipefail

# Requested cleaner-suffix coefficient HiVA sweep on bigcornea.
#
# Shared settings:
#   - all 8 GPUs
#   - BATCH_PER_GPU=80
#   - HIVA_DURATION_NOISY_LOSS_WEIGHT=1.0
#   - HIVA_DURATION_LOSS=duration_noisy_weights
#
# J1: S=4, sigma in {0.20, 0.30}
# J2: sigma=0.25, S in {2, 1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/finetune_bigcornea_ckpt_20k_hiva_coeff_cleaner_suffix.sh"
SWEEP_ID="${SWEEP_ID:-$(date +%Y%m%d_%H%M%S)}"

export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
export BATCH_PER_GPU="${BATCH_PER_GPU:-80}"
export EVAL_FREQ="${EVAL_FREQ:-0}"
export WANDB_ENABLE="${WANDB_ENABLE:-true}"
export WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"
export HIVA_DURATION_LOSS="${HIVA_DURATION_LOSS:-duration_noisy_weights}"
export HIVA_DURATION_NOISY_LOSS_WEIGHT="${HIVA_DURATION_NOISY_LOSS_WEIGHT:-1.0}"

run_one() {
  local job="$1"
  local sigma="$2"
  local s="$3"

  local sigma_label="${sigma/./p}"
  local run_id="${job}_${SWEEP_ID}_sigma${sigma_label}_s${s}"

  export RUN_ID="${run_id}"
  export RUN_NAME="smolvla_hiva_coeff_cleaner_suffix_bigcornea_${job}_sigma${sigma_label}_w1p0_b${BATCH_PER_GPU}_s${s}_${SWEEP_ID}"
  export HIVA_DURATION_NOISY_SIGMA="${sigma}"
  export S="${s}"
  export WANDB_NOTES="${job}: cleaner-suffix coefficient HiVA; sigma=${sigma}; duration_noisy_loss_weight=${HIVA_DURATION_NOISY_LOSS_WEIGHT}; S=${S}; batch_per_gpu=${BATCH_PER_GPU}; num_gpus=${NUM_GPUS}"

  echo
  echo "======================================================================"
  echo "Launching ${RUN_NAME}"
  echo "JOB=${job}"
  echo "HIVA_DURATION_NOISY_SIGMA=${HIVA_DURATION_NOISY_SIGMA}"
  echo "HIVA_DURATION_NOISY_LOSS_WEIGHT=${HIVA_DURATION_NOISY_LOSS_WEIGHT}"
  echo "S=${S}"
  echo "BATCH_PER_GPU=${BATCH_PER_GPU}"
  echo "======================================================================"

  bash "${BASE_SCRIPT}"
}

echo "Starting requested cleaner-suffix coefficient HiVA J1/J2 sweep ${SWEEP_ID}"
echo "BASE_SCRIPT=${BASE_SCRIPT}"
echo "GPU_IDS=${GPU_IDS}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "BATCH_PER_GPU=${BATCH_PER_GPU}"
echo "HIVA_DURATION_NOISY_LOSS_WEIGHT=${HIVA_DURATION_NOISY_LOSS_WEIGHT}"
echo "HIVA_DURATION_LOSS=${HIVA_DURATION_LOSS}"
echo "WANDB_ENABLE=${WANDB_ENABLE}"

run_one "J1" "0.20" "4"
run_one "J1" "0.30" "4"
run_one "J2" "0.25" "2"
run_one "J2" "0.25" "1"

echo "Completed requested cleaner-suffix coefficient HiVA J1/J2 sweep ${SWEEP_ID}"
