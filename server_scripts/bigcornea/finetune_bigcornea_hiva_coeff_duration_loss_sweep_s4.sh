#!/usr/bin/env bash
set -euo pipefail

# Six-run cleaner-suffix coefficient HiVA duration-loss sweep on bigcornea.
# Runs sequentially on all 8 GPUs:
#   BATCH_PER_GPU=80
#   sigma in {0.25, 0.20}
#   hiva duration noisy loss weight in {0.1, 0.5, 1.0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BASE_SCRIPT="${SCRIPT_DIR}/finetune_bigcornea_ckpt_20k_hiva_coeff_cleaner_suffix_b64.sh"
SWEEP_ID="${SWEEP_ID:-$(date +%Y%m%d_%H%M%S)}"

export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
export BATCH_PER_GPU="${BATCH_PER_GPU:-80}"
export S="${S:-4}"
export EVAL_FREQ="${EVAL_FREQ:-0}"
export WANDB_ENABLE="${WANDB_ENABLE:-true}"
export WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"
export HIVA_DURATION_LOSS="${HIVA_DURATION_LOSS:-duration_noisy_weights}"

sigmas=(0.25 0.20)
weights=(0.1 0.5 1.0)

echo "Starting cleaner-suffix coefficient HiVA duration-loss sweep ${SWEEP_ID}"
echo "GPU_IDS=${GPU_IDS}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "BATCH_PER_GPU=${BATCH_PER_GPU}"
echo "S=${S}"
echo "WANDB_ENABLE=${WANDB_ENABLE}"
echo "HIVA_DURATION_LOSS=${HIVA_DURATION_LOSS}"

for sigma in "${sigmas[@]}"; do
  for weight in "${weights[@]}"; do
    sigma_label="${sigma/./p}"
    weight_label="${weight/./p}"
    run_id="sweep_${SWEEP_ID}_sigma${sigma_label}_w${weight_label}"

    export RUN_ID="${run_id}"
    export RUN_NAME="smolvla_hiva_coeff_cleaner_suffix_bigcornea_sigma${sigma_label}_w${weight_label}_b${BATCH_PER_GPU}_s${S}_${SWEEP_ID}"
    export HIVA_DURATION_NOISY_SIGMA="${sigma}"
    export HIVA_DURATION_NOISY_LOSS_WEIGHT="${weight}"
    export WANDB_NOTES="cleaner-suffix coefficient HiVA duration loss sweep; sigma=${sigma}; duration_noisy_loss_weight=${weight}; S=${S}; batch_per_gpu=${BATCH_PER_GPU}; num_gpus=${NUM_GPUS}"

    echo
    echo "======================================================================"
    echo "Launching ${RUN_NAME}"
    echo "HIVA_DURATION_NOISY_SIGMA=${HIVA_DURATION_NOISY_SIGMA}"
    echo "HIVA_DURATION_NOISY_LOSS_WEIGHT=${HIVA_DURATION_NOISY_LOSS_WEIGHT}"
    echo "======================================================================"

    bash "${BASE_SCRIPT}"
  done
done

echo "Completed cleaner-suffix coefficient HiVA duration-loss sweep ${SWEEP_ID}"
