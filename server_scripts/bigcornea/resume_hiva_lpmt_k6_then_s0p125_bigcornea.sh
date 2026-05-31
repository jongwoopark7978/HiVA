#!/usr/bin/env bash
set -euo pipefail

# Resume the interrupted K=6 stage-0 LP-MT HiVA run from its latest checkpoint,
# then run the queued S=0.125 K=10 milestone job. Intended to be launched inside tmux.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
export BATCH_PER_GPU="${BATCH_PER_GPU:-64}"
export WANDB_ENABLE="${WANDB_ENABLE:-true}"
export WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"

K6_RUN_STAMP="20260518_153652"
K6_RUN_NAME="smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k6_f15_bigcornea_b64_s0p5_${K6_RUN_STAMP}"
K6_OUTPUT_DIR="${REPO_ROOT}/outputs/train/${K6_RUN_NAME}"
K6_CONFIG_PATH="${K6_OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"

if [[ ! -d "${K6_OUTPUT_DIR}/checkpoints/last" && ! -L "${K6_OUTPUT_DIR}/checkpoints/last" ]]; then
  echo "ERROR: cannot resume K=6 run; missing checkpoint link: ${K6_OUTPUT_DIR}/checkpoints/last" >&2
  exit 2
fi
if [[ ! -f "${K6_CONFIG_PATH}" ]]; then
  echo "ERROR: cannot resume K=6 run; missing train config: ${K6_CONFIG_PATH}" >&2
  exit 2
fi

echo "======================================================================"
echo "Step 1/2: resume K=6 stage-0 LP-MT HiVA"
echo "RUN_NAME=${K6_RUN_NAME}"
echo "OUTPUT_DIR=${K6_OUTPUT_DIR}"
echo "CHECKPOINT=$(readlink -f "${K6_OUTPUT_DIR}/checkpoints/last")"
echo "CONFIG_PATH=${K6_CONFIG_PATH}"
echo "======================================================================"

K_VALUES=6 \
RUN_STAMP="${K6_RUN_STAMP}" \
RUN_NAME="${K6_RUN_NAME}" \
OUTPUT_DIR="${K6_OUTPUT_DIR}" \
RESUME=true \
CONFIG_PATH="${K6_CONFIG_PATH}" \
S=0.5 \
BASE_STEPS=20000 \
SAVE_STEPS="[3000,3125,3500,3750,4375,5000]" \
bash "${SCRIPT_DIR}/run_hiva_coeff_lpmt_stage0_v5_k6_k12_s0p5_bigcornea.sh"

echo "======================================================================"
echo "Step 2/2: run S=0.125 K=10 milestone stage-0 LP-MT HiVA"
echo "======================================================================"

bash "${SCRIPT_DIR}/run_hiva_coeff_lpmt_stage0_v5_s0p125_milestone_bigcornea.sh"
