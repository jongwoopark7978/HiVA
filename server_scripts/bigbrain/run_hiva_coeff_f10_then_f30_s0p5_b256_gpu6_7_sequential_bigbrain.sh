#!/usr/bin/env bash
set -euo pipefail

# Sequential preview-length ablation on BigBrain GPUs 6,7.
#
# 1) MT HiVA case:    K=10, F=10, basis=canonical_mt
# 2) LP-MT HiVA case: K=10, F=30, basis=canonical_lp_mt
#
# Both use S=0.5, batch per GPU=256, total steps=5000, and save
# checkpoints at [3125,3500,3750,5000].

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"

GPU_IDS="${GPU_IDS:-6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
BATCH_PER_GPU="${BATCH_PER_GPU:-256}"
TOTAL_STEPS="${TOTAL_STEPS:-5000}"
SAVE_STEPS="${SAVE_STEPS:-[3125,3500,3750,5000]}"

echo "Sequential F10 -> F30 preview ablation"
echo "RUN_STAMP=${RUN_STAMP}"
echo "GPU_IDS=${GPU_IDS}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "BATCH_PER_GPU=${BATCH_PER_GPU}"
echo "TOTAL_STEPS=${TOTAL_STEPS}"
echo "SAVE_STEPS=${SAVE_STEPS}"

echo
echo "=============================="
echo "Starting case1: MT K=10 F=10"
echo "=============================="
RUN_STAMP="${RUN_STAMP}_f10" \
GPU_IDS="${GPU_IDS}" \
NUM_PROCESSES="${NUM_PROCESSES}" \
BATCH_PER_GPU="${BATCH_PER_GPU}" \
BATCH_SIZE="${BATCH_PER_GPU}" \
TOTAL_STEPS="${TOTAL_STEPS}" \
SAVE_STEPS="${SAVE_STEPS}" \
S="0.5" \
bash "${SCRIPT_DIR}/run_hiva_coeff_lpmt_stage0_v5_k10_f10_s0p25_bigbrain.sh"

echo
echo "==================================="
echo "Starting case2: LP-MT K=10 F=30"
echo "==================================="
RUN_STAMP="${RUN_STAMP}_f30" \
GPU_IDS="${GPU_IDS}" \
NUM_PROCESSES="${NUM_PROCESSES}" \
BATCH_PER_GPU="${BATCH_PER_GPU}" \
BATCH_SIZE="${BATCH_PER_GPU}" \
TOTAL_STEPS="${TOTAL_STEPS}" \
SAVE_STEPS="${SAVE_STEPS}" \
S="0.5" \
bash "${SCRIPT_DIR}/run_hiva_coeff_lpmt_stage0_v5_k10_f30_s0p25_bigbrain.sh"

echo "Sequential F10 -> F30 preview ablation finished at $(date)"
