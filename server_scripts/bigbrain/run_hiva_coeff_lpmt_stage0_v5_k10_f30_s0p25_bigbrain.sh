#!/usr/bin/env bash
set -euo pipefail

# BigBrain stage-0 LP-MT HiVA coefficient finetune for preview-length ablation.
# Case 2: K=10, F=30, D={4,6,10}. This keeps the same executable durations
# but fits a longer LP-MT preview horizon.
#
# Defaults mirror the S=0.5 k10_f15 b256 setup:
#   batch per GPU: 256
#   total steps: 5000
#   save steps: 3125,3500,3750,5000
#
# Override GPU_IDS, NUM_PROCESSES, BATCH_PER_GPU, TOTAL_STEPS, SAVE_STEPS, etc.
# if needed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigbrain/add_disk0/jongwoopark}"
HIVA_K="${HIVA_K:-10}"
HIVA_FIT_HORIZON="${HIVA_FIT_HORIZON:-30}"
HIVA_DEGREE="${HIVA_DEGREE:-3}"
S="${S:-0.5}"
TOTAL_STEPS="${TOTAL_STEPS:-5000}"
BATCH_PER_GPU="${BATCH_PER_GPU:-256}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
GPU_IDS="${GPU_IDS:-6,7}"
SAVE_STEPS="${SAVE_STEPS:-[3125,3500,3750,5000]}"

RUN_NAME="${RUN_NAME:-smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f30_bigbrain_b${BATCH_PER_GPU}_g${NUM_PROCESSES}_s0p5_steps${TOTAL_STEPS}_${RUN_STAMP}}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/hiva_coeff_lpmt_stage0_v5_k10_f30_s0p25_bigbrain_${RUN_STAMP}.log}"

SIDECAR="${SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f30_canonical_lp_mt.parquet}"
SIDECAR_SUMMARY="${SIDECAR_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f30_canonical_lp_mt.summary.json}"

echo "Launching LP-MT preview ablation case2: K=10 F=30 longer-preview"
echo "RUN_NAME=${RUN_NAME}"
echo "GPU_IDS=${GPU_IDS}"
echo "BATCH_PER_GPU=${BATCH_PER_GPU}"
echo "TOTAL_STEPS=${TOTAL_STEPS}"
echo "SAVE_STEPS=${SAVE_STEPS}"
echo "SIDECAR=${SIDECAR}"
echo "SIDECAR_SUMMARY=${SIDECAR_SUMMARY}"
echo "QUEUE_LOG=${QUEUE_LOG}"

if [[ ! -e "${SIDECAR}" || ! -e "${SIDECAR_SUMMARY}" ]]; then
  echo "ERROR: required K=10 F=30 sidecar files are missing." >&2
  echo "  ${SIDECAR}" >&2
  echo "  ${SIDECAR_SUMMARY}" >&2
  exit 2
fi

RUN_STAMP="${RUN_STAMP}" \
RUN_NAME="${RUN_NAME}" \
QUEUE_LOG="${QUEUE_LOG}" \
GPU_IDS="${GPU_IDS}" \
NUM_PROCESSES="${NUM_PROCESSES}" \
BATCH_PER_GPU="${BATCH_PER_GPU}" \
BATCH_SIZE="${BATCH_PER_GPU}" \
S="${S}" \
TOTAL_STEPS="${TOTAL_STEPS}" \
SAVE_STEPS="${SAVE_STEPS}" \
HIVA_K="${HIVA_K}" \
HIVA_DEGREE="${HIVA_DEGREE}" \
HIVA_FIT_HORIZON="${HIVA_FIT_HORIZON}" \
HIVA_DMAX="10" \
POLICY_CHUNK_SIZE="${POLICY_CHUNK_SIZE:-${HIVA_FIT_HORIZON}}" \
SIDECAR_ROOT="${SIDECAR_ROOT}" \
SIDECAR="${SIDECAR}" \
SIDECAR_SUMMARY="${SIDECAR_SUMMARY}" \
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}" \
DATA_REPO_ID="${DATA_REPO_ID:-local/libero_lerobot_v3_lerobotkeys}" \
WANDB_ENABLE="${WANDB_ENABLE:-true}" \
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}" \
WANDB_MODE="${WANDB_MODE:-online}" \
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}" \
WANDB_NOTES="${WANDB_NOTES:-LP-MT HiVA preview ablation case2 K=10 F=30 longer-preview; D=4,6,10; S=0.5; batch_per_gpu=${BATCH_PER_GPU}; GPUs=${GPU_IDS}; steps=${TOTAL_STEPS}; save_steps=${SAVE_STEPS}}" \
bash "${SCRIPT_DIR}/run_hiva_coeff_lpmt_stage0_p5_s0p5_5000_bigbrain.sh" --total-steps "${TOTAL_STEPS}"
