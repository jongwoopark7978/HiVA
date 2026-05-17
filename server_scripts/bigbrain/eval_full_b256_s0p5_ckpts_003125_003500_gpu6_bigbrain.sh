#!/usr/bin/env bash
set -euo pipefail

# Full 50-episode LIBERO eval for selected b256 S=0.5 stage0 LP-MT HiVA
# checkpoints on BigBrain GPU 6.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_ID="${GPU_ID:-6}"
RUNNER="${RUNNER:-${SCRIPT_DIR}/eval_lpmt_stage0_v5_all_ckpts_partial_gpu6_7_bs10_bigbrain.sh}"

TRAIN_DIR="${TRAIN_DIR:-/nfs/bigcornea.cs.stonybrook.edu/add_disk3/jongwoopark/HiVA_train/finetuning_stage0/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigbrain_b256_g2_s0p5_steps5000_20260515_233203}"
MODEL_NAME="$(basename "${TRAIN_DIR}")"
CKPTS_OVERRIDE="${CKPTS_OVERRIDE:-003125 003250 003375 003500}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigbrain/add_disk0/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"

LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"

SWEEP_OUTPUT_DIR="${SWEEP_OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/full_bigbrain_gpu${GPU_ID}_${MODEL_NAME}_ckpts_003125_003250_003375_003500_50eps_bs50_${TIMESTAMP}}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/eval_full_gpu${GPU_ID}_${MODEL_NAME}_ckpts_003125_003500_50eps_bs50_${TIMESTAMP}.queue.log}"

echo "Launching full b256 S=0.5 eval on GPU ${GPU_ID}"
echo "TRAIN_DIR=${TRAIN_DIR}"
echo "CKPTS_OVERRIDE=${CKPTS_OVERRIDE}"
echo "SWEEP_OUTPUT_DIR=${SWEEP_OUTPUT_DIR}"
echo "QUEUE_LOG=${QUEUE_LOG}"

TRAIN_DIR="${TRAIN_DIR}" \
MODEL_TAG="${MODEL_NAME}" \
CKPTS_OVERRIDE="${CKPTS_OVERRIDE}" \
GPU_IDS="${GPU_ID}" \
EVAL_CHECKPOINTS_IN_PARALLEL=1 \
N_EPISODES=50 \
EVAL_BATCH_SIZE=50 \
MAX_EPISODES_RENDERED=1 \
DATA_ROOT="${DATA_ROOT}" \
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR}" \
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY}" \
SWEEP_OUTPUT_DIR="${SWEEP_OUTPUT_DIR}" \
QUEUE_LOG="${QUEUE_LOG}" \
TIMESTAMP="${TIMESTAMP}" \
bash "${RUNNER}"
