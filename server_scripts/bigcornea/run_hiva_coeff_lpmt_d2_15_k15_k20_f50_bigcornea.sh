#!/usr/bin/env bash
set -euo pipefail

# Sequential bigcornea finetune sweep for LP-MT coefficient HiVA.
#
# Shared settings:
#   duration classes={2,15}
#   executable Dmax=15
#   preview / fit horizon=50
#   hiva_duration_loss=ce_mean
#   hiva_suffix_attention=duration_reads_coeffs
#   hiva_duration_head_type=residual_ffn
#   S=2
#   batch size=64 per GPU
#   all 8 GPUs by default
#
# Jobs:
#   job1: K=15
#   job2: K=20

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_SCRIPT="${REPO_ROOT}/server_scripts/bigcornea/finetune_bigcornea_ckpt_20k_hiva_coeff_lp_mt.sh"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"

SWEEP_TS="${SWEEP_TS:-$(date +%Y%m%d_%H%M%S)}"
OUTER_LOG="${OUTER_LOG:-${LOG_DIR}/hiva_coeff_lpmt_d2_15_k15_k20_f50_bigcornea_${SWEEP_TS}.outer.log}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-64}"
S="${S:-2}"

HIVA_DURATION_CLASSES="${HIVA_DURATION_CLASSES:-[2,15]}"
HIVA_DMAX="${HIVA_DMAX:-15}"
HIVA_FIT_HORIZON="${HIVA_FIT_HORIZON:-50}"
HIVA_BASIS_MODE="${HIVA_BASIS_MODE:-canonical_lp_mt}"
HIVA_DURATION_LOSS="${HIVA_DURATION_LOSS:-ce_mean}"
HIVA_SUFFIX_ATTENTION="${HIVA_SUFFIX_ATTENTION:-duration_reads_coeffs}"
HIVA_DURATION_HEAD_TYPE="${HIVA_DURATION_HEAD_TYPE:-residual_ffn}"
HIVA_DURATION_PREDICTION_TYPE="${HIVA_DURATION_PREDICTION_TYPE:-categorical}"
HIVA_DURATION_NOISY_LOSS_WEIGHT="${HIVA_DURATION_NOISY_LOSS_WEIGHT:-1.0}"
HIVA_DURATION_CLEAN_LOSS_WEIGHT="${HIVA_DURATION_CLEAN_LOSS_WEIGHT:-0.0}"
HIVA_DURATION_NOISY_SIGMA="${HIVA_DURATION_NOISY_SIGMA:-0.25}"
HIVA_DECODED_ACTION_LOSS_WEIGHT="${HIVA_DECODED_ACTION_LOSS_WEIGHT:-0.0}"
HIVA_DECODED_TR_LOSS_WEIGHT="${HIVA_DECODED_TR_LOSS_WEIGHT:-1.0}"
HIVA_DECODED_ROT_LOSS_WEIGHT="${HIVA_DECODED_ROT_LOSS_WEIGHT:-1.0}"
HIVA_DECODED_GRIP_LOSS_WEIGHT="${HIVA_DECODED_GRIP_LOSS_WEIGHT:-1.0}"
HIVA_DECODED_PREFIX_WEIGHT="${HIVA_DECODED_PREFIX_WEIGHT:-1.0}"
HIVA_DECODED_POST_DURATION_EXEC_WEIGHT="${HIVA_DECODED_POST_DURATION_EXEC_WEIGHT:-0.5}"
HIVA_DECODED_PREVIEW_WEIGHT="${HIVA_DECODED_PREVIEW_WEIGHT:-0.1}"
HIVA_DECODED_TERMINAL_WEIGHT="${HIVA_DECODED_TERMINAL_WEIGHT:-0.0}"
HIVA_DECODED_LOSS_BETA="${HIVA_DECODED_LOSS_BETA:-0.1}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"

DATA_ROOT="${DATA_ROOT:-/nfs/bigcornea/add_disk2/jongwoopark/libero_lerobot_v3_lerobotkeys}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea/add_disk2/jongwoopark}"

run_job() {
  local job_id="$1"
  local hiva_k="$2"

  local sidecar="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k${hiva_k}_f${HIVA_FIT_HORIZON}_canonical_lp_mt.parquet"
  local sidecar_summary="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k${hiva_k}_f${HIVA_FIT_HORIZON}_canonical_lp_mt.summary.json"
  local run_name="smolvla_hiva_coeff_lpmt_d2_15_bigcornea_${job_id}_k${hiva_k}_f${HIVA_FIT_HORIZON}_${HIVA_DURATION_HEAD_TYPE}_${HIVA_DURATION_LOSS}_${HIVA_SUFFIX_ATTENTION}_b${BATCH_PER_GPU}_s${S}_${SWEEP_TS}"
  run_name="${run_name//./p}"

  if [[ ! -f "${sidecar}" ]]; then
    echo "ERROR: missing coefficient sidecar: ${sidecar}" >&2
    exit 2
  fi
  if [[ ! -f "${sidecar_summary}" ]]; then
    echo "ERROR: missing coefficient sidecar summary: ${sidecar_summary}" >&2
    exit 2
  fi

  echo
  echo "======================================================================"
  echo "Starting ${job_id}: LP-MT d2/15, K=${hiva_k}, f=${HIVA_FIT_HORIZON}"
  echo "RUN_NAME=${run_name}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "BATCH_PER_GPU=${BATCH_PER_GPU}"
  echo "S=${S}"
  echo "HIVA_DURATION_CLASSES=${HIVA_DURATION_CLASSES}"
  echo "HIVA_DMAX=${HIVA_DMAX}"
  echo "HIVA_FIT_HORIZON=${HIVA_FIT_HORIZON}"
  echo "HIVA_DURATION_LOSS=${HIVA_DURATION_LOSS}"
  echo "HIVA_SUFFIX_ATTENTION=${HIVA_SUFFIX_ATTENTION}"
  echo "HIVA_DURATION_HEAD_TYPE=${HIVA_DURATION_HEAD_TYPE}"
  echo "HIVA_DECODED_ACTION_LOSS_WEIGHT=${HIVA_DECODED_ACTION_LOSS_WEIGHT}"
  echo "HIVA_DECODED_PREFIX_WEIGHT=${HIVA_DECODED_PREFIX_WEIGHT}"
  echo "HIVA_DECODED_POST_DURATION_EXEC_WEIGHT=${HIVA_DECODED_POST_DURATION_EXEC_WEIGHT}"
  echo "HIVA_DECODED_PREVIEW_WEIGHT=${HIVA_DECODED_PREVIEW_WEIGHT}"
  echo "HIVA_DECODED_TERMINAL_WEIGHT=${HIVA_DECODED_TERMINAL_WEIGHT}"
  echo "SIDECAR=${sidecar}"
  echo "SIDECAR_SUMMARY=${sidecar_summary}"
  echo "======================================================================"

  GPU_IDS="${GPU_IDS}" \
  NUM_GPUS="${NUM_GPUS}" \
  NUM_PROCESSES="${NUM_PROCESSES}" \
  BATCH_PER_GPU="${BATCH_PER_GPU}" \
  S="${S}" \
  HIVA_DURATION_CLASSES="${HIVA_DURATION_CLASSES}" \
  HIVA_DMAX="${HIVA_DMAX}" \
  HIVA_FIT_HORIZON="${HIVA_FIT_HORIZON}" \
  HIVA_K="${hiva_k}" \
  HIVA_BASIS_MODE="${HIVA_BASIS_MODE}" \
  HIVA_DURATION_LOSS="${HIVA_DURATION_LOSS}" \
  HIVA_SUFFIX_ATTENTION="${HIVA_SUFFIX_ATTENTION}" \
  HIVA_DURATION_HEAD_TYPE="${HIVA_DURATION_HEAD_TYPE}" \
  HIVA_DURATION_PREDICTION_TYPE="${HIVA_DURATION_PREDICTION_TYPE}" \
  HIVA_DURATION_NOISY_LOSS_WEIGHT="${HIVA_DURATION_NOISY_LOSS_WEIGHT}" \
  HIVA_DURATION_CLEAN_LOSS_WEIGHT="${HIVA_DURATION_CLEAN_LOSS_WEIGHT}" \
  HIVA_DURATION_NOISY_SIGMA="${HIVA_DURATION_NOISY_SIGMA}" \
  HIVA_DECODED_ACTION_LOSS_WEIGHT="${HIVA_DECODED_ACTION_LOSS_WEIGHT}" \
  HIVA_DECODED_TR_LOSS_WEIGHT="${HIVA_DECODED_TR_LOSS_WEIGHT}" \
  HIVA_DECODED_ROT_LOSS_WEIGHT="${HIVA_DECODED_ROT_LOSS_WEIGHT}" \
  HIVA_DECODED_GRIP_LOSS_WEIGHT="${HIVA_DECODED_GRIP_LOSS_WEIGHT}" \
  HIVA_DECODED_PREFIX_WEIGHT="${HIVA_DECODED_PREFIX_WEIGHT}" \
  HIVA_DECODED_POST_DURATION_EXEC_WEIGHT="${HIVA_DECODED_POST_DURATION_EXEC_WEIGHT}" \
  HIVA_DECODED_PREVIEW_WEIGHT="${HIVA_DECODED_PREVIEW_WEIGHT}" \
  HIVA_DECODED_TERMINAL_WEIGHT="${HIVA_DECODED_TERMINAL_WEIGHT}" \
  HIVA_DECODED_LOSS_BETA="${HIVA_DECODED_LOSS_BETA}" \
  POLICY_CHUNK_SIZE="${HIVA_FIT_HORIZON}" \
  POLICY_N_ACTION_STEPS="${HIVA_DMAX}" \
  DATA_ROOT="${DATA_ROOT}" \
  SIDECAR="${sidecar}" \
  SIDECAR_SUMMARY="${sidecar_summary}" \
  WANDB_ENABLE="${WANDB_ENABLE}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  RUN_NAME="${run_name}" \
  bash "${TRAIN_SCRIPT}"

  echo "Finished ${job_id}: ${run_name}"
}

{
  echo "LP-MT d2/15 K15/K20 f50 bigcornea sweep started at $(date)"
  echo "Host: $(hostname)"
  echo "Outer log: ${OUTER_LOG}"
  echo "Train script: ${TRAIN_SCRIPT}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "NUM_GPUS=${NUM_GPUS}"
  echo "NUM_PROCESSES=${NUM_PROCESSES}"
  echo "BATCH_PER_GPU=${BATCH_PER_GPU}"
  echo "S=${S}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "SIDECAR_ROOT=${SIDECAR_ROOT}"
  echo "WANDB_ENABLE=${WANDB_ENABLE}"
  echo "WANDB_PROJECT=${WANDB_PROJECT}"

  run_job "job1" 15
  run_job "job2" 20

  echo "LP-MT d2/15 K15/K20 f50 bigcornea sweep finished at $(date)"
} 2>&1 | tee -a "${OUTER_LOG}"
