#!/usr/bin/env bash
set -euo pipefail

# Two-job bigcornea test for the new MT coefficient-HiVA duration methods.
#
# Shared settings:
#   duration classes=[6,10,15]
#   hiva_duration_head_type=residual_ffn
#   hiva_duration_loss=ce_mean
#   hiva_basis_mode=canonical_mt
#   GPU_IDS=0,1,2,3,4,5,6,7
#   BATCH_PER_GPU=64
#   S=4
#
# Jobs:
#   job1: hiva_suffix_attention=duration_reads_coeffs, clean duration loss disabled
#   job2: hiva_suffix_attention=full, HIVA_DURATION_CLEAN_LOSS_WEIGHT=1.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

BUILD_SIDECAR_SCRIPT="${SCRIPT_DIR}/build_hiva_coeff_mt_sidecar_bigcornea.sh"
TRAIN_SCRIPT="${SCRIPT_DIR}/finetune_bigcornea_ckpt_20k_hiva_coeff_mt.sh"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"

SWEEP_TS="${SWEEP_TS:-$(date +%Y%m%d_%H%M%S)}"
OUTER_LOG="${OUTER_LOG:-${LOG_DIR}/hiva_coeff_mt_k10_new_methods_bigcornea_${SWEEP_TS}.outer.log}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-64}"
S="${S:-4}"
HIVA_K="${HIVA_K:-10}"
HIVA_DMAX="${HIVA_DMAX:-15}"
HIVA_DEGREE="${HIVA_DEGREE:-3}"
HIVA_BASIS_MODE="${HIVA_BASIS_MODE:-canonical_mt}"
HIVA_DURATION_CLASSES="${HIVA_DURATION_CLASSES:-[6,10,15]}"
HIVA_DURATION_HEAD_TYPE="residual_ffn"
HIVA_DURATION_LOSS="${HIVA_DURATION_LOSS:-ce_mean}"
HIVA_DURATION_NOISY_LOSS_WEIGHT="${HIVA_DURATION_NOISY_LOSS_WEIGHT:-1.0}"
HIVA_DURATION_NOISY_SIGMA="${HIVA_DURATION_NOISY_SIGMA:-0.25}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"

DATA_ROOT="${DATA_ROOT:-/nfs/bigcornea/add_disk2/jongwoopark/libero_lerobot_v3_lerobotkeys}"
DURATION_SIDECAR="${DURATION_SIDECAR:-/nfs/bigcornea/add_disk2/jongwoopark/libero_duration_sidecar_all_episodes.parquet}"
SIDECAR="${SIDECAR:-/nfs/bigcornea/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d6_10_15_k${HIVA_K}_canonical_mt.parquet}"
SIDECAR_SUMMARY="${SIDECAR_SUMMARY:-/nfs/bigcornea/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d6_10_15_k${HIVA_K}_canonical_mt.summary.json}"

run_job() {
  local job_id="$1"
  local suffix_attention="$2"
  local clean_loss_weight="$3"
  local label="$4"

  local run_name
  run_name="smolvla_hiva_coeff_mt_k${HIVA_K}_bigcornea_${job_id}_${HIVA_DURATION_HEAD_TYPE}_${HIVA_DURATION_LOSS}_${suffix_attention}_clean${clean_loss_weight}_w${HIVA_DURATION_NOISY_LOSS_WEIGHT}_sigma${HIVA_DURATION_NOISY_SIGMA}_b${BATCH_PER_GPU}_s${S}_${SWEEP_TS}"
  run_name="${run_name//./p}"

  echo
  echo "======================================================================"
  echo "Starting ${job_id}: ${label}"
  echo "RUN_NAME=${run_name}"
  echo "S=${S}"
  echo "HIVA_DURATION_CLASSES=${HIVA_DURATION_CLASSES}"
  echo "HIVA_DURATION_LOSS=${HIVA_DURATION_LOSS}"
  echo "HIVA_SUFFIX_ATTENTION=${suffix_attention}"
  echo "HIVA_DURATION_HEAD_TYPE=${HIVA_DURATION_HEAD_TYPE}"
  echo "HIVA_DURATION_NOISY_SIGMA=${HIVA_DURATION_NOISY_SIGMA}"
  echo "HIVA_DURATION_NOISY_LOSS_WEIGHT=${HIVA_DURATION_NOISY_LOSS_WEIGHT}"
  echo "HIVA_DURATION_CLEAN_LOSS_WEIGHT=${clean_loss_weight}"
  echo "HIVA_K=${HIVA_K}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "BATCH_PER_GPU=${BATCH_PER_GPU}"
  echo "SIDECAR=${SIDECAR}"
  echo "======================================================================"

  GPU_IDS="${GPU_IDS}" \
  NUM_GPUS="${NUM_GPUS}" \
  NUM_PROCESSES="${NUM_PROCESSES}" \
  BATCH_PER_GPU="${BATCH_PER_GPU}" \
  S="${S}" \
  HIVA_K="${HIVA_K}" \
  HIVA_DMAX="${HIVA_DMAX}" \
  HIVA_DEGREE="${HIVA_DEGREE}" \
  HIVA_BASIS_MODE="${HIVA_BASIS_MODE}" \
  HIVA_DURATION_CLASSES="${HIVA_DURATION_CLASSES}" \
  HIVA_DURATION_LOSS="${HIVA_DURATION_LOSS}" \
  HIVA_SUFFIX_ATTENTION="${suffix_attention}" \
  HIVA_DURATION_HEAD_TYPE="${HIVA_DURATION_HEAD_TYPE}" \
  HIVA_DURATION_NOISY_LOSS_WEIGHT="${HIVA_DURATION_NOISY_LOSS_WEIGHT}" \
  HIVA_DURATION_CLEAN_LOSS_WEIGHT="${clean_loss_weight}" \
  HIVA_DURATION_NOISY_SIGMA="${HIVA_DURATION_NOISY_SIGMA}" \
  DATA_ROOT="${DATA_ROOT}" \
  SIDECAR="${SIDECAR}" \
  SIDECAR_SUMMARY="${SIDECAR_SUMMARY}" \
  WANDB_ENABLE="${WANDB_ENABLE}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  RUN_NAME="${run_name}" \
  bash "${TRAIN_SCRIPT}"

  echo "Finished ${job_id}: ${run_name}"
}

{
  echo "MT HiVA K=10 new-methods sweep started at $(date)"
  echo "Host: $(hostname)"
  echo "Outer log: ${OUTER_LOG}"
  echo "Build script: ${BUILD_SIDECAR_SCRIPT}"
  echo "Train script: ${TRAIN_SCRIPT}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "NUM_GPUS=${NUM_GPUS}"
  echo "NUM_PROCESSES=${NUM_PROCESSES}"
  echo "BATCH_PER_GPU=${BATCH_PER_GPU}"
  echo "S=${S}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "DURATION_SIDECAR=${DURATION_SIDECAR}"
  echo "SIDECAR=${SIDECAR}"
  echo "SIDECAR_SUMMARY=${SIDECAR_SUMMARY}"
  echo "WANDB_ENABLE=${WANDB_ENABLE}"
  echo "WANDB_PROJECT=${WANDB_PROJECT}"

  DATA_ROOT="${DATA_ROOT}" \
  DURATION_SIDECAR="${DURATION_SIDECAR}" \
  OUTPUT="${SIDECAR}" \
  SUMMARY_JSON="${SIDECAR_SUMMARY}" \
  HIVA_K="${HIVA_K}" \
  HIVA_DMAX="${HIVA_DMAX}" \
  HIVA_DEGREE="${HIVA_DEGREE}" \
  bash "${BUILD_SIDECAR_SCRIPT}"

  run_job "job1" "duration_reads_coeffs" "0.0" "MT duration reads coefficient tokens with no clean-duration CE"
  run_job "job2" "full" "1.0" "MT full suffix attention with clean coefficient teacher-forced duration CE"

  echo "MT HiVA K=10 new-methods sweep finished at $(date)"
} 2>&1 | tee -a "${OUTER_LOG}"
