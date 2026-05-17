#!/usr/bin/env bash
set -euo pipefail

# Sequential bigcornea sweep for canonical max-target (MT) HiVA K=10.
#
# All jobs force hiva_duration_head_type=residual_ffn and use all 8 GPUs by default:
#   job1: S=4, hiva_duration_loss=ce_mean, hiva_suffix_attention=full
#   job2: S=4, hiva_duration_loss=ce_mean, hiva_suffix_attention=duration_prefix
#   job3a: repeat job1 with S=2
#   job3b: repeat job2 with S=2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

BUILD_SIDECAR_SCRIPT="${SCRIPT_DIR}/build_hiva_coeff_mt_sidecar_bigcornea.sh"
TRAIN_SCRIPT="${SCRIPT_DIR}/finetune_bigcornea_ckpt_20k_hiva_coeff_mt.sh"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"

SWEEP_TS="${SWEEP_TS:-$(date +%Y%m%d_%H%M%S)}"
OUTER_LOG="${OUTER_LOG:-${LOG_DIR}/hiva_coeff_mt_k10_residual_ffn_bigcornea_${SWEEP_TS}.outer.log}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-64}"
HIVA_K="${HIVA_K:-10}"
HIVA_BASIS_MODE="${HIVA_BASIS_MODE:-canonical_mt}"
HIVA_DURATION_HEAD_TYPE="residual_ffn"
HIVA_DURATION_NOISY_LOSS_WEIGHT="${HIVA_DURATION_NOISY_LOSS_WEIGHT:-1.0}"
HIVA_DURATION_CLEAN_LOSS_WEIGHT="${HIVA_DURATION_CLEAN_LOSS_WEIGHT:-0.0}"
HIVA_DURATION_NOISY_SIGMA="${HIVA_DURATION_NOISY_SIGMA:-0.25}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"

DATA_ROOT="${DATA_ROOT:-/nfs/bigcornea/add_disk2/jongwoopark/libero_lerobot_v3_lerobotkeys}"
DURATION_SIDECAR="${DURATION_SIDECAR:-/nfs/bigcornea/add_disk2/jongwoopark/libero_duration_sidecar_all_episodes.parquet}"
SIDECAR="${SIDECAR:-/nfs/bigcornea/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d6_10_15_k${HIVA_K}_canonical_mt.parquet}"
SIDECAR_SUMMARY="${SIDECAR_SUMMARY:-/nfs/bigcornea/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d6_10_15_k${HIVA_K}_canonical_mt.summary.json}"

run_job() {
  local job_id="$1"
  local step_reduction="$2"
  local duration_loss="$3"
  local suffix_attention="$4"
  local label="$5"

  local run_name
  run_name="smolvla_hiva_coeff_mt_k${HIVA_K}_bigcornea_${job_id}_${HIVA_DURATION_HEAD_TYPE}_${duration_loss}_${suffix_attention}_w${HIVA_DURATION_NOISY_LOSS_WEIGHT}_sigma${HIVA_DURATION_NOISY_SIGMA}_b${BATCH_PER_GPU}_s${step_reduction}_${SWEEP_TS}"
  run_name="${run_name//./p}"

  echo
  echo "======================================================================"
  echo "Starting ${job_id}: ${label}"
  echo "RUN_NAME=${run_name}"
  echo "S=${step_reduction}"
  echo "HIVA_DURATION_LOSS=${duration_loss}"
  echo "HIVA_SUFFIX_ATTENTION=${suffix_attention}"
  echo "HIVA_DURATION_HEAD_TYPE=${HIVA_DURATION_HEAD_TYPE}"
  echo "HIVA_DURATION_NOISY_SIGMA=${HIVA_DURATION_NOISY_SIGMA}"
  echo "HIVA_DURATION_NOISY_LOSS_WEIGHT=${HIVA_DURATION_NOISY_LOSS_WEIGHT}"
  echo "HIVA_DURATION_CLEAN_LOSS_WEIGHT=${HIVA_DURATION_CLEAN_LOSS_WEIGHT}"
  echo "HIVA_K=${HIVA_K}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "BATCH_PER_GPU=${BATCH_PER_GPU}"
  echo "SIDECAR=${SIDECAR}"
  echo "======================================================================"

  GPU_IDS="${GPU_IDS}" \
  NUM_GPUS="${NUM_GPUS}" \
  NUM_PROCESSES="${NUM_PROCESSES}" \
  BATCH_PER_GPU="${BATCH_PER_GPU}" \
  S="${step_reduction}" \
  HIVA_K="${HIVA_K}" \
  HIVA_BASIS_MODE="${HIVA_BASIS_MODE}" \
  HIVA_DURATION_LOSS="${duration_loss}" \
  HIVA_SUFFIX_ATTENTION="${suffix_attention}" \
  HIVA_DURATION_HEAD_TYPE="${HIVA_DURATION_HEAD_TYPE}" \
  HIVA_DURATION_NOISY_LOSS_WEIGHT="${HIVA_DURATION_NOISY_LOSS_WEIGHT}" \
  HIVA_DURATION_CLEAN_LOSS_WEIGHT="${HIVA_DURATION_CLEAN_LOSS_WEIGHT}" \
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
  echo "MT HiVA K=10 residual-FFN sweep started at $(date)"
  echo "Host: $(hostname)"
  echo "Outer log: ${OUTER_LOG}"
  echo "Build script: ${BUILD_SIDECAR_SCRIPT}"
  echo "Train script: ${TRAIN_SCRIPT}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "NUM_GPUS=${NUM_GPUS}"
  echo "NUM_PROCESSES=${NUM_PROCESSES}"
  echo "BATCH_PER_GPU=${BATCH_PER_GPU}"
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
  bash "${BUILD_SIDECAR_SCRIPT}"

  run_job "job1" 4 "ce_mean" "full" "MT max-target, batch-mean duration CE, full suffix attention"
  run_job "job2" 4 "ce_mean" "duration_prefix" "MT max-target, batch-mean duration CE, duration-prefix suffix attention"
  run_job "job3a" 2 "ce_mean" "full" "Repeat job1 at S=2"
  run_job "job3b" 2 "ce_mean" "duration_prefix" "Repeat job2 at S=2"

  echo "MT HiVA K=10 residual-FFN sweep finished at $(date)"
} 2>&1 | tee -a "${OUTER_LOG}"
