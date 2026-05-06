#!/usr/bin/env bash
set -euo pipefail

# Sequential bigflow sweep for HP HiVA coefficient SmolVLA, canonical HP K=10.
#
# Runs six jobs on GPUs 4-7 by default:
#   1) S=4, hiva_duration_loss=duration_noisy_weights, hiva_suffix_attention=full
#   2) S=4, hiva_duration_loss=ce_mean,                hiva_suffix_attention=full
#   3) S=4, hiva_duration_loss=ce_mean,                hiva_suffix_attention=duration_prefix
#   4) S=2, hiva_duration_loss=duration_noisy_weights, hiva_suffix_attention=full
#   5) S=2, hiva_duration_loss=ce_mean,                hiva_suffix_attention=full
#   6) S=2, hiva_duration_loss=ce_mean,                hiva_suffix_attention=duration_prefix
#
# Example:
#   bash server_scripts/bigflow/run_hiva_coeff_hp_k10_attention_loss_sweep_bigflow.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_SCRIPT="${REPO_ROOT}/server_scripts/bigflow/finetune_bigflow_ckpt_20k_hiva_coeff_hp.sh"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"

SWEEP_TS="${SWEEP_TS:-$(date +%Y%m%d_%H%M%S)}"
OUTER_LOG="${OUTER_LOG:-${LOG_DIR}/hiva_coeff_hp_k10_attention_loss_sweep_${SWEEP_TS}.outer.log}"

GPU_IDS="${GPU_IDS:-4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-160}"
HIVA_K="${HIVA_K:-10}"
HIVA_BASIS_MODE="${HIVA_BASIS_MODE:-canonical_hp}"
HIVA_DURATION_NOISY_LOSS_WEIGHT="${HIVA_DURATION_NOISY_LOSS_WEIGHT:-1.0}"
HIVA_DURATION_NOISY_SIGMA="${HIVA_DURATION_NOISY_SIGMA:-0.25}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"

run_job() {
  local job_id="$1"
  local step_reduction="$2"
  local duration_loss="$3"
  local suffix_attention="$4"
  local label="$5"

  local run_name
  run_name="smolvla_hiva_coeff_hp_k${HIVA_K}_bigflow_${job_id}_${duration_loss}_${suffix_attention}_w${HIVA_DURATION_NOISY_LOSS_WEIGHT}_sigma${HIVA_DURATION_NOISY_SIGMA}_b${BATCH_PER_GPU}_s${step_reduction}_${SWEEP_TS}"
  run_name="${run_name//./p}"

  echo
  echo "======================================================================"
  echo "Starting ${job_id}: ${label}"
  echo "RUN_NAME=${run_name}"
  echo "S=${step_reduction}"
  echo "HIVA_DURATION_LOSS=${duration_loss}"
  echo "HIVA_SUFFIX_ATTENTION=${suffix_attention}"
  echo "HIVA_DURATION_NOISY_SIGMA=${HIVA_DURATION_NOISY_SIGMA}"
  echo "HIVA_DURATION_NOISY_LOSS_WEIGHT=${HIVA_DURATION_NOISY_LOSS_WEIGHT}"
  echo "HIVA_K=${HIVA_K}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "BATCH_PER_GPU=${BATCH_PER_GPU}"
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
  HIVA_DURATION_NOISY_LOSS_WEIGHT="${HIVA_DURATION_NOISY_LOSS_WEIGHT}" \
  HIVA_DURATION_NOISY_SIGMA="${HIVA_DURATION_NOISY_SIGMA}" \
  WANDB_ENABLE="${WANDB_ENABLE}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  RUN_NAME="${run_name}" \
  bash "${TRAIN_SCRIPT}"

  echo "Finished ${job_id}: ${run_name}"
}

{
  echo "HP HiVA K=10 attention/loss sweep started at $(date)"
  echo "Host: $(hostname)"
  echo "Outer log: ${OUTER_LOG}"
  echo "Train script: ${TRAIN_SCRIPT}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "NUM_GPUS=${NUM_GPUS}"
  echo "NUM_PROCESSES=${NUM_PROCESSES}"
  echo "BATCH_PER_GPU=${BATCH_PER_GPU}"
  echo "WANDB_ENABLE=${WANDB_ENABLE}"
  echo "WANDB_PROJECT=${WANDB_PROJECT}"

  run_job "job1" 4 "duration_noisy_weights" "full" "normalized noisy duration CE, full suffix attention"
  run_job "job2" 4 "ce_mean" "full" "batch-mean CE, full suffix attention"
  run_job "job3" 4 "ce_mean" "duration_prefix" "batch-mean CE, duration-prefix suffix attention"
  run_job "job4a" 2 "duration_noisy_weights" "full" "normalized noisy duration CE, full suffix attention"
  run_job "job4b" 2 "ce_mean" "full" "batch-mean CE, full suffix attention"
  run_job "job4c" 2 "ce_mean" "duration_prefix" "batch-mean CE, duration-prefix suffix attention"

  echo "HP HiVA K=10 attention/loss sweep finished at $(date)"
} 2>&1 | tee -a "${OUTER_LOG}"
