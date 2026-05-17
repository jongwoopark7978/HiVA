#!/usr/bin/env bash
set -euo pipefail

# Sequential bigcornea sweep for MT coefficient-HiVA with continuous duration.
#
# Shared settings:
#   duration classes=[2,15]
#   S=2
#   hiva_duration_prediction_type=continuous_fm
#   no categorical duration head is used in continuous_fm mode
#   canonical MT d2/15 sidecar under /nfs/bigcornea/add_disk2/jongwoopark
#
# Jobs:
#   job1: hiva_duration_cont_norm=bounded,  hiva_suffix_attention=full
#   job2: hiva_duration_cont_norm=mean_std, hiva_suffix_attention=full
#   job3: hiva_duration_cont_norm=bounded,  hiva_suffix_attention=duration_reads_coeffs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_SCRIPT="${SCRIPT_DIR}/finetune_bigcornea_ckpt_20k_hiva_coeff_mt_continuous.sh"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"

SWEEP_TS="${SWEEP_TS:-$(date +%Y%m%d_%H%M%S)}"
OUTER_LOG="${OUTER_LOG:-${LOG_DIR}/hiva_coeff_mt_continuous_d2_15_bigcornea_${SWEEP_TS}.outer.log}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-64}"
S="${S:-2}"
HIVA_K="${HIVA_K:-10}"
HIVA_DMAX="${HIVA_DMAX:-15}"
HIVA_DEGREE="${HIVA_DEGREE:-3}"
HIVA_BASIS_MODE="${HIVA_BASIS_MODE:-canonical_mt}"
HIVA_DURATION_CLASSES="${HIVA_DURATION_CLASSES:-[2,15]}"
HIVA_DURATION_PREDICTION_TYPE="${HIVA_DURATION_PREDICTION_TYPE:-continuous_fm}"
HIVA_DURATION_FM_LOSS_WEIGHT="${HIVA_DURATION_FM_LOSS_WEIGHT:-1.0}"
HIVA_DURATION_LOSS="${HIVA_DURATION_LOSS:-ce_mean}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"

DATA_ROOT="${DATA_ROOT:-/nfs/bigcornea/add_disk2/jongwoopark/libero_lerobot_v3_lerobotkeys}"
SIDECAR="${SIDECAR:-/nfs/bigcornea/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k${HIVA_K}_canonical_mt.parquet}"
SIDECAR_SUMMARY="${SIDECAR_SUMMARY:-/nfs/bigcornea/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k${HIVA_K}_canonical_mt.summary.json}"

run_job() {
  local job_id="$1"
  local duration_cont_norm="$2"
  local suffix_attention="$3"

  local run_name
  run_name="smolvla_hiva_coeff_mt_contdur_d2_15_bigcornea_${job_id}_${duration_cont_norm}_${suffix_attention}_k${HIVA_K}_b${BATCH_PER_GPU}_s${S}_${SWEEP_TS}"
  run_name="${run_name//./p}"

  echo
  echo "======================================================================"
  echo "Starting ${job_id}: cont_norm=${duration_cont_norm}, suffix_attention=${suffix_attention}"
  echo "RUN_NAME=${run_name}"
  echo "S=${S}"
  echo "HIVA_DURATION_CLASSES=${HIVA_DURATION_CLASSES}"
  echo "HIVA_DURATION_PREDICTION_TYPE=${HIVA_DURATION_PREDICTION_TYPE}"
  echo "HIVA_DURATION_FM_LOSS_WEIGHT=${HIVA_DURATION_FM_LOSS_WEIGHT}"
  echo "HIVA_DURATION_CONT_NORM=${duration_cont_norm}"
  echo "HIVA_SUFFIX_ATTENTION=${suffix_attention}"
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
  HIVA_DURATION_PREDICTION_TYPE="${HIVA_DURATION_PREDICTION_TYPE}" \
  HIVA_DURATION_FM_LOSS_WEIGHT="${HIVA_DURATION_FM_LOSS_WEIGHT}" \
  HIVA_DURATION_CONT_NORM="${duration_cont_norm}" \
  HIVA_DURATION_LOSS="${HIVA_DURATION_LOSS}" \
  HIVA_SUFFIX_ATTENTION="${suffix_attention}" \
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
  echo "MT HiVA continuous d2/15 sweep started at $(date)"
  echo "Host: $(hostname)"
  echo "Outer log: ${OUTER_LOG}"
  echo "Train script: ${TRAIN_SCRIPT}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "NUM_GPUS=${NUM_GPUS}"
  echo "NUM_PROCESSES=${NUM_PROCESSES}"
  echo "BATCH_PER_GPU=${BATCH_PER_GPU}"
  echo "S=${S}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "SIDECAR=${SIDECAR}"
  echo "SIDECAR_SUMMARY=${SIDECAR_SUMMARY}"
  echo "WANDB_ENABLE=${WANDB_ENABLE}"
  echo "WANDB_PROJECT=${WANDB_PROJECT}"

  run_job "job1" "bounded" "full"
  run_job "job2" "mean_std" "full"
  run_job "job3" "bounded" "duration_reads_coeffs"

  echo "MT HiVA continuous d2/15 sweep finished at $(date)"
} 2>&1 | tee -a "${OUTER_LOG}"
