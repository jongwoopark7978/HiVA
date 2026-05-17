#!/usr/bin/env bash
set -euo pipefail

# Queue LP-MT HiVA coeff-modality-pool finetunes on bigcornea.
#
# Common settings:
#   K=10, F=15, hiva_duration_loss=ce_mean, hiva_suffix_attention=full,
#   hiva_duration_readout=coeff_modality_pool, batch per GPU=64, GPUs=0-7.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-64}"
S="${S:-2}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea/add_disk2/jongwoopark}"
K="${K:-10}"
F="${F:-15}"
DMAX="${DMAX:-10}"
FINETUNE_SCRIPT="${FINETUNE_SCRIPT:-${SCRIPT_DIR}/finetune_bigcornea_ckpt_20k_hiva_coeff_lp_mt.sh}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

run_job() {
  local job_id="$1"
  local dur_tag="$2"
  local class_expr="$3"
  local sidecar="$4"
  local summary="$5"
  local duration_sidecar="$6"
  local duration_summary="$7"

  require_file "${sidecar}"
  require_file "${summary}"
  require_file "${duration_sidecar}"
  require_file "${duration_summary}"

  local run_name="smolvla_hiva_coeff_lpmt_coeffpool_${job_id}_d${dur_tag}_full_ce_mean_k${K}_f${F}_bigcornea_b${BATCH_PER_GPU}_s${S}_$(date +%Y%m%d_%H%M%S)"

  echo
  echo "======================================================================"
  echo "Launching ${run_name}"
  echo "duration_classes=${class_expr}"
  echo "sidecar=${sidecar}"
  echo "summary=${summary}"
  echo "duration_sidecar=${duration_sidecar}"
  echo "duration_summary=${duration_summary}"
  echo "======================================================================"

  GPU_IDS="${GPU_IDS}" \
  NUM_GPUS="${NUM_GPUS}" \
  NUM_PROCESSES="${NUM_PROCESSES}" \
  BATCH_PER_GPU="${BATCH_PER_GPU}" \
  S="${S}" \
  WANDB_ENABLE="${WANDB_ENABLE}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  HIVA_DURATION_CLASSES="${class_expr}" \
  HIVA_DMAX="${DMAX}" \
  HIVA_FIT_HORIZON="${F}" \
  HIVA_K="${K}" \
  HIVA_BASIS_MODE="canonical_lp_mt" \
  HIVA_DURATION_LOSS="ce_mean" \
  HIVA_SUFFIX_ATTENTION="full" \
  HIVA_DURATION_READOUT="coeff_modality_pool" \
  HIVA_DURATION_HEAD_TYPE="${HIVA_DURATION_HEAD_TYPE:-residual_ffn}" \
  POLICY_CHUNK_SIZE="${F}" \
  POLICY_N_ACTION_STEPS="${DMAX}" \
  SIDECAR="${sidecar}" \
  SIDECAR_SUMMARY="${summary}" \
  RUN_NAME="${run_name}" \
  bash "${FINETUNE_SCRIPT}"
}

run_job \
  "job1_v5" \
  "4_6_10" \
  "[4,6,10]" \
  "${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet" \
  "${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json" \
  "${SIDECAR_ROOT}/libero_duration_sidecar_v5_d4_6_10_wide_commit6_all_episodes.parquet" \
  "${SIDECAR_ROOT}/libero_duration_sidecar_v5_d4_6_10_wide_commit6_all_episodes.summary.json"

run_job \
  "job2_v6" \
  "4_10" \
  "[4,10]" \
  "${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v6_d4_10_wide_commit4_k10_f15_canonical_lp_mt.parquet" \
  "${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v6_d4_10_wide_commit4_k10_f15_canonical_lp_mt.summary.json" \
  "${SIDECAR_ROOT}/libero_duration_sidecar_v6_d4_10_wide_commit4_all_episodes.parquet" \
  "${SIDECAR_ROOT}/libero_duration_sidecar_v6_d4_10_wide_commit4_all_episodes.summary.json"

run_job \
  "job3_v7" \
  "2_4_6_10" \
  "[2,4,6,10]" \
  "${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v7_d2_4_6_10_wide_prenear2_commit6_k10_f15_canonical_lp_mt.parquet" \
  "${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v7_d2_4_6_10_wide_prenear2_commit6_k10_f15_canonical_lp_mt.summary.json" \
  "${SIDECAR_ROOT}/libero_duration_sidecar_v7_d2_4_6_10_wide_prenear2_commit6_all_episodes.parquet" \
  "${SIDECAR_ROOT}/libero_duration_sidecar_v7_d2_4_6_10_wide_prenear2_commit6_all_episodes.summary.json"

run_job \
  "job4_v8" \
  "2_4_10" \
  "[2,4,10]" \
  "${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v8_d2_4_10_wide_prenear2_commit4_k10_f15_canonical_lp_mt.parquet" \
  "${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v8_d2_4_10_wide_prenear2_commit4_k10_f15_canonical_lp_mt.summary.json" \
  "${SIDECAR_ROOT}/libero_duration_sidecar_v8_d2_4_10_wide_prenear2_commit4_all_episodes.parquet" \
  "${SIDECAR_ROOT}/libero_duration_sidecar_v8_d2_4_10_wide_prenear2_commit4_all_episodes.summary.json"
