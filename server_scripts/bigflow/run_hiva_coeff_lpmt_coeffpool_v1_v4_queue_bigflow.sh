#!/usr/bin/env bash
set -euo pipefail

# Queue LP-MT HiVA coeff-modality-pool finetunes after the current GPU 0,2,3 job.
#
# Common settings:
#   K=10, F=15, hiva_duration_loss=ce_mean, hiva_suffix_attention=full,
#   hiva_duration_readout=coeff_modality_pool, batch per GPU=160.
#
# Notes:
#   v2/v4 duration sidecars were originally produced from wider class vocabularies with unused
#   classes. We build remapped coefficient sidecars for dur4,10 and dur2,4,10 so duration_class
#   targets are contiguous and match the requested classifier output size.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_BIN}/python}"

WAIT_FOR_SESSION="${WAIT_FOR_SESSION:-hiva_lpmt_d2_10_coeffpool_gpu023_20260510_024619}"
GPU_IDS="${GPU_IDS:-0,2,3}"
NUM_GPUS="${NUM_GPUS:-3}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-160}"
S="${S:-2}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"

DATA_ROOT="${DATA_ROOT:-/nfs/bigflow/add_disk0/jongwoopark/libero_lerobot_v3_lerobotkeys}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigflow/add_disk0/jongwoopark}"
K="${K:-10}"
F="${F:-15}"
Dmax="${Dmax:-10}"

COEFF_BUILDER="${COEFF_BUILDER:-${REPO_ROOT}/src/lerobot/scripts/build_hiva_coeff_sidecar_lp_mt.py}"
FINETUNE_SCRIPT="${FINETUNE_SCRIPT:-${SCRIPT_DIR}/finetune_bigflow_ckpt_20k_hiva_coeff_lp_mt.sh}"

wait_for_current_job() {
  if [[ -z "${WAIT_FOR_SESSION}" ]]; then
    return
  fi
  while tmux has-session -t "${WAIT_FOR_SESSION}" 2>/dev/null; do
    echo "Waiting for tmux session ${WAIT_FOR_SESSION} to finish before launching queued jobs..."
    sleep 300
  done
}

build_coeff_if_missing() {
  local duration_sidecar="$1"
  local output="$2"
  local summary="$3"
  shift 3
  local classes=("$@")
  if [[ -s "${output}" && -s "${summary}" ]]; then
    echo "Using existing remapped coefficient sidecar: ${output}"
    return
  fi
  echo "Building remapped coefficient sidecar: ${output}"
  "${PYTHON_BIN}" "${COEFF_BUILDER}" \
    --data-root "${DATA_ROOT}" \
    --duration-sidecar "${duration_sidecar}" \
    --output "${output}" \
    --summary-json "${summary}" \
    --duration-classes "${classes[@]}" \
    --dmax "${Dmax}" \
    --fit-horizon "${F}" \
    --n-ctrl "${K}" \
    --degree 3 \
    --rot-scale-eta 0.5 \
    --preview-tail-weight 1.0 \
    --smooth 0.0
}

run_job() {
  local dur_tag="$1"
  local class_expr="$2"
  local sidecar="$3"
  local summary="$4"
  local run_name="smolvla_hiva_coeff_lpmt_dur${dur_tag}_coeffpool_full_ce_mean_k${K}_f${F}_bigflow_b${BATCH_PER_GPU}_s${S}_$(date +%Y%m%d_%H%M%S)"
  echo
  echo "======================================================================"
  echo "Launching ${run_name}"
  echo "classes=${class_expr}"
  echo "sidecar=${sidecar}"
  echo "summary=${summary}"
  echo "======================================================================"
  GPU_IDS="${GPU_IDS}" \
  NUM_GPUS="${NUM_GPUS}" \
  NUM_PROCESSES="${NUM_PROCESSES}" \
  BATCH_PER_GPU="${BATCH_PER_GPU}" \
  S="${S}" \
  WANDB_ENABLE="${WANDB_ENABLE}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  HIVA_DURATION_CLASSES="${class_expr}" \
  HIVA_DMAX="${Dmax}" \
  HIVA_FIT_HORIZON="${F}" \
  HIVA_K="${K}" \
  HIVA_BASIS_MODE="canonical_lp_mt" \
  HIVA_DURATION_LOSS="ce_mean" \
  HIVA_SUFFIX_ATTENTION="full" \
  HIVA_DURATION_READOUT="coeff_modality_pool" \
  HIVA_DURATION_HEAD_TYPE="residual_ffn" \
  POLICY_CHUNK_SIZE="${F}" \
  POLICY_N_ACTION_STEPS="${Dmax}" \
  SIDECAR="${sidecar}" \
  SIDECAR_SUMMARY="${summary}" \
  RUN_NAME="${run_name}" \
  bash "${FINETUNE_SCRIPT}"
}

wait_for_current_job

V2_REMAP="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v2_d4_10_commit4_k${K}_f${F}_canonical_lp_mt.parquet"
V2_REMAP_SUMMARY="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v2_d4_10_commit4_k${K}_f${F}_canonical_lp_mt.summary.json"
V4_REMAP="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v4_d2_4_10_prenear2_commit4_k${K}_f${F}_canonical_lp_mt.parquet"
V4_REMAP_SUMMARY="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v4_d2_4_10_prenear2_commit4_k${K}_f${F}_canonical_lp_mt.summary.json"

build_coeff_if_missing \
  "${SIDECAR_ROOT}/libero_duration_sidecar_v2_d4_10_commit4_all_episodes.parquet" \
  "${V2_REMAP}" \
  "${V2_REMAP_SUMMARY}" \
  4 10

build_coeff_if_missing \
  "${SIDECAR_ROOT}/libero_duration_sidecar_v4_d2_4_10_prenear2_commit4_all_episodes.parquet" \
  "${V4_REMAP}" \
  "${V4_REMAP_SUMMARY}" \
  2 4 10

run_job \
  "4,6,10" \
  "[4,6,10]" \
  "${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v1_d4_6_10_commit6_k${K}_f${F}_canonical_lp_mt.parquet" \
  "${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v1_d4_6_10_commit6_k${K}_f${F}_canonical_lp_mt.summary.json"

run_job \
  "4,10" \
  "[4,10]" \
  "${V2_REMAP}" \
  "${V2_REMAP_SUMMARY}"

run_job \
  "2,4,6,10" \
  "[2,4,6,10]" \
  "${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v3_d2_4_6_10_prenear2_commit6_k${K}_f${F}_canonical_lp_mt.parquet" \
  "${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v3_d2_4_6_10_prenear2_commit6_k${K}_f${F}_canonical_lp_mt.summary.json"

run_job \
  "2,4,10" \
  "[2,4,10]" \
  "${V4_REMAP}" \
  "${V4_REMAP_SUMMARY}"
