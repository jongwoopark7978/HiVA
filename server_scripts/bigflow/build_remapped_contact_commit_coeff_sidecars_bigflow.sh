#!/usr/bin/env bash
set -euo pipefail

# Rebuild contact-commit LP-MT coefficient sidecars with compact duration class vocabularies.
#
# v2/v6 assign only {4,10}; the original sidecars were generated with {4,6,10}.
# v4/v8 assign only {2,4,10}; the original sidecars were generated with {2,4,6,10}.
# Rebuilding makes duration_class contiguous for the classifier output size used in training.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_BIN}/python}"
DATA_ROOT="${DATA_ROOT:-/nfs/bigflow/add_disk0/jongwoopark/libero_lerobot_v3_lerobotkeys}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigflow/add_disk0/jongwoopark}"
K="${K:-10}"
F="${F:-15}"
DMAX="${DMAX:-10}"
COEFF_BUILDER="${COEFF_BUILDER:-${REPO_ROOT}/src/lerobot/scripts/build_hiva_coeff_sidecar_lp_mt.py}"

build_one() {
  local tag="$1"
  local duration_sidecar="$2"
  local output="$3"
  local summary="$4"
  shift 4
  local classes=("$@")

  echo
  echo "======================================================================"
  echo "Rebuilding ${tag}"
  echo "duration_sidecar=${duration_sidecar}"
  echo "output=${output}"
  echo "summary=${summary}"
  echo "duration_classes=${classes[*]}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (builder is CPU/NFS-bound)"
  echo "======================================================================"

  rm -f "${output}" "${summary}"
  "${PYTHON_BIN}" "${COEFF_BUILDER}" \
    --data-root "${DATA_ROOT}" \
    --duration-sidecar "${duration_sidecar}" \
    --output "${output}" \
    --summary-json "${summary}" \
    --duration-classes "${classes[@]}" \
    --dmax "${DMAX}" \
    --fit-horizon "${F}" \
    --n-ctrl "${K}" \
    --degree 3 \
    --rot-scale-eta 0.5 \
    --preview-tail-weight 1.0 \
    --smooth 0.0
}

build_one \
  "v2_d4_10_commit4" \
  "${SIDECAR_ROOT}/libero_duration_sidecar_v2_d4_10_commit4_all_episodes.parquet" \
  "${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v2_d4_10_commit4_k${K}_f${F}_canonical_lp_mt.parquet" \
  "${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v2_d4_10_commit4_k${K}_f${F}_canonical_lp_mt.summary.json" \
  4 10

build_one \
  "v4_d2_4_10_prenear2_commit4" \
  "${SIDECAR_ROOT}/libero_duration_sidecar_v4_d2_4_10_prenear2_commit4_all_episodes.parquet" \
  "${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v4_d2_4_10_prenear2_commit4_k${K}_f${F}_canonical_lp_mt.parquet" \
  "${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v4_d2_4_10_prenear2_commit4_k${K}_f${F}_canonical_lp_mt.summary.json" \
  2 4 10

build_one \
  "v6_d4_10_wide_commit4" \
  "${SIDECAR_ROOT}/libero_duration_sidecar_v6_d4_10_wide_commit4_all_episodes.parquet" \
  "${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v6_d4_10_wide_commit4_k${K}_f${F}_canonical_lp_mt.parquet" \
  "${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v6_d4_10_wide_commit4_k${K}_f${F}_canonical_lp_mt.summary.json" \
  4 10

build_one \
  "v8_d2_4_10_wide_prenear2_commit4" \
  "${SIDECAR_ROOT}/libero_duration_sidecar_v8_d2_4_10_wide_prenear2_commit4_all_episodes.parquet" \
  "${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v8_d2_4_10_wide_prenear2_commit4_k${K}_f${F}_canonical_lp_mt.parquet" \
  "${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v8_d2_4_10_wide_prenear2_commit4_k${K}_f${F}_canonical_lp_mt.summary.json" \
  2 4 10
