#!/usr/bin/env bash
set -euo pipefail

# Build v5 wide-commit LP-MT coefficient sidecars for multiple B-spline degrees.
#
# Source duration sidecar is unchanged across degrees:
#   libero_duration_sidecar_v5_d4_6_10_wide_commit6_all_episodes.parquet
#
# Outputs include the degree in the filename:
#   libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p${P}_f15_canonical_lp_mt.parquet

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
DEGREES="${DEGREES:-4 5 6 7}"
OVERWRITE="${OVERWRITE:-false}"
COEFF_BUILDER="${COEFF_BUILDER:-${REPO_ROOT}/src/lerobot/scripts/build_hiva_coeff_sidecar_lp_mt.py}"

DURATION_SIDECAR="${DURATION_SIDECAR:-${SIDECAR_ROOT}/libero_duration_sidecar_v5_d4_6_10_wide_commit6_all_episodes.parquet}"

for degree in ${DEGREES}; do
  output="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k${K}_p${degree}_f${F}_canonical_lp_mt.parquet"
  summary="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k${K}_p${degree}_f${F}_canonical_lp_mt.summary.json"

  echo
  echo "======================================================================"
  echo "Building v5 LP-MT coefficient sidecar with degree P=${degree}"
  echo "DURATION_SIDECAR=${DURATION_SIDECAR}"
  echo "OUTPUT=${output}"
  echo "SUMMARY=${summary}"
  echo "K=${K}, F=${F}, DMAX=${DMAX}, DEGREE=${degree}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (builder is CPU/NFS-bound)"
  echo "======================================================================"

  if [[ "${OVERWRITE}" == "true" ]]; then
    rm -f "${output}" "${summary}"
  fi
  if [[ -s "${output}" && -s "${summary}" ]]; then
    echo "Sidecar exists. Set OVERWRITE=true to rebuild: ${output}"
    continue
  fi

  "${PYTHON_BIN}" "${COEFF_BUILDER}" \
    --data-root "${DATA_ROOT}" \
    --duration-sidecar "${DURATION_SIDECAR}" \
    --output "${output}" \
    --summary-json "${summary}" \
    --duration-classes 4 6 10 \
    --dmax "${DMAX}" \
    --fit-horizon "${F}" \
    --n-ctrl "${K}" \
    --degree "${degree}" \
    --rot-scale-eta 0.5 \
    --preview-tail-weight 1.0 \
    --smooth 0.0
done
