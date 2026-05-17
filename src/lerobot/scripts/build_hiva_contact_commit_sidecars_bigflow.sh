#!/usr/bin/env bash
set -euo pipefail

# Build contact-commit duration sidecars and matching canonical LP-MT coefficient sidecars.
#
# Versions:
#   v1: D={4,6,10}; commit d=6 on [b-4,e+4]; transition d=4 on [b-24,b-5] and [e+5,e+14]; free d=10
#   v2: D={4,10}; commit d=4 on [b-4,e+4]; transition d=4 on [b-24,b-5] and [e+5,e+14]; free d=10
#   v3: D={2,4,6,10}; pre-near d=2 on [b-14,b-5]; pre-far d=4 on [b-24,b-15]; commit d=6 on [b-4,e+4]; post d=4 on [e+5,e+14]; free d=10
#   v4: D={2,4,10}; pre-near d=2 on [b-14,b-5]; pre-far d=4 on [b-24,b-15]; commit d=4 on [b-4,e+4]; post d=4 on [e+5,e+14]; free d=10
#   v5: D={4,6,10}; commit d=6 on [b-10,e+10]; transition d=4 on [b-40,b-11] and [e+11,e+30]; free d=10
#   v6: D={4,10}; commit d=4 on [b-10,e+10]; transition d=4 on [b-40,b-11] and [e+11,e+30]; free d=10
#   v7: D={2,4,6,10}; pre-near d=2 on [b-25,b-11]; pre-far d=4 on [b-40,b-26]; commit d=6 on [b-10,e+10]; post d=4 on [e+11,e+30]; free d=10
#   v8: D={2,4,10}; pre-near d=2 on [b-25,b-11]; pre-far d=4 on [b-40,b-26]; commit d=4 on [b-10,e+10]; post d=4 on [e+11,e+30]; free d=10
#
# Example:
#   OVERWRITE=true VERSION=all HIVA_K=10 HIVA_FIT_HORIZON=15 \
#     bash server_scripts/bigflow/build_hiva_contact_commit_sidecars_bigflow.sh
#
# To build only one version:
#   VERSION=v3 OVERWRITE=true bash server_scripts/bigflow/build_hiva_contact_commit_sidecars_bigflow.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_BIN}/python}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-/nfs/bigflow/add_disk0/jongwoopark/libero_lerobot_v3_lerobotkeys}"
DATA_REPO_ID="${DATA_REPO_ID:-local/libero_lerobot_v3_lerobotkeys}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigflow/add_disk0/jongwoopark}"
INTERACTION_ZONE_SIDECAR="${INTERACTION_ZONE_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_interaction_zones_all_episodes.parquet}"
VERSION="${VERSION:-all}"
OVERWRITE="${OVERWRITE:-false}"

HIVA_K="${HIVA_K:-10}"
HIVA_DMAX="${HIVA_DMAX:-10}"
HIVA_FIT_HORIZON="${HIVA_FIT_HORIZON:-15}"
HIVA_DEGREE="${HIVA_DEGREE:-3}"
ROT_SCALE_ETA="${ROT_SCALE_ETA:-0.5}"
PREVIEW_TAIL_WEIGHT="${PREVIEW_TAIL_WEIGHT:-1.0}"
SMOOTH_WEIGHT="${SMOOTH_WEIGHT:-0.0}"

MERGE_WINDOW="${MERGE_WINDOW:-3}"
MATCH_WINDOW="${MATCH_WINDOW:-20}"
LABELER_VERSION_PREFIX="${LABELER_VERSION_PREFIX:-hiva_contact_commit}"
EPISODE_INDICES="${EPISODE_INDICES:-}"
EPISODE_START="${EPISODE_START:-}"
MAX_EPISODES="${MAX_EPISODES:-}"
PURECONTACT_PCF_JSON="${PURECONTACT_PCF_JSON:-}"

# Relative script paths inside the repository. Override these if the files are temporarily staged elsewhere.
DURATION_BUILDER="${DURATION_BUILDER:-${REPO_ROOT}/src/lerobot/scripts/build_hiva_duration_sidecar_contact_commit.py}"
COEFF_BUILDER="${COEFF_BUILDER:-${REPO_ROOT}/src/lerobot/scripts/build_hiva_coeff_sidecar_lp_mt.py}"

version_duration_classes() {
  case "$1" in
    v1|v5) echo "4 6 10" ;;
    v2|v6) echo "4 10" ;;
    v3|v7) echo "2 4 6 10" ;;
    v4|v8) echo "2 4 10" ;;
    *) echo "Unknown contact-commit version: $1" >&2; exit 2 ;;
  esac
}

version_tag() {
  case "$1" in
    v1) echo "v1_d4_6_10_commit6" ;;
    v2) echo "v2_d4_10_commit4" ;;
    v3) echo "v3_d2_4_6_10_prenear2_commit6" ;;
    v4) echo "v4_d2_4_10_prenear2_commit4" ;;
    v5) echo "v5_d4_6_10_wide_commit6" ;;
    v6) echo "v6_d4_10_wide_commit4" ;;
    v7) echo "v7_d2_4_6_10_wide_prenear2_commit6" ;;
    v8) echo "v8_d2_4_10_wide_prenear2_commit4" ;;
    *) echo "Unknown contact-commit version: $1" >&2; exit 2 ;;
  esac
}

build_one() {
  local version="$1"
  local classes tag duration_sidecar duration_summary coeff_sidecar coeff_summary
  classes="$(version_duration_classes "${version}")"
  tag="$(version_tag "${version}")"
  duration_sidecar="${SIDECAR_ROOT}/libero_duration_sidecar_${tag}_all_episodes.parquet"
  duration_summary="${SIDECAR_ROOT}/libero_duration_sidecar_${tag}_all_episodes.summary.json"
  coeff_sidecar="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_${tag}_k${HIVA_K}_f${HIVA_FIT_HORIZON}_canonical_lp_mt.parquet"
  coeff_summary="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_${tag}_k${HIVA_K}_f${HIVA_FIT_HORIZON}_canonical_lp_mt.summary.json"

  echo
  echo "======================================================================"
  echo "Building ${version} (${tag})"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "DURATION_CLASSES=${classes}"
  echo "HIVA_DMAX=${HIVA_DMAX}"
  echo "HIVA_FIT_HORIZON=${HIVA_FIT_HORIZON}"
  echo "HIVA_K=${HIVA_K}"
  echo "DURATION_SIDECAR=${duration_sidecar}"
  echo "COEFF_SIDECAR=${coeff_sidecar}"
  echo "INTERACTION_ZONE_SIDECAR=${INTERACTION_ZONE_SIDECAR}"
  echo "======================================================================"

  if [[ "${OVERWRITE}" == "true" || ! -e "${duration_sidecar}" || ! -e "${duration_summary}" ]]; then
    duration_args=(
      --dataset.repo-id "${DATA_REPO_ID}"
      --dataset.root "${DATA_ROOT}"
      --output "${duration_sidecar}"
      --summary-json "${duration_summary}"
      --profile "${version}"
      --merge-window "${MERGE_WINDOW}"
      --match-window "${MATCH_WINDOW}"
      --labeler-version "${LABELER_VERSION_PREFIX}_${version}"
      --interaction-zone-sidecar "${INTERACTION_ZONE_SIDECAR}"
    )
    if [[ -n "${EPISODE_INDICES}" ]]; then duration_args+=(--episode-indices "${EPISODE_INDICES}"); fi
    if [[ -n "${EPISODE_START}" ]]; then duration_args+=(--episode-start "${EPISODE_START}"); fi
    if [[ -n "${MAX_EPISODES}" ]]; then duration_args+=(--max-episodes "${MAX_EPISODES}"); fi
    if [[ -n "${PURECONTACT_PCF_JSON}" ]]; then duration_args+=(--purecontact-pcf-json "${PURECONTACT_PCF_JSON}"); fi
    "${PYTHON_BIN}" "${DURATION_BUILDER}" "${duration_args[@]}"
  else
    echo "Duration sidecar exists. Set OVERWRITE=true to rebuild: ${duration_sidecar}"
  fi

  if [[ "${OVERWRITE}" == "true" || ! -e "${coeff_sidecar}" || ! -e "${coeff_summary}" ]]; then
    read -r -a class_args <<< "${classes}"
    "${PYTHON_BIN}" "${COEFF_BUILDER}" \
      --data-root "${DATA_ROOT}" \
      --duration-sidecar "${duration_sidecar}" \
      --output "${coeff_sidecar}" \
      --summary-json "${coeff_summary}" \
      --duration-classes "${class_args[@]}" \
      --dmax "${HIVA_DMAX}" \
      --fit-horizon "${HIVA_FIT_HORIZON}" \
      --n-ctrl "${HIVA_K}" \
      --degree "${HIVA_DEGREE}" \
      --rot-scale-eta "${ROT_SCALE_ETA}" \
      --preview-tail-weight "${PREVIEW_TAIL_WEIGHT}" \
      --smooth "${SMOOTH_WEIGHT}"
  else
    echo "Coefficient sidecar exists. Set OVERWRITE=true to rebuild: ${coeff_sidecar}"
  fi
}

if [[ "${VERSION}" == "all" ]]; then
  for version in v1 v2 v3 v4 v5 v6 v7 v8; do
    build_one "${version}"
  done
elif [[ "${VERSION}" == "wide" || "${VERSION}" == "v5-v8" ]]; then
  for version in v5 v6 v7 v8; do
    build_one "${version}"
  done
else
  build_one "${VERSION}"
fi
