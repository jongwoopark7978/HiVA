#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${PYTHONPATH:-/home/jongwoopark/lerobot/src}"

DATA_ROOT="${DATA_ROOT:-/nfs/bigbrain/add_disk0/jongwoopark/libero_lerobot_v3_lerobotkeys}"
DATA_REPO_ID="${DATA_REPO_ID:-local/libero_lerobot_v3_lerobotkeys}"
OUTPUT_DIR="${OUTPUT_DIR:-/nfs/bigbrain/add_disk0/jongwoopark/duration_replay/purecontact_pcf_replays_saved}"
PURECONTACT_PCF_JSON="${PURECONTACT_PCF_JSON:-${DATA_ROOT}/meta/purecontact_pcf.json}"
CONDA_ENV="${CONDA_ENV:-smolvla_libero}"

VIDEOS_PER_COMBO="${VIDEOS_PER_COMBO:-10}"
SEED="${SEED:-0}"
OBSERVATION_HEIGHT="${OBSERVATION_HEIGHT:-360}"
OBSERVATION_WIDTH="${OBSERVATION_WIDTH:-360}"
VIDEO_FPS="${VIDEO_FPS:-20}"

run_python() {
  if [[ -n "${CONDA_ENV}" ]]; then
    conda run -n "${CONDA_ENV}" python "$@"
  else
    python "$@"
  fi
}

run_python /home/jongwoopark/lerobot/src/lerobot/scripts/detect_pure_contact_pcf_replays.py \
  --dataset.root "${DATA_ROOT}" \
  --dataset.repo-id "${DATA_REPO_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --pcf-source saved \
  --purecontact-pcf-json "${PURECONTACT_PCF_JSON}" \
  --seed "${SEED}" \
  --observation-height "${OBSERVATION_HEIGHT}" \
  --observation-width "${OBSERVATION_WIDTH}" \
  --video-fps "${VIDEO_FPS}" \
  --videos-per-combo "${VIDEOS_PER_COMBO}"
