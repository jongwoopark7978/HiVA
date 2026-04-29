#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${PYTHONPATH:-/home/jongwoopark/lerobot/src}"

DATA_ROOT="${DATA_ROOT:-/nfs/bigbrain/add_disk0/jongwoopark/libero_lerobot_v3_lerobotkeys}"
DATA_REPO_ID="${DATA_REPO_ID:-local/libero_lerobot_v3_lerobotkeys}"
OUTPUT_DIR="${OUTPUT_DIR:-/nfs/bigbrain/add_disk0/jongwoopark/duration_replay/purecontact_pcf_replays_recomputed}"
PURECONTACT_PCF_JSON="${PURECONTACT_PCF_JSON:-${DATA_ROOT}/meta/purecontact_pcf.json}"
CONDA_ENV="${CONDA_ENV:-smolvla_libero}"

VIDEOS_PER_COMBO="${VIDEOS_PER_COMBO:-100000}"
MAX_EPISODES_PER_COMBO="${MAX_EPISODES_PER_COMBO:-100000}"
SEED="${SEED:-0}"
OBSERVATION_HEIGHT="${OBSERVATION_HEIGHT:-360}"
OBSERVATION_WIDTH="${OBSERVATION_WIDTH:-360}"
VIDEO_FPS="${VIDEO_FPS:-20}"
MOVEMENT_WINDOW="${MOVEMENT_WINDOW:-10}"
MOVEMENT_THRESHOLD="${MOVEMENT_THRESHOLD:-0.03}"
SAME_SIGN_K="${SAME_SIGN_K:-5}"
INIT_RESET_MODE="${INIT_RESET_MODE:-auto}"
FALLBACK_SEEDS="${FALLBACK_SEEDS:-2,3,4,5,10,42,100}"
SKIP_VIDEOS="${SKIP_VIDEOS:-1}"

SKIP_VIDEO_ARGS=()
if [[ "${SKIP_VIDEOS}" == "1" || "${SKIP_VIDEOS}" == "true" || "${SKIP_VIDEOS}" == "TRUE" ]]; then
  SKIP_VIDEO_ARGS+=(--skip-videos)
fi

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
  --pcf-source detect \
  --seed "${SEED}" \
  --observation-height "${OBSERVATION_HEIGHT}" \
  --observation-width "${OBSERVATION_WIDTH}" \
  --video-fps "${VIDEO_FPS}" \
  --movement-window "${MOVEMENT_WINDOW}" \
  --movement-threshold "${MOVEMENT_THRESHOLD}" \
  --same-sign-k "${SAME_SIGN_K}" \
  --init-reset-mode "${INIT_RESET_MODE}" \
  --fallback-seeds "${FALLBACK_SEEDS}" \
  --videos-per-combo "${VIDEOS_PER_COMBO}" \
  --max-episodes-per-combo "${MAX_EPISODES_PER_COMBO}" \
  "${SKIP_VIDEO_ARGS[@]}" \
  --write-purecontact-pcf-json "${PURECONTACT_PCF_JSON}"
