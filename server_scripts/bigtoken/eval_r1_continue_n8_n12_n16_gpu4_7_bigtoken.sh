#!/usr/bin/env bash
set -euo pipefail

# Continuation for the R1 original SmolVLA action-step sweep.
#
# The previous run timestamp is kept for output directories so completed and
# resumed suites are summarized together:
#   - n_action_steps=8: run only missing libero_object and libero_goal.
#   - n_action_steps=12: run all 4 suites.
#   - n_action_steps=16: run all 4 suites.
#
# Uses GPU_IDS=4,5,6,7 by default because GPUs 0-3 ran out of memory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_TIMESTAMP="${OUTPUT_TIMESTAMP:-20260502_202358}"
CONT_TIMESTAMP="${CONT_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"

TASK_IDS_ALL="${TASK_IDS_ALL:-[0,1,2,3,4,5,6,7,8,9]}"
N_EPISODES="${N_EPISODES:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-1}"
EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT:-400}"
EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT:-40}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"

ORIGINAL_POLICY_PATH="${ORIGINAL_POLICY_PATH:-/home/jongwoopark/lerobot/outputs/train/smolvla_libero_from_official_best_avg77.5_20260425_004040/checkpoints/last/pretrained_model}"

run_suite() {
  local checkpoint_label="$1"
  local n_action_steps="$2"
  local suite="$3"
  local gpu_id="$4"
  local base_output_dir="${REPO_ROOT}/outputs/eval/full_bigtoken_${checkpoint_label}_${OUTPUT_TIMESTAMP}"
  local log_path="${LOG_DIR}/continue_${checkpoint_label}_${suite}_${CONT_TIMESTAMP}.log"

  echo "[${checkpoint_label}] Starting ${suite} on GPU ${gpu_id}"
  echo "[${checkpoint_label}] Log: ${log_path}"
  echo "[${checkpoint_label}] Output root: ${base_output_dir}"

  env \
    GPU_IDS="${gpu_id}" \
    TASKS="${suite}" \
    N_EPISODES="${N_EPISODES}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED}" \
    TIMESTAMP="${CONT_TIMESTAMP}" \
    POLICY_PATH="${ORIGINAL_POLICY_PATH}" \
    CHECKPOINT_LABEL="${checkpoint_label}" \
    USE_DURATION_HEAD=false \
    N_ACTION_STEPS="${n_action_steps}" \
    OBJECT_TASK_IDS="${TASK_IDS_ALL}" \
    GOAL_TASK_IDS="${TASK_IDS_ALL}" \
    SPATIAL_TASK_IDS="${TASK_IDS_ALL}" \
    LIBERO10_TASK_IDS="${TASK_IDS_ALL}" \
    BASE_OUTPUT_DIR="${base_output_dir}" \
    WRITE_OVERLAY_SUMMARY=false \
    bash "${SCRIPT_DIR}/eval_duration_overlay_bigtoken.sh" \
    > "${log_path}" 2>&1
}

write_summary() {
  local checkpoint_label="$1"
  local base_output_dir="${REPO_ROOT}/outputs/eval/full_bigtoken_${checkpoint_label}_${OUTPUT_TIMESTAMP}"

  BASE_OUTPUT_DIR="${base_output_dir}" \
  EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT}" \
  EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT}" \
  python - <<'PY'
import json
import os
import sys
from pathlib import Path

base = Path(os.environ["BASE_OUTPUT_DIR"])
expected_episode_count = int(os.environ["EXPECTED_EPISODE_COUNT"])
expected_video_count = int(os.environ["EXPECTED_VIDEO_COUNT"])
eval_infos = sorted(base.glob("*/eval_info.json"))

per_task = []
per_group = {}
episodes = []
video_paths = []
for path in eval_infos:
    info = json.loads(path.read_text())
    for task_info in info.get("per_task", []):
        per_task.append(task_info)
        for episode in task_info.get("metrics", {}).get("episode_metrics", []):
            episode = dict(episode)
            episode["task_group"] = task_info.get("task_group")
            episode["task_id"] = task_info.get("task_id")
            episode["task_prompt"] = task_info.get("task_prompt") or episode.get("display_metrics", {}).get("task_prompt")
            episodes.append(episode)
    per_group.update(info.get("per_group", {}))
    video_paths.extend(info.get("overall", {}).get("video_paths", []))


def mean(values):
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def final_duration(duration_sequence):
    if not duration_sequence:
        return None
    return duration_sequence[-1]


successes = [bool(ep.get("success")) for ep in episodes]
display_metrics = [ep.get("display_metrics", {}) for ep in episodes]
summary = {
    "per_task": per_task,
    "per_group": per_group,
    "overall": {
        "n_episodes": len(episodes),
        "pc_success": (sum(successes) / len(successes) * 100) if successes else None,
        "mean_final_duration": mean([final_duration(m.get("duration")) for m in display_metrics]),
        "mean_inference_calls": mean([m.get("inference_calls") for m in display_metrics]),
        "mean_duration": mean([m.get("mean_duration") for m in display_metrics]),
        "mean_total_time_s": mean([m.get("total_time_s") for m in display_metrics]),
        "task_prompts": sorted({ep.get("task_prompt") for ep in episodes if ep.get("task_prompt")}),
        "episode_metrics": episodes,
        "video_paths": video_paths,
        "n_video_paths": len(video_paths),
    },
}

summary_path = base / "overlay_eval_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(f"Wrote combined overlay summary: {summary_path}")
print(f"Collected {len(episodes)} episodes and {len(video_paths)} video paths.")
if expected_episode_count >= 0 and len(episodes) != expected_episode_count:
    print(
        f"Expected {expected_episode_count} episodes, but collected {len(episodes)}.",
        file=sys.stderr,
    )
    sys.exit(2)
if expected_video_count >= 0 and len(video_paths) != expected_video_count:
    print(
        f"Expected {expected_video_count} videos, but collected {len(video_paths)} video paths.",
        file=sys.stderr,
    )
    sys.exit(2)
PY
}

run_config() {
  local checkpoint_label="$1"
  local n_action_steps="$2"
  shift 2
  local suites=("$@")

  echo "===== Starting continuation ${checkpoint_label} at $(date) ====="
  echo "ORIGINAL_POLICY_PATH=${ORIGINAL_POLICY_PATH}"
  echo "N_ACTION_STEPS=${n_action_steps}"
  echo "SUITES=${suites[*]}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "OUTPUT_TIMESTAMP=${OUTPUT_TIMESTAMP}"
  echo "CONT_TIMESTAMP=${CONT_TIMESTAMP}"

  if [[ ! -d "${ORIGINAL_POLICY_PATH}" ]]; then
    echo "Missing policy directory: ${ORIGINAL_POLICY_PATH}" >&2
    exit 1
  fi

  local pids=()
  for idx in "${!suites[@]}"; do
    local suite="${suites[$idx]}"
    local gpu_id="${GPU_ARRAY[$((idx % ${#GPU_ARRAY[@]}))]}"
    run_suite "${checkpoint_label}" "${n_action_steps}" "${suite}" "${gpu_id}" &
    pids+=("$!")
  done

  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done

  if [[ "${status}" -ne 0 ]]; then
    echo "Continuation ${checkpoint_label} failed; not starting later configs." >&2
    exit "${status}"
  fi

  write_summary "${checkpoint_label}"
  echo "===== Finished continuation ${checkpoint_label} at $(date) ====="
}

echo "OUTPUT_TIMESTAMP=${OUTPUT_TIMESTAMP}"
echo "CONT_TIMESTAMP=${CONT_TIMESTAMP}"
echo "GPU_IDS=${GPU_IDS}"
echo "TASK_IDS_ALL=${TASK_IDS_ALL}"
echo "N_EPISODES=${N_EPISODES}"
echo "MAX_EPISODES_RENDERED=${MAX_EPISODES_RENDERED}"

run_config \
  "smolvla_official_best_avg77_5_n_action_steps_8" \
  "8" \
  libero_object libero_goal

run_config \
  "smolvla_official_best_avg77_5_n_action_steps_12" \
  "12" \
  libero_object libero_goal libero_spatial libero_10

run_config \
  "smolvla_official_best_avg77_5_n_action_steps_16" \
  "16" \
  libero_object libero_goal libero_spatial libero_10

echo "R1 continuation finished at $(date)."
