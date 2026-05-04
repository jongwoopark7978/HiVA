#!/usr/bin/env bash
set -euo pipefail

# Evaluate the clean/noisy bigflow S1 duration checkpoint on bigtoken.
#
# This runs all 4 LIBERO suites, task_ids [0..9], with N_EPISODES=10 per task
# by default. Suites run concurrently across GPU_IDS=0,1,2,3.
#
# Video policy: render only the first episode for each task in each suite.
# With the defaults, that is 4 suites * 10 tasks * 1 video = 40 videos.
#
# Example:
#   nohup setsid bash server_scripts/bigtoken/eval_s1_duration_clean_noisy_bigflow_parallel_bigtoken.sh \
#     > outputs/eval_logs/eval_s1_duration_clean_noisy_bigflow_parallel_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"

TASK_IDS_ALL="${TASK_IDS_ALL:-[0,1,2,3,4,5,6,7,8,9]}"
N_EPISODES="${N_EPISODES:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-1}"
EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT:-40}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
SUITES=(libero_object libero_goal libero_spatial libero_10)

POLICY_PATH="${POLICY_PATH:-/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_duration_clean_noisy_bigflow_s1_20260430_214831/checkpoints/last/pretrained_model}"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-smolvla_hiva_duration_clean_noisy_bigflow_s1}"

run_suite() {
  local suite="$1"
  local gpu_id="$2"
  local base_output_dir="${REPO_ROOT}/outputs/eval/full_bigtoken_${CHECKPOINT_LABEL}_${TIMESTAMP}"
  local log_path="${LOG_DIR}/full_bigtoken_${CHECKPOINT_LABEL}_${suite}_${TIMESTAMP}.log"

  echo "[${CHECKPOINT_LABEL}] Starting ${suite} on GPU ${gpu_id}"
  echo "[${CHECKPOINT_LABEL}] Log: ${log_path}"
  echo "[${CHECKPOINT_LABEL}] Output root: ${base_output_dir}"

  env \
    GPU_IDS="${gpu_id}" \
    TASKS="${suite}" \
    N_EPISODES="${N_EPISODES}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED}" \
    TIMESTAMP="${TIMESTAMP}" \
    POLICY_PATH="${POLICY_PATH}" \
    CHECKPOINT_LABEL="${CHECKPOINT_LABEL}" \
    USE_DURATION_HEAD=true \
    N_ACTION_STEPS=8 \
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
  local base_output_dir="${REPO_ROOT}/outputs/eval/full_bigtoken_${CHECKPOINT_LABEL}_${TIMESTAMP}"

  BASE_OUTPUT_DIR="${base_output_dir}" EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT}" python - <<'PY'
import json
import os
import sys
from pathlib import Path

base = Path(os.environ["BASE_OUTPUT_DIR"])
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
    overall = info.get("overall", {})
    video_paths.extend(overall.get("video_paths", []))


def mean(values):
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def final_duration(duration_sequence):
    if not duration_sequence:
        return None
    return duration_sequence[-1]


successes = [bool(ep.get("success")) for ep in episodes]
display_metrics = [ep.get("display_metrics", {}) for ep in episodes]
task_prompts = sorted({ep.get("task_prompt") for ep in episodes if ep.get("task_prompt")})
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
        "task_prompts": task_prompts,
        "episode_metrics": episodes,
        "video_paths": video_paths,
        "n_video_paths": len(video_paths),
    },
}

summary_path = base / "overlay_eval_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(f"Wrote combined overlay summary: {summary_path}")
print(f"Collected {len(video_paths)} video paths.")
if expected_video_count >= 0 and len(video_paths) != expected_video_count:
    print(
        f"Expected {expected_video_count} videos, but collected {len(video_paths)} video paths.",
        file=sys.stderr,
    )
    sys.exit(2)
PY
}

echo "===== Starting checkpoint ${CHECKPOINT_LABEL} at $(date) ====="
echo "POLICY_PATH=${POLICY_PATH}"
echo "TASK_IDS_ALL=${TASK_IDS_ALL}"
echo "N_EPISODES=${N_EPISODES}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "MAX_EPISODES_RENDERED=${MAX_EPISODES_RENDERED}"
echo "EXPECTED_VIDEO_COUNT=${EXPECTED_VIDEO_COUNT}"
echo "GPU_IDS=${GPU_IDS}"

if [[ ! -d "${POLICY_PATH}" ]]; then
  echo "Missing POLICY_PATH directory: ${POLICY_PATH}" >&2
  exit 1
fi

pids=()
for idx in "${!SUITES[@]}"; do
  suite="${SUITES[$idx]}"
  gpu_id="${GPU_ARRAY[$((idx % ${#GPU_ARRAY[@]}))]}"
  run_suite "${suite}" "${gpu_id}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "Checkpoint ${CHECKPOINT_LABEL} failed." >&2
  exit "${status}"
fi

write_summary
echo "===== Finished checkpoint ${CHECKPOINT_LABEL} at $(date) ====="
