#!/usr/bin/env bash
set -euo pipefail

# Full LIBERO eval for two original SmolVLA S=0.5 checkpoints on bigtoken.
#
# Each checkpoint evaluates all 4 LIBERO suites, task_ids [0..9], with
# N_EPISODES=50 per task and eval.batch_size=4 by default.
# The 004375 checkpoint uses GPU0-3; the 005000 checkpoint uses GPU4-7.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"

TASK_IDS_ALL="${TASK_IDS_ALL:-[0,1,2,3,4,5,6,7,8,9]}"
N_EPISODES="${N_EPISODES:-50}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-1}"
EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT:-2000}"
EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT:-40}"
N_ACTION_STEPS="${N_ACTION_STEPS:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/jongwoopark/hf_datasets_cache}"
mkdir -p "${HF_DATASETS_CACHE}"

TRAIN_DIR="${TRAIN_DIR:-/home/jongwoopark/lerobot/outputs/train/smolvla_original_bigcornea_s0p5_b64_g8_20260515_223843_888072651_pid2567557}"
CKPT4375_POLICY_PATH="${CKPT4375_POLICY_PATH:-${TRAIN_DIR}/checkpoints/004375/pretrained_model}"
CKPT5000_POLICY_PATH="${CKPT5000_POLICY_PATH:-${TRAIN_DIR}/checkpoints/005000/pretrained_model}"

SUITES=(libero_object libero_goal libero_spatial libero_10)

run_suite() {
  local checkpoint_label="$1"
  local policy_path="$2"
  local gpu_id="$3"
  local suite="$4"
  local base_output_dir="${REPO_ROOT}/outputs/eval/full_bigtoken_${checkpoint_label}_${TIMESTAMP}"
  local log_path="${LOG_DIR}/full_bigtoken_${checkpoint_label}_${suite}_${TIMESTAMP}.log"

  echo "[${checkpoint_label}] Starting ${suite} on GPU ${gpu_id}"
  echo "[${checkpoint_label}] Log: ${log_path}"
  echo "[${checkpoint_label}] Output root: ${base_output_dir}"

  env \
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
    GPU_IDS="${gpu_id}" \
    TASKS="${suite}" \
    N_EPISODES="${N_EPISODES}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED}" \
    TIMESTAMP="${TIMESTAMP}" \
    POLICY_PATH="${policy_path}" \
    CHECKPOINT_LABEL="${checkpoint_label}" \
    USE_DURATION_HEAD=false \
    N_ACTION_STEPS="${N_ACTION_STEPS}" \
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
  local base_output_dir="${REPO_ROOT}/outputs/eval/full_bigtoken_${checkpoint_label}_${TIMESTAMP}"

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
    print(f"Expected {expected_episode_count} episodes, but collected {len(episodes)}.", file=sys.stderr)
    sys.exit(2)
if expected_video_count >= 0 and len(video_paths) != expected_video_count:
    print(f"Expected {expected_video_count} videos, but collected {len(video_paths)} video paths.", file=sys.stderr)
    sys.exit(2)
PY
}

run_checkpoint() {
  local checkpoint_label="$1"
  local policy_path="$2"
  local gpu_ids="$3"
  IFS=',' read -r -a gpu_array <<< "${gpu_ids}"

  echo "===== Starting checkpoint ${checkpoint_label} at $(date) ====="
  echo "POLICY_PATH=${policy_path}"
  echo "GPU_IDS=${gpu_ids}"
  echo "N_EPISODES=${N_EPISODES}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
  echo "MAX_EPISODES_RENDERED=${MAX_EPISODES_RENDERED}"
  echo "N_ACTION_STEPS=${N_ACTION_STEPS}"

  if [[ ! -d "${policy_path}" ]]; then
    echo "Missing policy directory: ${policy_path}" >&2
    return 1
  fi

  local pids=()
  for idx in "${!SUITES[@]}"; do
    local suite="${SUITES[$idx]}"
    local gpu_id="${gpu_array[$((idx % ${#gpu_array[@]}))]}"
    run_suite "${checkpoint_label}" "${policy_path}" "${gpu_id}" "${suite}" &
    pids+=("$!")
  done

  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done

  if [[ "${status}" -ne 0 ]]; then
    echo "Checkpoint ${checkpoint_label} failed." >&2
    return "${status}"
  fi

  write_summary "${checkpoint_label}"
  echo "===== Finished checkpoint ${checkpoint_label} at $(date) ====="
}

echo "===== Original SmolVLA S=0.5 full eval started at $(date) ====="
echo "TIMESTAMP=${TIMESTAMP}"
echo "TRAIN_DIR=${TRAIN_DIR}"
echo "CKPT4375_POLICY_PATH=${CKPT4375_POLICY_PATH}"
echo "CKPT5000_POLICY_PATH=${CKPT5000_POLICY_PATH}"
echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"

run_checkpoint \
  "smolvla_original_bigcornea_s0p5_b64_g8_20260515_223843_ckpt004375_50eps_bs${EVAL_BATCH_SIZE}" \
  "${CKPT4375_POLICY_PATH}" \
  "0,1,2,3" &
pid_4375="$!"

run_checkpoint \
  "smolvla_original_bigcornea_s0p5_b64_g8_20260515_223843_ckpt005000_50eps_bs${EVAL_BATCH_SIZE}" \
  "${CKPT5000_POLICY_PATH}" \
  "4,5,6,7" &
pid_5000="$!"

status=0
for pid in "${pid_4375}" "${pid_5000}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "One or more S=0.5 original SmolVLA full evals failed." >&2
  exit "${status}"
fi

echo "===== Original SmolVLA S=0.5 full eval finished at $(date) ====="
