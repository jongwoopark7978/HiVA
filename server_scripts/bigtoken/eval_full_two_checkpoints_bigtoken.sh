#!/usr/bin/env bash
set -euo pipefail

# Full LIBERO benchmark eval on bigtoken for two checkpoints, run sequentially:
#   1. HiVA duration-token SmolVLA
#   2. vanilla SmolVLA
#
# Within each checkpoint, the four LIBERO suites run in parallel on GPU 0-3.
# Each suite evaluates task_ids [0..9] with N_EPISODES=10 per task by default,
# matching the existing SmolVLA LIBERO eval scripts in this repo.
#
# Example:
#   nohup setsid bash server_scripts/bigtoken/eval_full_two_checkpoints_bigtoken.sh \
#     > outputs/eval_logs/full_eval_two_checkpoints_bigtoken_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"

TASK_IDS_ALL="${TASK_IDS_ALL:-[0,1,2,3,4,5,6,7,8,9]}"
N_EPISODES="${N_EPISODES:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
SUITES=(libero_object libero_goal libero_spatial libero_10)

DURATION_POLICY_PATH="${DURATION_POLICY_PATH:-/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_duration_token_bigcornea_full_s8_20260430_152419/checkpoints/last/pretrained_model}"
VANILLA_POLICY_PATH="${VANILLA_POLICY_PATH:-/home/jongwoopark/lerobot/outputs/train/smolvla_libero_from_official_best_avg77.5_20260425_004040/checkpoints/last/pretrained_model}"

run_suite() {
  local checkpoint_label="$1"
  local policy_path="$2"
  local use_duration_head="$3"
  local n_action_steps="$4"
  local suite="$5"
  local gpu_id="$6"
  local base_output_dir="${REPO_ROOT}/outputs/eval/full_bigtoken_${checkpoint_label}_${TIMESTAMP}"
  local log_path="${LOG_DIR}/full_bigtoken_${checkpoint_label}_${suite}_${TIMESTAMP}.log"

  echo "[${checkpoint_label}] Starting ${suite} on GPU ${gpu_id}"
  echo "[${checkpoint_label}] Log: ${log_path}"
  echo "[${checkpoint_label}] Output root: ${base_output_dir}"

  env \
    GPU_IDS="${gpu_id}" \
    TASKS="${suite}" \
    N_EPISODES="${N_EPISODES}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    TIMESTAMP="${TIMESTAMP}" \
    POLICY_PATH="${policy_path}" \
    CHECKPOINT_LABEL="${checkpoint_label}" \
    USE_DURATION_HEAD="${use_duration_head}" \
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
  local base_output_dir="${REPO_ROOT}/outputs/eval/full_bigtoken_${checkpoint_label}_${TIMESTAMP}"

  BASE_OUTPUT_DIR="${base_output_dir}" python - <<'PY'
import json
import os
from pathlib import Path

base = Path(os.environ["BASE_OUTPUT_DIR"])
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
    },
}

summary_path = base / "overlay_eval_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(f"Wrote combined overlay summary: {summary_path}")
PY
}

run_checkpoint() {
  local checkpoint_label="$1"
  local policy_path="$2"
  local use_duration_head="$3"
  local n_action_steps="$4"

  echo "===== Starting checkpoint ${checkpoint_label} at $(date) ====="
  echo "POLICY_PATH=${policy_path}"
  echo "USE_DURATION_HEAD=${use_duration_head}"
  echo "N_ACTION_STEPS=${n_action_steps}"
  echo "TASK_IDS_ALL=${TASK_IDS_ALL}"
  echo "N_EPISODES=${N_EPISODES}"
  echo "GPU_IDS=${GPU_IDS}"

  local pids=()
  for idx in "${!SUITES[@]}"; do
    local suite="${SUITES[$idx]}"
    local gpu_id="${GPU_ARRAY[$((idx % ${#GPU_ARRAY[@]}))]}"
    run_suite "${checkpoint_label}" "${policy_path}" "${use_duration_head}" "${n_action_steps}" "${suite}" "${gpu_id}" &
    pids+=("$!")
  done

  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done

  if [[ "${status}" -ne 0 ]]; then
    echo "Checkpoint ${checkpoint_label} failed; not starting later checkpoints." >&2
    exit "${status}"
  fi
  write_summary "${checkpoint_label}"
  echo "===== Finished checkpoint ${checkpoint_label} at $(date) ====="
}

run_checkpoint \
  "smolvla_hiva_duration_token_bigcornea_full_s8_full" \
  "${DURATION_POLICY_PATH}" \
  "true" \
  "8"

run_checkpoint \
  "smolvla_official_best_avg77_5_vanilla_full" \
  "${VANILLA_POLICY_PATH}" \
  "false" \
  "1"

echo "All full evaluations finished at $(date)."
