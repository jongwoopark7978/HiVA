#!/usr/bin/env bash
set -euo pipefail

# Resume the mean-weighted HiVA coefficient eval after the first S1 run
# completed object/spatial/libero_10 but failed before writing libero_goal.
# This script runs the missing S1 libero_goal first, writes the S1 combined
# summary, then runs full S2/S4/S8 sequentially via the main launcher.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"

TASK_IDS_ALL="${TASK_IDS_ALL:-[0,1,2,3,4,5,6,7,8,9]}"
N_EPISODES="${N_EPISODES:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-1}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-1}"
EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT:-400}"
EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT:-40}"

S1_GOAL_GPU_ID="${S1_GOAL_GPU_ID:-0}"
S1_GOAL_TIMESTAMP="${S1_GOAL_TIMESTAMP:-20260504_232610}"
S1_POLICY_PATH="${S1_POLICY_PATH:-/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_bigflow_b160_full_s1_rerun_20260504_125652/checkpoints/last/pretrained_model}"
S1_CHECKPOINT_LABEL="${S1_CHECKPOINT_LABEL:-smolvla_hiva_coeff_bigflow_b160_full_s1_rerun_20260504_125652_10eps_bs4}"
S1_BASE_OUTPUT_DIR="${S1_BASE_OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/full_bigtoken_${S1_CHECKPOINT_LABEL}_${S1_GOAL_TIMESTAMP}}"

DATA_ROOT="${DATA_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/libero_lerobot_v3_lerobotkeys}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d6_10_15_k6_all_episodes.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d6_10_15_k6_all_episodes.summary.json}"
CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
source "/home/jongwoopark/lerobot/server_scripts/common_hf_cache.sh"
setup_hf_datasets_cache

N_ACTION_STEPS="${N_ACTION_STEPS:-15}"
CHUNK_SIZE="${CHUNK_SIZE:-15}"
NUM_STEPS="${NUM_STEPS:-10}"
RENAME_MAP='{"observation.images.image":"observation.images.agentview","observation.images.image2":"observation.images.wrist"}'

wait_for_gpu() {
  local gpu_id="$1"
  local free_min_mib="${2:-10000}"
  while true; do
    local free_mib
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}" | awk 'NR == 1 {gsub(/ /, ""); print $1}')"
    echo "gpu${gpu_id}=${free_mib}MiB_free (need >= ${free_min_mib} MiB)"
    if (( free_mib >= free_min_mib )); then
      return
    fi
    sleep 60
  done
}

run_s1_goal() {
  local safe_task_ids="${TASK_IDS_ALL//[^0-9A-Za-z_-]/_}"
  local run_name="hiva_coeff_mean_bigtoken_${S1_CHECKPOINT_LABEL}_libero_goal_taskids_${safe_task_ids}_resume_${TIMESTAMP}"
  local output_dir="${S1_BASE_OUTPUT_DIR}/libero_goal_taskids_${safe_task_ids}"
  local log_path="${LOG_DIR}/${run_name}.log"
  local mem_csv="${LOG_DIR}/${run_name}_gpu${S1_GOAL_GPU_ID}_memory.csv"

  echo "===== Running missing S1 libero_goal at $(date) ====="
  echo "S1_POLICY_PATH=${S1_POLICY_PATH}"
  echo "S1_BASE_OUTPUT_DIR=${S1_BASE_OUTPUT_DIR}"
  echo "S1_GOAL_GPU_ID=${S1_GOAL_GPU_ID}"
  echo "S1 goal log: ${log_path}"
  echo "S1 goal memory CSV: ${mem_csv}"

  wait_for_gpu "${S1_GOAL_GPU_ID}" 10000

  (
    echo "timestamp,gpu${S1_GOAL_GPU_ID}_used_mib"
    while true; do
      printf '%s,' "$(date '+%Y-%m-%d %H:%M:%S')"
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${S1_GOAL_GPU_ID}" | awk 'NR == 1 {gsub(/ /, ""); print $1}'
      sleep 2
    done
  ) > "${mem_csv}" &
  local monitor_pid="$!"

  set +e
  CUDA_VISIBLE_DEVICES="${S1_GOAL_GPU_ID}" \
  MUJOCO_EGL_DEVICE_ID="${S1_GOAL_GPU_ID}" \
  lerobot-eval \
    --policy.path="${S1_POLICY_PATH}" \
    --policy.device=cuda \
    --policy.num_steps="${NUM_STEPS}" \
    --policy.chunk_size="${CHUNK_SIZE}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.use_duration_head=false \
    --policy.hiva_coeff_sidecar_path="${HIVA_COEFF_SIDECAR}" \
    --policy.hiva_coeff_sidecar_summary_path="${HIVA_COEFF_SUMMARY}" \
    --env.type=libero \
    --env.task=libero_goal \
    --env.task_ids="${TASK_IDS_ALL}" \
    --env.control_mode=relative \
    --env.max_parallel_tasks="${MAX_PARALLEL_TASKS}" \
    --eval.batch_size="${EVAL_BATCH_SIZE}" \
    --eval.n_episodes="${N_EPISODES}" \
    --eval.max_episodes_rendered="${MAX_EPISODES_RENDERED}" \
    --rename_map="${RENAME_MAP}" \
    --output_dir="${output_dir}" \
    --job_name="${run_name}" \
    > "${log_path}" 2>&1
  local status="$?"
  set -e

  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true

  local peak_mem="NA"
  if [[ -s "${mem_csv}" ]]; then
    peak_mem="$(tail -n +2 "${mem_csv}" | awk -F, 'max < $2 {max=$2} END {if (max == "") print "NA"; else print max}')"
  fi
  echo "S1 libero_goal status=${status}; peak gpu${S1_GOAL_GPU_ID} memory=${peak_mem} MiB"
  if [[ "${status}" -ne 0 ]]; then
    tail -n 120 "${log_path}" || true
    exit "${status}"
  fi
}

write_summary() {
  BASE_OUTPUT_DIR="${S1_BASE_OUTPUT_DIR}" \
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

run_remaining_s2_s4_s8() {
  echo "===== Running remaining S2/S4/S8 at $(date) ====="
  START_INDEX=1 \
  SUITE_GPU_IDS="${SUITE_GPU_IDS:-0,2,1,3}" \
  TIMESTAMP="${REMAINING_TIMESTAMP:-${TIMESTAMP}}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  N_EPISODES="${N_EPISODES}" \
  MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED}" \
  bash "${SCRIPT_DIR}/eval_hiva_coeff_bigflow_mean_s1_s2_s4_s8_10eps_bigtoken.sh"
}

echo "TIMESTAMP=${TIMESTAMP}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "TASK_IDS_ALL=${TASK_IDS_ALL}"
echo "S1_BASE_OUTPUT_DIR=${S1_BASE_OUTPUT_DIR}"

run_s1_goal
write_summary
run_remaining_s2_s4_s8

echo "===== Resume script finished at $(date) ====="
