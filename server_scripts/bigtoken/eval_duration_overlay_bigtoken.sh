#!/usr/bin/env bash
set -euo pipefail

# Generate LIBERO eval videos with the built-in policy-call overlay:
#   D=<execution duration> INF=<policy inference call count>
#   MD=<mean duration per call> TT=<episode total time> <SC|FA>
#
# For duration-token policies, set USE_DURATION_HEAD=true. For vanilla SmolVLA
# policies, set USE_DURATION_HEAD=false; the overlay will report the fixed
# execution horizon used by the policy.
#
# Defaults render 2 episodes from task_id 0 of each requested LIBERO suite:
#   libero_object, libero_goal, libero_spatial, libero_10
#
# Example:
#   POLICY_PATH=/home/jongwoopark/lerobot/outputs/train/<run>/checkpoints/last/pretrained_model \
#   CHECKPOINT_LABEL=my_checkpoint \
#   GPU_IDS=0,1,2,3 \
#   bash server_scripts/bigtoken/eval_duration_overlay_bigtoken.sh
#
# Dataset and duration sidecar are intentionally pointed at the bigcornea NFS
# copy so evaluation on bigtoken uses the same metadata source as the
# bigcornea training/eval setup.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

: "${POLICY_PATH:?Set POLICY_PATH to a checkpoint pretrained_model directory.}"

CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-$(basename "$(dirname "$(dirname "${POLICY_PATH}")")")}"

DATA_ROOT="${DATA_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/libero_lerobot_v3_lerobotkeys}"
SIDECAR="${SIDECAR:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/libero_duration_sidecar_all_episodes.parquet}"
SIDECAR_SUMMARY="${SIDECAR_SUMMARY:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/libero_duration_sidecar_all_episodes.summary.json}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/jongwoo_hf_datasets_cache}"
mkdir -p "${HF_DATASETS_CACHE}"

GPU_IDS="${GPU_IDS:-0}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"

N_EPISODES="${N_EPISODES:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-10}"
TASKS=(${TASKS:-libero_object libero_goal libero_spatial libero_10})
USE_DURATION_HEAD="${USE_DURATION_HEAD:-true}"
DURATION_TRAIN_REUSE_PREFIX_CACHE="${DURATION_TRAIN_REUSE_PREFIX_CACHE:-true}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
WRITE_OVERLAY_SUMMARY="${WRITE_OVERLAY_SUMMARY:-true}"

# Restrict to one representative task from each suite by default. Override any
# of these to render a different task id, e.g. SPATIAL_TASK_IDS='[3]'.
OBJECT_TASK_IDS="${OBJECT_TASK_IDS:-[0]}"
GOAL_TASK_IDS="${GOAL_TASK_IDS:-[0]}"
SPATIAL_TASK_IDS="${SPATIAL_TASK_IDS:-[0]}"
LIBERO10_TASK_IDS="${LIBERO10_TASK_IDS:-[0]}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/eval_duration_overlay_bigtoken_${CHECKPOINT_LABEL}_${TIMESTAMP}.log"

exec > >(tee -a "${MASTER_LOG}") 2>&1

RENAME_MAP='{"observation.images.image":"observation.images.agentview","observation.images.image2":"observation.images.wrist"}'
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/duration_overlay_bigtoken_${CHECKPOINT_LABEL}_${TIMESTAMP}}"

task_ids_for() {
  case "$1" in
    libero_object) echo "${OBJECT_TASK_IDS}" ;;
    libero_goal) echo "${GOAL_TASK_IDS}" ;;
    libero_spatial) echo "${SPATIAL_TASK_IDS}" ;;
    libero_10) echo "${LIBERO10_TASK_IDS}" ;;
    *) echo "${TASK_IDS:-[0]}" ;;
  esac
}

run_task() {
  local gpu_id="$1"
  local task_name="$2"
  local task_ids="$3"
  local safe_task_ids="${task_ids//[^0-9A-Za-z_-]/_}"
  local run_name="duration_overlay_bigtoken_${CHECKPOINT_LABEL}_${task_name}_taskids_${safe_task_ids}_${TIMESTAMP}"
  local output_dir="${BASE_OUTPUT_DIR}/${task_name}_taskids_${safe_task_ids}"
  local task_log="${LOG_DIR}/${run_name}.log"

  echo "Starting ${task_name} task_ids=${task_ids} on GPU ${gpu_id}"
  echo "Task log: ${task_log}"
  echo "Output dir: ${output_dir}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  lerobot-eval \
    --policy.path="${POLICY_PATH}" \
    --policy.device=cuda \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.use_duration_head="${USE_DURATION_HEAD}" \
    --policy.duration_train_reuse_prefix_cache="${DURATION_TRAIN_REUSE_PREFIX_CACHE}" \
    --policy.duration_sidecar_path="${SIDECAR}" \
    --env.type=libero \
    --env.task="${task_name}" \
    --env.task_ids="${task_ids}" \
    --env.control_mode=relative \
    --env.max_parallel_tasks=1 \
    --eval.batch_size="${EVAL_BATCH_SIZE}" \
    --eval.n_episodes="${N_EPISODES}" \
    --eval.max_episodes_rendered="${MAX_EPISODES_RENDERED}" \
    --rename_map="${RENAME_MAP}" \
    --output_dir="${output_dir}" \
    --job_name="${run_name}" \
    > "${task_log}" 2>&1
}

echo "Logging to ${MASTER_LOG}"
echo "Host: $(hostname)"
echo "POLICY_PATH=${POLICY_PATH}"
echo "CHECKPOINT_LABEL=${CHECKPOINT_LABEL}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "SIDECAR=${SIDECAR}"
echo "SIDECAR_SUMMARY=${SIDECAR_SUMMARY}"
echo "GPU_IDS=${GPU_IDS}"
echo "USE_DURATION_HEAD=${USE_DURATION_HEAD}"
echo "DURATION_TRAIN_REUSE_PREFIX_CACHE=${DURATION_TRAIN_REUSE_PREFIX_CACHE}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "WRITE_OVERLAY_SUMMARY=${WRITE_OVERLAY_SUMMARY}"
echo "TASKS=${TASKS[*]}"
echo "N_EPISODES=${N_EPISODES}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "MAX_EPISODES_RENDERED=${MAX_EPISODES_RENDERED}"
echo "BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR}"

if [[ ! -d "${POLICY_PATH}" ]]; then
  echo "Missing POLICY_PATH directory: ${POLICY_PATH}" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Missing DATA_ROOT directory: ${DATA_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${SIDECAR}" ]]; then
  echo "Missing SIDECAR file: ${SIDECAR}" >&2
  exit 1
fi

for idx in "${!TASKS[@]}"; do
  gpu_id="${GPU_ARRAY[$((idx % ${#GPU_ARRAY[@]}))]}"
  task_name="${TASKS[$idx]}"
  task_ids="$(task_ids_for "${task_name}")"
  run_task "${gpu_id}" "${task_name}" "${task_ids}"
done

if [[ "${WRITE_OVERLAY_SUMMARY}" == "true" ]]; then
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR}" python - <<'PY'
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
fi

echo "Duration overlay video generation finished."
echo "Videos are under: ${BASE_OUTPUT_DIR}"
