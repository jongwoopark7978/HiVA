#!/usr/bin/env bash
set -euo pipefail

# Partial LIBERO evaluation for two coefficient-HiVA checkpoints on bigtoken.
# Uses only physical GPU7. Suites and checkpoints run sequentially.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"

PHYSICAL_GPU_ID="${PHYSICAL_GPU_ID:-7}"
# On this bigtoken host, MuJoCo's EGL device id that maps to physical GPU7 is 4.
# CUDA sees physical GPU7 first, so policy.device=cuda still uses GPU7.
CUDA_VISIBLE_DEVICES_FOR_EVAL="${CUDA_VISIBLE_DEVICES_FOR_EVAL:-7,4}"
MUJOCO_EGL_DEVICE_ID_FOR_EVAL="${MUJOCO_EGL_DEVICE_ID_FOR_EVAL:-4}"

TASK_IDS_ALL="${TASK_IDS_ALL:-[0,1,2,3,4,5,6,7,8,9]}"
N_EPISODES="${N_EPISODES:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-1}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-1}"
EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT:-400}"
EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT:-40}"
SUITES=(libero_object libero_goal libero_spatial libero_10)

DATA_ROOT="${DATA_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/libero_lerobot_v3_lerobotkeys}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d6_10_15_k6_all_episodes.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d6_10_15_k6_all_episodes.summary.json}"
CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
source "/home/jongwoopark/lerobot/server_scripts/common_hf_cache.sh"
setup_hf_datasets_cache

N_ACTION_STEPS="${N_ACTION_STEPS:-15}"
CHUNK_SIZE="${CHUNK_SIZE:-15}"
NUM_STEPS="${NUM_STEPS:-10}"
RENAME_MAP='{"observation.images.image":"observation.images.agentview","observation.images.image2":"observation.images.wrist"}'

POLICY_PATHS=(
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_cleaner_suffix_bigflow_job1_lambda1p0_b160_s4_20260505_143646_20260505_143647_001226288_pid466408/checkpoints/last/pretrained_model"
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_cleaner_suffix_bigcornea_J2_sigma0p25_w1p0_b80_s1_j1j2_b80_w1_20260505_023954/checkpoints/last/pretrained_model"
)

CHECKPOINT_LABELS=(
  "smolvla_hiva_coeff_cleaner_suffix_bigflow_job1_lambda1p0_b160_s4_20260505_143647_10eps_bs4_gpu7"
  "smolvla_hiva_coeff_cleaner_suffix_bigcornea_J2_sigma0p25_w1p0_b80_s1_20260505_023954_10eps_bs4_gpu7"
)

task_ids_for() {
  case "$1" in
    libero_object) echo "${OBJECT_TASK_IDS:-${TASK_IDS_ALL}}" ;;
    libero_goal) echo "${GOAL_TASK_IDS:-${TASK_IDS_ALL}}" ;;
    libero_spatial) echo "${SPATIAL_TASK_IDS:-${TASK_IDS_ALL}}" ;;
    libero_10) echo "${LIBERO10_TASK_IDS:-${TASK_IDS_ALL}}" ;;
    *) echo "${TASK_IDS:-${TASK_IDS_ALL}}" ;;
  esac
}

run_suite() {
  local policy_path="$1"
  local checkpoint_label="$2"
  local base_output_dir="$3"
  local suite="$4"
  local task_ids
  task_ids="$(task_ids_for "${suite}")"
  local safe_task_ids="${task_ids//[^0-9A-Za-z_-]/_}"
  local run_name="hiva_coeff_bigtoken_${checkpoint_label}_${suite}_taskids_${safe_task_ids}_${TIMESTAMP}"
  local output_dir="${base_output_dir}/${suite}_taskids_${safe_task_ids}"
  local log_path="${LOG_DIR}/${run_name}.log"

  echo "[${checkpoint_label}] Starting ${suite} task_ids=${task_ids} on physical GPU ${PHYSICAL_GPU_ID}"
  echo "[${checkpoint_label}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_FOR_EVAL}, MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID_FOR_EVAL}"
  echo "[${checkpoint_label}] Log: ${log_path}"
  echo "[${checkpoint_label}] Output dir: ${output_dir}"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_FOR_EVAL}" \
  MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID_FOR_EVAL}" \
  lerobot-eval \
    --policy.path="${policy_path}" \
    --policy.device=cuda \
    --policy.num_steps="${NUM_STEPS}" \
    --policy.chunk_size="${CHUNK_SIZE}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.use_duration_head=false \
    --policy.hiva_coeff_sidecar_path="${HIVA_COEFF_SIDECAR}" \
    --policy.hiva_coeff_sidecar_summary_path="${HIVA_COEFF_SUMMARY}" \
    --env.type=libero \
    --env.task="${suite}" \
    --env.task_ids="${task_ids}" \
    --env.control_mode=relative \
    --env.max_parallel_tasks="${MAX_PARALLEL_TASKS}" \
    --eval.batch_size="${EVAL_BATCH_SIZE}" \
    --eval.n_episodes="${N_EPISODES}" \
    --eval.max_episodes_rendered="${MAX_EPISODES_RENDERED}" \
    --rename_map="${RENAME_MAP}" \
    --output_dir="${output_dir}" \
    --job_name="${run_name}" \
    > "${log_path}" 2>&1
}

write_summary() {
  local base_output_dir="$1"
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

echo "===== GPU7-only coeff HiVA eval started at $(date) ====="
echo "TIMESTAMP=${TIMESTAMP}"
echo "PHYSICAL_GPU_ID=${PHYSICAL_GPU_ID}"
echo "CUDA_VISIBLE_DEVICES_FOR_EVAL=${CUDA_VISIBLE_DEVICES_FOR_EVAL}"
echo "MUJOCO_EGL_DEVICE_ID_FOR_EVAL=${MUJOCO_EGL_DEVICE_ID_FOR_EVAL}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "TASK_IDS_ALL=${TASK_IDS_ALL}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Missing DATA_ROOT directory: ${DATA_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${HIVA_COEFF_SIDECAR}" ]]; then
  echo "Missing HIVA_COEFF_SIDECAR file: ${HIVA_COEFF_SIDECAR}" >&2
  exit 1
fi
if [[ ! -f "${HIVA_COEFF_SUMMARY}" ]]; then
  echo "Missing HIVA_COEFF_SUMMARY file: ${HIVA_COEFF_SUMMARY}" >&2
  exit 1
fi

for idx in "${!POLICY_PATHS[@]}"; do
  policy_path="${POLICY_PATHS[$idx]}"
  checkpoint_label="${CHECKPOINT_LABELS[$idx]}"
  base_output_dir="${REPO_ROOT}/outputs/eval/full_bigtoken_${checkpoint_label}_${TIMESTAMP}"

  echo "===== Starting checkpoint ${checkpoint_label} at $(date) ====="
  echo "POLICY_PATH=${policy_path}"
  echo "BASE_OUTPUT_DIR=${base_output_dir}"
  if [[ ! -d "${policy_path}" ]]; then
    echo "Missing policy directory: ${policy_path}" >&2
    exit 1
  fi

  for suite in "${SUITES[@]}"; do
    run_suite "${policy_path}" "${checkpoint_label}" "${base_output_dir}" "${suite}"
  done
  write_summary "${base_output_dir}"
  echo "===== Finished checkpoint ${checkpoint_label} at $(date) ====="
done

echo "===== GPU7-only coeff HiVA eval finished at $(date) ====="
