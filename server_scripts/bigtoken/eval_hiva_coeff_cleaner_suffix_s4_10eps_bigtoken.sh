#!/usr/bin/env bash
set -euo pipefail

# Full LIBERO eval for a cleaner-suffix HiVA coefficient SmolVLA checkpoint on
# bigtoken.
#
# Evaluates all 4 LIBERO suites, task_ids [0..9], with N_EPISODES=10 per task.
# Suites run in parallel on GPU_IDS=0,1,2,3 by default. Each task renders only
# the first episode, producing 40 videos total with the defaults.
#
# The checkpoint was trained on bigcornea and stores /nfs/bigcornea/... sidecar
# paths in its config. On bigtoken, override them to the mounted
# /nfs/bigcornea.cs.stonybrook.edu/... paths below.
#
# Example:
#   nohup setsid bash server_scripts/bigtoken/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh \
#     > outputs/eval_logs/eval_hiva_coeff_cleaner_suffix_s4_10eps_$(date +%Y%m%d_%H%M%S).outer.log 2>&1 < /dev/null &

# eval bs, mem
# 4, 18
# 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"

TASK_IDS_ALL="${TASK_IDS_ALL:-[0,1,2,3,4,5,6,7,8,9]}"
N_EPISODES="${N_EPISODES:-10}"
# Run two vectorized LIBERO envs per task on each suite GPU. Keep
# MAX_PARALLEL_TASKS=1 because sharing one policy across concurrent task threads
# is risky; batch_size gives us multiple simulations on the same GPU safely.
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-1}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-1}"
EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT:-400}"
EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT:-40}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
SUITES_CSV="${SUITES_CSV:-libero_object,libero_goal,libero_spatial,libero_10}"
IFS=',' read -r -a SUITES <<< "${SUITES_CSV}"
SPLIT_LIBERO10_ACROSS_GPUS="${SPLIT_LIBERO10_ACROSS_GPUS:-0}"
STAGED_LIBERO10_AFTER_SHORT="${STAGED_LIBERO10_AFTER_SHORT:-1}"
LIBERO10_INITIAL_TASK_IDS="${LIBERO10_INITIAL_TASK_IDS:-}"

POLICY_PATH="${POLICY_PATH:-/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_cleaner_suffix_bigcornea_sigma0p25_w0p1_b80_s4_durw_sweep_b80_s4_20260504_175101/checkpoints/last/pretrained_model}"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-smolvla_hiva_coeff_cleaner_suffix_bigcornea_sigma0p25_w0p1_b80_s4_20260504_175101_10eps}"

DATA_ROOT="${DATA_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/libero_lerobot_v3_lerobotkeys}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d6_10_15_k6_all_episodes.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d6_10_15_k6_all_episodes.summary.json}"
CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/jongwoo_hf_datasets_cache}"
mkdir -p "${HF_DATASETS_CACHE}"

N_ACTION_STEPS="${N_ACTION_STEPS:-15}"
CHUNK_SIZE="${CHUNK_SIZE:-15}"
NUM_STEPS="${NUM_STEPS:-10}"
HIVA_DURATION_EXECUTION_MAP="${HIVA_DURATION_EXECUTION_MAP:-}"
HIVA_RESIDUAL_INFERENCE_WEIGHT="${HIVA_RESIDUAL_INFERENCE_WEIGHT:-}"
RENAME_MAP='{"observation.images.image":"observation.images.agentview","observation.images.image2":"observation.images.wrist"}'
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/full_bigtoken_${CHECKPOINT_LABEL}_${TIMESTAMP}}"

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
  local suite="$1"
  local gpu_id="$2"
  local task_ids="$3"
  local safe_task_ids="${task_ids//[^0-9A-Za-z_-]/_}"
  local run_name="hiva_coeff_bigtoken_${CHECKPOINT_LABEL}_${suite}_taskids_${safe_task_ids}_${TIMESTAMP}"
  local output_dir="${BASE_OUTPUT_DIR}/${suite}_taskids_${safe_task_ids}"
  local log_path="${LOG_DIR}/${run_name}.log"

  echo "[${CHECKPOINT_LABEL}] Starting ${suite} task_ids=${task_ids} on GPU ${gpu_id}"
  echo "[${CHECKPOINT_LABEL}] Log: ${log_path}"
  echo "[${CHECKPOINT_LABEL}] Output dir: ${output_dir}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  MUJOCO_EGL_DEVICE_ID="${gpu_id}" \
  lerobot-eval \
    --policy.path="${POLICY_PATH}" \
    --policy.device=cuda \
    --policy.num_steps="${NUM_STEPS}" \
    --policy.chunk_size="${CHUNK_SIZE}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.use_duration_head=false \
    --policy.hiva_coeff_sidecar_path="${HIVA_COEFF_SIDECAR}" \
    --policy.hiva_coeff_sidecar_summary_path="${HIVA_COEFF_SUMMARY}" \
    --policy.hiva_duration_execution_map="${HIVA_DURATION_EXECUTION_MAP}" \
    ${HIVA_RESIDUAL_INFERENCE_WEIGHT:+--policy.hiva_residual_inference_weight="${HIVA_RESIDUAL_INFERENCE_WEIGHT}"} \
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

split_task_ids_for_gpus() {
  TASK_IDS_TO_SPLIT="$1" N_SHARDS="${#GPU_ARRAY[@]}" python - <<'PY'
import ast
import os

task_ids = ast.literal_eval(os.environ["TASK_IDS_TO_SPLIT"])
if isinstance(task_ids, int):
    task_ids = [task_ids]
task_ids = [int(task_id) for task_id in task_ids]
n_shards = max(1, min(int(os.environ["N_SHARDS"]), len(task_ids)))
base, rem = divmod(len(task_ids), n_shards)
start = 0
for shard_idx in range(n_shards):
    size = base + int(shard_idx < rem)
    shard = task_ids[start : start + size]
    start += size
    print(f"{shard_idx}|[{','.join(str(task_id) for task_id in shard)}]")
PY
}

staged_libero10_task_ids() {
  TASK_IDS_TO_STAGE="$1" LIBERO10_INITIAL_TASK_IDS="${LIBERO10_INITIAL_TASK_IDS}" python - <<'PY'
import ast
import os

task_ids = ast.literal_eval(os.environ["TASK_IDS_TO_STAGE"])
if isinstance(task_ids, int):
    task_ids = [task_ids]
task_ids = [int(task_id) for task_id in task_ids]

initial_override = os.environ.get("LIBERO10_INITIAL_TASK_IDS", "").strip()
if initial_override:
    initial = [int(task_id) for task_id in ast.literal_eval(initial_override)]
else:
    initial = task_ids[: min(7, len(task_ids))]
initial_set = set(initial)
tail = [task_id for task_id in task_ids if task_id not in initial_set]

print(f"initial|[{','.join(str(task_id) for task_id in initial)}]")
for task_id in tail:
    print(f"tail|[{task_id}]")
PY
}

write_summary() {
  BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR}" \
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
episodes = []
video_paths = []
group_video_paths = {}
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
    group_names = list(info.get("per_group", {}).keys())
    if len(group_names) == 1:
        group_video_paths.setdefault(group_names[0], []).extend(info.get("overall", {}).get("video_paths", []))
    video_paths.extend(info.get("overall", {}).get("video_paths", []))


def mean(values):
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def final_duration(duration_sequence):
    if not duration_sequence:
        return None
    return duration_sequence[-1]


def summarize_episodes(group_episodes, group_video_paths=None):
    group_video_paths = group_video_paths or []
    successes = [bool(ep.get("success")) for ep in group_episodes]
    display_metrics = [ep.get("display_metrics", {}) for ep in group_episodes]
    return {
        "avg_sum_reward": mean([ep.get("sum_reward") for ep in group_episodes]),
        "avg_max_reward": mean([ep.get("max_reward") for ep in group_episodes]),
        "pc_success": (sum(successes) / len(successes) * 100) if successes else None,
        "n_episodes": len(group_episodes),
        "mean_final_duration": mean([final_duration(m.get("duration")) for m in display_metrics]),
        "mean_inference_calls": mean([m.get("inference_calls") for m in display_metrics]),
        "mean_duration": mean([m.get("mean_duration") for m in display_metrics]),
        "mean_total_time_s": mean([m.get("total_time_s") for m in display_metrics]),
        "task_prompts": sorted({ep.get("task_prompt") for ep in group_episodes if ep.get("task_prompt")}),
        "episode_metrics": group_episodes,
        "video_paths": group_video_paths,
    }


episodes_by_group = {}
for episode in episodes:
    episodes_by_group.setdefault(episode.get("task_group"), []).append(episode)
per_group = {
    group: summarize_episodes(group_episodes, group_video_paths.get(group, []))
    for group, group_episodes in sorted(episodes_by_group.items())
}

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

echo "===== Starting checkpoint ${CHECKPOINT_LABEL} at $(date) ====="
echo "TIMESTAMP=${TIMESTAMP}"
echo "POLICY_PATH=${POLICY_PATH}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "HIVA_COEFF_SIDECAR=${HIVA_COEFF_SIDECAR}"
echo "HIVA_COEFF_SUMMARY=${HIVA_COEFF_SUMMARY}"
echo "TASK_IDS_ALL=${TASK_IDS_ALL}"
echo "N_EPISODES=${N_EPISODES}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "MAX_PARALLEL_TASKS=${MAX_PARALLEL_TASKS}"
echo "MAX_EPISODES_RENDERED=${MAX_EPISODES_RENDERED}"
echo "EXPECTED_EPISODE_COUNT=${EXPECTED_EPISODE_COUNT}"
echo "EXPECTED_VIDEO_COUNT=${EXPECTED_VIDEO_COUNT}"
echo "GPU_IDS=${GPU_IDS}"
echo "SUITES_CSV=${SUITES_CSV}"
echo "SPLIT_LIBERO10_ACROSS_GPUS=${SPLIT_LIBERO10_ACROSS_GPUS}"
echo "STAGED_LIBERO10_AFTER_SHORT=${STAGED_LIBERO10_AFTER_SHORT}"
echo "LIBERO10_INITIAL_TASK_IDS=${LIBERO10_INITIAL_TASK_IDS}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "NUM_STEPS=${NUM_STEPS}"
echo "HIVA_DURATION_EXECUTION_MAP=${HIVA_DURATION_EXECUTION_MAP}"
echo "HIVA_RESIDUAL_INFERENCE_WEIGHT=${HIVA_RESIDUAL_INFERENCE_WEIGHT}"
echo "BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR}"

if [[ ! -d "${POLICY_PATH}" ]]; then
  echo "Missing policy directory: ${POLICY_PATH}" >&2
  exit 1
fi
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

short_pids=()
libero10_pids=()
tail_libero10_shards=()
for idx in "${!SUITES[@]}"; do
  suite="${SUITES[$idx]}"
  task_ids="$(task_ids_for "${suite}")"
  if [[ "${suite}" == "libero_10" && "${STAGED_LIBERO10_AFTER_SHORT}" == "1" && "${#GPU_ARRAY[@]}" -gt 1 ]]; then
    while IFS='|' read -r phase shard_task_ids; do
      [[ -n "${shard_task_ids}" ]] || continue
      if [[ "${phase}" == "initial" ]]; then
        gpu_id="${GPU_ARRAY[$((${#GPU_ARRAY[@]} - 1))]}"
        run_suite "${suite}" "${gpu_id}" "${shard_task_ids}" &
        libero10_pids+=("$!")
      elif [[ "${phase}" == "tail" ]]; then
        tail_libero10_shards+=("${shard_task_ids}")
      fi
    done < <(staged_libero10_task_ids "${task_ids}")
  elif [[ "${suite}" == "libero_10" && "${SPLIT_LIBERO10_ACROSS_GPUS}" == "1" && "${#GPU_ARRAY[@]}" -gt 1 ]]; then
    while IFS='|' read -r shard_idx shard_task_ids; do
      [[ -n "${shard_task_ids}" ]] || continue
      gpu_id="${GPU_ARRAY[$((shard_idx % ${#GPU_ARRAY[@]}))]}"
      run_suite "${suite}" "${gpu_id}" "${shard_task_ids}" &
      libero10_pids+=("$!")
    done < <(split_task_ids_for_gpus "${task_ids}")
  else
    gpu_id="${GPU_ARRAY[$((idx % ${#GPU_ARRAY[@]}))]}"
    run_suite "${suite}" "${gpu_id}" "${task_ids}" &
    if [[ "${suite}" == "libero_10" ]]; then
      libero10_pids+=("$!")
    else
      short_pids+=("$!")
    fi
  fi
done

status=0
for pid in "${short_pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${STAGED_LIBERO10_AFTER_SHORT}" == "1" && "${#tail_libero10_shards[@]}" -gt 0 ]]; then
  tail_gpu_count=$((${#GPU_ARRAY[@]} - 1))
  for shard_idx in "${!tail_libero10_shards[@]}"; do
    shard_task_ids="${tail_libero10_shards[$shard_idx]}"
    gpu_id="${GPU_ARRAY[$((shard_idx % tail_gpu_count))]}"
    run_suite "libero_10" "${gpu_id}" "${shard_task_ids}" &
    libero10_pids+=("$!")
  done
fi

for pid in "${libero10_pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "Checkpoint ${CHECKPOINT_LABEL} evaluation failed." >&2
  exit "${status}"
fi

write_summary
echo "===== Finished checkpoint ${CHECKPOINT_LABEL} at $(date) ====="
