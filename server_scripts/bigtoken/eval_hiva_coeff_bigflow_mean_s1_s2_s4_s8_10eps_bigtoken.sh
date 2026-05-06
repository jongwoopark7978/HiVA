#!/usr/bin/env bash
set -euo pipefail

# Partial LIBERO eval for mean-weighted HiVA coefficient SmolVLA checkpoints on bigtoken.
# Runs checkpoints sequentially; for each checkpoint, the four LIBERO suites run in
# parallel across GPU_IDS.

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
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-1}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-1}"
EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT:-400}"
EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT:-40}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
SUITE_GPU_IDS="${SUITE_GPU_IDS:-${GPU_IDS}}"
IFS=',' read -r -a SUITE_GPU_ARRAY <<< "${SUITE_GPU_IDS}"
SUITES=(libero_object libero_goal libero_spatial libero_10)
START_INDEX="${START_INDEX:-0}"

WAIT_FOR_GPU_MEMORY="${WAIT_FOR_GPU_MEMORY:-true}"
FREE_MEM_MIN_MIB="${FREE_MEM_MIN_MIB:-10000}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"
MONITOR_GPU_MEMORY="${MONITOR_GPU_MEMORY:-true}"
MEMORY_MONITOR_INTERVAL_SECONDS="${MEMORY_MONITOR_INTERVAL_SECONDS:-2}"

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
RENAME_MAP='{"observation.images.image":"observation.images.agentview","observation.images.image2":"observation.images.wrist"}'
MEMORY_MONITOR_CSV="${LOG_DIR}/hiva_coeff_mean_bigtoken_${TIMESTAMP}_gpu_memory.csv"
MEMORY_MONITOR_SUMMARY="${LOG_DIR}/hiva_coeff_mean_bigtoken_${TIMESTAMP}_gpu_memory_summary.json"

POLICY_PATHS=(
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_bigflow_b160_full_s1_rerun_20260504_125652/checkpoints/last/pretrained_model"
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_bigflow_b160_full_s2_20260504_052506/checkpoints/last/pretrained_model"
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_bigflow_b160_full_s4_20260504_030522/checkpoints/last/pretrained_model"
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_bigflow_b160_full_s8_20260504_015310/checkpoints/last/pretrained_model"
)

CHECKPOINT_LABELS=(
  "smolvla_hiva_coeff_bigflow_b160_full_s1_rerun_20260504_125652_10eps_bs4"
  "smolvla_hiva_coeff_bigflow_b160_full_s2_20260504_052506_10eps_bs4"
  "smolvla_hiva_coeff_bigflow_b160_full_s4_20260504_030522_10eps_bs4"
  "smolvla_hiva_coeff_bigflow_b160_full_s8_20260504_015310_10eps_bs4"
)
END_INDEX="${END_INDEX:-$((${#POLICY_PATHS[@]} - 1))}"

task_ids_for() {
  case "$1" in
    libero_object) echo "${OBJECT_TASK_IDS:-${TASK_IDS_ALL}}" ;;
    libero_goal) echo "${GOAL_TASK_IDS:-${TASK_IDS_ALL}}" ;;
    libero_spatial) echo "${SPATIAL_TASK_IDS:-${TASK_IDS_ALL}}" ;;
    libero_10) echo "${LIBERO10_TASK_IDS:-${TASK_IDS_ALL}}" ;;
    *) echo "${TASK_IDS:-${TASK_IDS_ALL}}" ;;
  esac
}

gpu_free_mib() {
  local gpu_id="$1"
  nvidia-smi \
    --query-gpu=memory.free \
    --format=csv,noheader,nounits \
    -i "${gpu_id}" | awk 'NR == 1 {gsub(/ /, ""); print $1}'
}

wait_for_gpus() {
  if [[ "${WAIT_FOR_GPU_MEMORY}" != "true" ]]; then
    return
  fi

  while true; do
    local ready=1
    local status_parts=()
    for gpu_id in "${GPU_ARRAY[@]}"; do
      local free_mib
      free_mib="$(gpu_free_mib "${gpu_id}")"
      status_parts+=("gpu${gpu_id}=${free_mib}MiB_free")
      if (( free_mib < FREE_MEM_MIN_MIB )); then
        ready=0
      fi
    done

    echo "GPU free memory: ${status_parts[*]} (need >= ${FREE_MEM_MIN_MIB} MiB each)"
    if (( ready == 1 )); then
      return
    fi
    sleep "${GPU_WAIT_SECONDS}"
  done
}

start_memory_monitor() {
  if [[ "${MONITOR_GPU_MEMORY}" != "true" ]]; then
    return
  fi

  ROOT_PID="$$" \
  GPU_IDS="${GPU_IDS}" \
  MEMORY_MONITOR_CSV="${MEMORY_MONITOR_CSV}" \
  MEMORY_MONITOR_SUMMARY="${MEMORY_MONITOR_SUMMARY}" \
  MEMORY_MONITOR_INTERVAL_SECONDS="${MEMORY_MONITOR_INTERVAL_SECONDS}" \
  python - <<'PY' &
import csv
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

root_pid = int(os.environ["ROOT_PID"])
gpu_ids = [int(part) for part in os.environ["GPU_IDS"].split(",") if part]
out_csv = Path(os.environ["MEMORY_MONITOR_CSV"])
out_summary = Path(os.environ["MEMORY_MONITOR_SUMMARY"])
interval_s = float(os.environ["MEMORY_MONITOR_INTERVAL_SECONDS"])


def root_alive():
    return subprocess.run(
        ["ps", "-p", str(root_pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def job_pids():
    out = subprocess.check_output(["ps", "-eo", "pid,pgid"], text=True)
    pids = {root_pid}
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) != 2:
            continue
        pid, pgid = map(int, parts)
        if pgid == root_pid:
            pids.add(pid)
    return pids


def gpu_totals():
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            ",".join(map(str, gpu_ids)),
        ],
        text=True,
    )
    totals = {}
    for line in out.splitlines():
        idx, used = [part.strip() for part in line.split(",")[:2]]
        totals[int(idx)] = int(used)
    return totals


def xml_process_mem(pids):
    out = subprocess.check_output(
        ["nvidia-smi", "-q", "-x", "-i", ",".join(map(str, gpu_ids))],
        text=True,
    )
    tree = ET.fromstring(out)
    per_gpu = {idx: 0 for idx in gpu_ids}
    entries = []
    for gpu_index, gpu in enumerate(tree.findall("gpu")):
        for proc in gpu.findall("./processes/process_info"):
            pid_text = proc.findtext("pid")
            if not pid_text:
                continue
            pid = int(pid_text)
            if pid not in pids:
                continue
            mem_text = proc.findtext("used_memory") or "0 MiB"
            mem = int(mem_text.split()[0])
            ptype = proc.findtext("type") or ""
            name = proc.findtext("process_name") or ""
            per_gpu[gpu_index] += mem
            entries.append(
                {
                    "gpu": gpu_index,
                    "pid": pid,
                    "type": ptype,
                    "used_memory_mib": mem,
                    "name": name,
                }
            )
    return per_gpu, entries


fieldnames = [
    "timestamp",
    *[f"gpu{idx}_total_used_mib" for idx in gpu_ids],
    *[f"gpu{idx}_job_used_mib" for idx in gpu_ids],
    "job_total_used_mib",
    "job_pids",
    "entries_json",
]
peaks = {
    "gpu_total_used_mib": {str(idx): 0 for idx in gpu_ids},
    "gpu_job_used_mib": {str(idx): 0 for idx in gpu_ids},
    "job_total_used_mib": 0,
}
rows = 0
out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    while root_alive():
        try:
            pids = job_pids()
            totals = gpu_totals()
            job_mem, entries = xml_process_mem(pids)
            job_total = sum(job_mem.values())
            row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
            for idx in gpu_ids:
                total_used = totals.get(idx, 0)
                job_used = job_mem.get(idx, 0)
                row[f"gpu{idx}_total_used_mib"] = total_used
                row[f"gpu{idx}_job_used_mib"] = job_used
                peaks["gpu_total_used_mib"][str(idx)] = max(
                    peaks["gpu_total_used_mib"][str(idx)], total_used
                )
                peaks["gpu_job_used_mib"][str(idx)] = max(
                    peaks["gpu_job_used_mib"][str(idx)], job_used
                )
            row["job_total_used_mib"] = job_total
            row["job_pids"] = " ".join(map(str, sorted(pids)))
            row["entries_json"] = json.dumps(entries, separators=(",", ":"))
            peaks["job_total_used_mib"] = max(peaks["job_total_used_mib"], job_total)
            writer.writerow(row)
            f.flush()
            rows += 1
        except Exception as exc:
            out_summary.write_text(
                json.dumps({"error": repr(exc), "peaks_so_far": peaks, "rows": rows}, indent=2)
            )
        time.sleep(interval_s)

out_summary.write_text(
    json.dumps(
        {
            "root_pid": root_pid,
            "gpu_ids": gpu_ids,
            "sample_interval_s": interval_s,
            "rows": rows,
            "peaks": peaks,
        },
        indent=2,
    )
)
PY
  MEMORY_MONITOR_PID="$!"
  echo "Started GPU memory monitor PID ${MEMORY_MONITOR_PID}"
  echo "Memory monitor CSV: ${MEMORY_MONITOR_CSV}"
  echo "Memory monitor summary: ${MEMORY_MONITOR_SUMMARY}"
}

stop_memory_monitor() {
  if [[ -n "${MEMORY_MONITOR_PID:-}" ]]; then
    kill "${MEMORY_MONITOR_PID}" 2>/dev/null || true
    wait "${MEMORY_MONITOR_PID}" 2>/dev/null || true
  fi
}

run_suite() {
  local policy_path="$1"
  local checkpoint_label="$2"
  local base_output_dir="$3"
  local suite="$4"
  local gpu_id="$5"
  local task_ids="$6"
  local safe_task_ids="${task_ids//[^0-9A-Za-z_-]/_}"
  local run_name="hiva_coeff_mean_bigtoken_${checkpoint_label}_${suite}_taskids_${safe_task_ids}_${TIMESTAMP}"
  local output_dir="${base_output_dir}/${suite}_taskids_${safe_task_ids}"
  local log_path="${LOG_DIR}/${run_name}.log"

  echo "[${checkpoint_label}] Starting ${suite} task_ids=${task_ids} on GPU ${gpu_id}"
  echo "[${checkpoint_label}] Log: ${log_path}"
  echo "[${checkpoint_label}] Output dir: ${output_dir}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  MUJOCO_EGL_DEVICE_ID="${gpu_id}" \
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

run_checkpoint() {
  local policy_path="$1"
  local checkpoint_label="$2"
  local base_output_dir="${REPO_ROOT}/outputs/eval/full_bigtoken_${checkpoint_label}_${TIMESTAMP}"

  echo "===== Starting checkpoint ${checkpoint_label} at $(date) ====="
  echo "POLICY_PATH=${policy_path}"
  echo "BASE_OUTPUT_DIR=${base_output_dir}"
  wait_for_gpus

  pids=()
  for idx in "${!SUITES[@]}"; do
    suite="${SUITES[$idx]}"
    gpu_id="${SUITE_GPU_ARRAY[$((idx % ${#SUITE_GPU_ARRAY[@]}))]}"
    task_ids="$(task_ids_for "${suite}")"
    run_suite "${policy_path}" "${checkpoint_label}" "${base_output_dir}" "${suite}" "${gpu_id}" "${task_ids}" &
    pids+=("$!")
  done

  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done

  if [[ "${status}" -ne 0 ]]; then
    echo "Checkpoint ${checkpoint_label} evaluation failed." >&2
    exit "${status}"
  fi

  write_summary "${base_output_dir}"
  echo "===== Finished checkpoint ${checkpoint_label} at $(date) ====="
}

echo "TIMESTAMP=${TIMESTAMP}"
echo "TASK_IDS_ALL=${TASK_IDS_ALL}"
echo "N_EPISODES=${N_EPISODES}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "MAX_PARALLEL_TASKS=${MAX_PARALLEL_TASKS}"
echo "MAX_EPISODES_RENDERED=${MAX_EPISODES_RENDERED}"
echo "EXPECTED_EPISODE_COUNT=${EXPECTED_EPISODE_COUNT}"
echo "EXPECTED_VIDEO_COUNT=${EXPECTED_VIDEO_COUNT}"
echo "GPU_IDS=${GPU_IDS}"
echo "SUITE_GPU_IDS=${SUITE_GPU_IDS}"
echo "START_INDEX=${START_INDEX}"
echo "END_INDEX=${END_INDEX}"
echo "WAIT_FOR_GPU_MEMORY=${WAIT_FOR_GPU_MEMORY}"
echo "FREE_MEM_MIN_MIB=${FREE_MEM_MIN_MIB}"
echo "MONITOR_GPU_MEMORY=${MONITOR_GPU_MEMORY}"
echo "MEMORY_MONITOR_INTERVAL_SECONDS=${MEMORY_MONITOR_INTERVAL_SECONDS}"
echo "MEMORY_MONITOR_CSV=${MEMORY_MONITOR_CSV}"
echo "MEMORY_MONITOR_SUMMARY=${MEMORY_MONITOR_SUMMARY}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "HIVA_COEFF_SIDECAR=${HIVA_COEFF_SIDECAR}"
echo "HIVA_COEFF_SUMMARY=${HIVA_COEFF_SUMMARY}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "NUM_STEPS=${NUM_STEPS}"

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

start_memory_monitor

for idx in "${!POLICY_PATHS[@]}"; do
  if (( idx < START_INDEX || idx > END_INDEX )); then
    continue
  fi
  policy_path="${POLICY_PATHS[$idx]}"
  checkpoint_label="${CHECKPOINT_LABELS[$idx]}"
  if [[ ! -d "${policy_path}" ]]; then
    echo "Missing policy directory: ${policy_path}" >&2
    exit 1
  fi
  run_checkpoint "${policy_path}" "${checkpoint_label}"
done

echo "===== Finished all checkpoints at $(date) ====="
