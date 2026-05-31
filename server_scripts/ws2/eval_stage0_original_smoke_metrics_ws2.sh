#!/usr/bin/env bash
set -euo pipefail

# Smoke evaluation for one task from each LIBERO suite on ws2.
#
# Defaults:
#   - 4 suites: libero_object, libero_goal, libero_spatial, libero_10
#   - task_id [0] for every suite
#   - 4 episodes per task, eval batch size 4
#   - one suite per GPU on GPUs 0,1,2,3
#
# Outputs:
#   - raw per-suite eval_info.json files
#   - episode_metrics.json: one row per evaluated episode
#   - metrics_summary.json: task-level, suite-level, and all-suites aggregates
#   - metrics_summary.csv: compact aggregate table
#
# The aggregate records both all-episode and success-only averages for:
#   - total completion time
#   - model inference call count

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

POLICY_PATH="${POLICY_PATH:-/raid/jongwoopark/HiVA/HiVA_train/finetuning_stage0/Acc73.5_smolvla_original_bigcornea_s0p5_b64_g8_20260515_223843_888072651_pid2567557/checkpoints/003750/pretrained_model}"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-Acc73p5_stage0_original_ckpt003750_smoke_metrics}"

DATA_ROOT="${DATA_ROOT:-/raid/jongwoopark/HiVA/libero_lerobot_v3_lerobotkeys}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/raid/jongwoopark/HiVA/HiVA_sidecars}"
SIDECAR="${SIDECAR:-${SIDECAR_ROOT}/libero_duration_sidecar_all_episodes.parquet}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
source "/home/jongwoopark/lerobot/server_scripts/common_hf_cache.sh"
setup_hf_datasets_cache

# LIBERO prompts for this config on first import. Create it explicitly so
# noninteractive multi-GPU evals do not crash with EOFError.
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/raid/jongwoopark/HiVA/libero_config_ws2}"
LIBERO_BENCHMARK_ROOT="${LIBERO_BENCHMARK_ROOT:-/home/jongwoopark/miniconda3/envs/smolvla_libero/lib/python3.12/site-packages/libero/libero}"
export LIBERO_CONFIG_PATH
mkdir -p "${LIBERO_CONFIG_PATH}"
cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<YAML
assets: ${LIBERO_BENCHMARK_ROOT}/assets
bddl_files: ${LIBERO_BENCHMARK_ROOT}/bddl_files
benchmark_root: ${LIBERO_BENCHMARK_ROOT}
datasets: ${DATA_ROOT}
init_states: ${LIBERO_BENCHMARK_ROOT}/init_files
YAML

GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
SUITES_CSV="${SUITES_CSV:-libero_object,libero_goal,libero_spatial,libero_10}"
IFS=',' read -r -a SUITES <<< "${SUITES_CSV}"

TASK_IDS_ALL="${TASK_IDS_ALL:-[0]}"
OBJECT_TASK_IDS="${OBJECT_TASK_IDS:-${TASK_IDS_ALL}}"
GOAL_TASK_IDS="${GOAL_TASK_IDS:-${TASK_IDS_ALL}}"
SPATIAL_TASK_IDS="${SPATIAL_TASK_IDS:-${TASK_IDS_ALL}}"
LIBERO10_TASK_IDS="${LIBERO10_TASK_IDS:-${TASK_IDS_ALL}}"

N_EPISODES="${N_EPISODES:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-0}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-1}"
N_ACTION_STEPS="${N_ACTION_STEPS:-1}"
USE_DURATION_HEAD="${USE_DURATION_HEAD:-false}"
DURATION_TRAIN_REUSE_PREFIX_CACHE="${DURATION_TRAIN_REUSE_PREFIX_CACHE:-true}"
POLICY_NUM_STEPS="${POLICY_NUM_STEPS:-}"
POLICY_CHUNK_SIZE="${POLICY_CHUNK_SIZE:-}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-}"
HIVA_DURATION_EXECUTION_MAP="${HIVA_DURATION_EXECUTION_MAP:-}"
HIVA_RESIDUAL_INFERENCE_WEIGHT="${HIVA_RESIDUAL_INFERENCE_WEIGHT:-}"

if [[ -z "${RENAME_MAP:-}" ]]; then
  RENAME_MAP='{"observation.images.image":"observation.images.agentview","observation.images.image2":"observation.images.wrist"}'
fi
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/raid/jongwoopark/HiVA/HiVA_repo/lerobot/outputs/eval/ws2_smoke_metrics_${CHECKPOINT_LABEL}_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-/raid/jongwoopark/HiVA/HiVA_repo/lerobot/outputs/eval_logs}"
mkdir -p "${BASE_OUTPUT_DIR}" "${LOG_DIR}"

if [[ -z "${EXPECTED_EPISODE_COUNT:-}" ]]; then
  EXPECTED_EPISODE_COUNT="$(
    SUITES_CSV="${SUITES_CSV}" \
    TASK_IDS_ALL="${TASK_IDS_ALL}" \
    OBJECT_TASK_IDS="${OBJECT_TASK_IDS}" \
    GOAL_TASK_IDS="${GOAL_TASK_IDS}" \
    SPATIAL_TASK_IDS="${SPATIAL_TASK_IDS}" \
    LIBERO10_TASK_IDS="${LIBERO10_TASK_IDS}" \
    N_EPISODES="${N_EPISODES}" \
    python - <<'PY'
import ast
import os

def count_tasks(raw):
    parsed = ast.literal_eval(raw)
    if isinstance(parsed, int):
        return 1
    return len(parsed)

task_ids_by_suite = {
    "libero_object": os.environ["OBJECT_TASK_IDS"],
    "libero_goal": os.environ["GOAL_TASK_IDS"],
    "libero_spatial": os.environ["SPATIAL_TASK_IDS"],
    "libero_10": os.environ["LIBERO10_TASK_IDS"],
}
n_episodes = int(os.environ["N_EPISODES"])
total = 0
for suite in os.environ["SUITES_CSV"].split(","):
    total += count_tasks(task_ids_by_suite.get(suite, os.environ["TASK_IDS_ALL"])) * n_episodes
print(total)
PY
  )"
fi

task_ids_for() {
  case "$1" in
    libero_object) echo "${OBJECT_TASK_IDS}" ;;
    libero_goal) echo "${GOAL_TASK_IDS}" ;;
    libero_spatial) echo "${SPATIAL_TASK_IDS}" ;;
    libero_10) echo "${LIBERO10_TASK_IDS}" ;;
    *) echo "${TASK_IDS_ALL}" ;;
  esac
}

run_suite() {
  local suite="$1"
  local gpu_id="$2"
  local task_ids="$3"
  local safe_task_ids="${task_ids//[^0-9A-Za-z_-]/_}"
  local run_name="ws2_smoke_metrics_${CHECKPOINT_LABEL}_${suite}_taskids_${safe_task_ids}_${TIMESTAMP}"
  local output_dir="${BASE_OUTPUT_DIR}/${suite}_taskids_${safe_task_ids}"
  local log_path="${LOG_DIR}/${run_name}.log"
  local -a policy_extra_args=()

  [[ -n "${POLICY_NUM_STEPS}" ]] && policy_extra_args+=(--policy.num_steps="${POLICY_NUM_STEPS}")
  [[ -n "${POLICY_CHUNK_SIZE}" ]] && policy_extra_args+=(--policy.chunk_size="${POLICY_CHUNK_SIZE}")
  [[ -n "${HIVA_COEFF_SIDECAR}" ]] && policy_extra_args+=(--policy.hiva_coeff_sidecar_path="${HIVA_COEFF_SIDECAR}")
  [[ -n "${HIVA_COEFF_SUMMARY}" ]] && policy_extra_args+=(--policy.hiva_coeff_sidecar_summary_path="${HIVA_COEFF_SUMMARY}")
  [[ -n "${HIVA_DURATION_EXECUTION_MAP}" ]] && policy_extra_args+=(--policy.hiva_duration_execution_map="${HIVA_DURATION_EXECUTION_MAP}")
  [[ -n "${HIVA_RESIDUAL_INFERENCE_WEIGHT}" ]] && policy_extra_args+=(--policy.hiva_residual_inference_weight="${HIVA_RESIDUAL_INFERENCE_WEIGHT}")

  echo "[$(date)] Starting ${suite} task_ids=${task_ids} on GPU ${gpu_id}"
  echo "[$(date)] Log: ${log_path}"
  echo "[$(date)] Output: ${output_dir}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  MUJOCO_EGL_DEVICE_ID="${gpu_id}" \
  DATA_ROOT="${DATA_ROOT}" \
  lerobot-eval \
    --policy.path="${POLICY_PATH}" \
    --policy.device=cuda \
    "${policy_extra_args[@]}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.use_duration_head="${USE_DURATION_HEAD}" \
    --policy.duration_train_reuse_prefix_cache="${DURATION_TRAIN_REUSE_PREFIX_CACHE}" \
    --policy.duration_sidecar_path="${SIDECAR}" \
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

write_metrics_summary() {
  BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR}" \
  EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT}" \
  python - <<'PY'
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

base = Path(os.environ["BASE_OUTPUT_DIR"])
expected_episode_count = int(os.environ["EXPECTED_EPISODE_COUNT"])
eval_infos = sorted(base.glob("*/eval_info.json"))

episodes = []
raw_per_task = []
raw_per_suite = {}
video_paths = []


def number(value):
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def horizon_key(value):
    value = number(value)
    if value is None:
        return None
    return str(int(value)) if value.is_integer() else str(value)


for path in eval_infos:
    info = json.loads(path.read_text())
    raw_per_task.extend(info.get("per_task", []))
    raw_per_suite.update(info.get("per_group", {}))
    video_paths.extend(info.get("overall", {}).get("video_paths", []))
    for task_info in info.get("per_task", []):
        suite = task_info.get("task_group")
        task_id = task_info.get("task_id")
        task_prompt = task_info.get("task_prompt")
        for episode in task_info.get("metrics", {}).get("episode_metrics", []):
            display = episode.get("display_metrics", {}) or {}
            success = bool(episode.get("success"))
            duration_sequence = display.get("duration") or []
            action_horizon_call_counts = Counter(
                key for key in (horizon_key(value) for value in duration_sequence) if key is not None
            )
            model_inference_calls = display.get("inference_calls")
            model_forward_latency_s = display.get("model_forward_latency_s")
            mean_model_forward_latency_s = display.get("mean_model_forward_latency_s")
            model_forward_latency_sequence_s = display.get("model_forward_latency_sequence_s") or []
            total_executed_actions = sum(number(value) or 0.0 for value in duration_sequence)
            avg_executed_action_horizon_per_call = (
                total_executed_actions / float(model_inference_calls)
                if number(model_inference_calls)
                else None
            )
            row = {
                "suite": suite,
                "task_id": task_id,
                "episode_ix": episode.get("episode_ix"),
                "task_prompt": task_prompt or display.get("task_prompt"),
                "success": success,
                "accuracy": 1.0 if success else 0.0,
                "sum_reward": episode.get("sum_reward"),
                "max_reward": episode.get("max_reward"),
                "seed": episode.get("seed"),
                "total_completion_time_s": display.get("total_time_s"),
                "model_inference_calls": model_inference_calls,
                "model_forward_latency_s": model_forward_latency_s,
                "mean_model_forward_latency_s": mean_model_forward_latency_s,
                "model_forward_latency_sequence_s": model_forward_latency_sequence_s,
                "total_executed_actions": total_executed_actions,
                "avg_executed_action_horizon_per_call": avg_executed_action_horizon_per_call,
                "action_horizon_call_counts": dict(
                    sorted(action_horizon_call_counts.items(), key=lambda item: number(item[0]) or 0.0)
                ),
                "action_jitter_metrics": display.get("action_jitter_metrics"),
                "final_duration": (duration_sequence or [None])[-1],
                "duration_sequence": duration_sequence,
                "mean_duration": display.get("mean_duration"),
            }
            episodes.append(row)

def mean(values):
    vals = [number(value) for value in values]
    vals = [value for value in vals if value is not None]
    return sum(vals) / len(vals) if vals else None


def avg_calls_by_action_horizon(group_episodes):
    if not group_episodes:
        return None
    counts = Counter()
    for ep in group_episodes:
        counts.update(ep.get("action_horizon_call_counts", {}))
    return {
        key: counts[key] / len(group_episodes)
        for key in sorted(counts, key=lambda value: number(value) or 0.0)
    }

def avg_model_forward_latency_per_call(group_episodes):
    latency_sum = 0.0
    call_count = 0.0
    for ep in group_episodes:
        latency = number(ep.get("model_forward_latency_s"))
        calls = number(ep.get("model_inference_calls"))
        if latency is None or calls is None or calls <= 0:
            continue
        latency_sum += latency
        call_count += calls
    return latency_sum / call_count if call_count > 0 else None


def avg_action_jitter_metrics(group_episodes):
    grouped_values = defaultdict(lambda: defaultdict(list))
    top_level_values = defaultdict(list)
    for ep in group_episodes:
        metrics = ep.get("action_jitter_metrics")
        if not isinstance(metrics, dict):
            continue
        for key in ("n_actions", "action_dim", "boundary_count"):
            value = number(metrics.get(key))
            if value is not None:
                top_level_values[key].append(value)
        for group_name, group_metrics in (metrics.get("groups") or {}).items():
            if not isinstance(group_metrics, dict):
                continue
            for metric_name, value in group_metrics.items():
                value = number(value)
                if value is not None:
                    grouped_values[str(group_name)][str(metric_name)].append(value)
    if not grouped_values and not top_level_values:
        return None
    return {
        "n_episodes": sum(1 for ep in group_episodes if isinstance(ep.get("action_jitter_metrics"), dict)),
        "mean_n_actions": mean(top_level_values.get("n_actions", [])),
        "mean_action_dim": mean(top_level_values.get("action_dim", [])),
        "mean_boundary_count": mean(top_level_values.get("boundary_count", [])),
        "groups": {
            group_name: {
                metric_name: mean(values)
                for metric_name, values in sorted(metric_values.items())
            }
            for group_name, metric_values in sorted(grouped_values.items())
        },
    }


def summarize(group_episodes):
    success_eps = [ep for ep in group_episodes if ep["success"]]
    failure_eps = [ep for ep in group_episodes if not ep["success"]]
    total_calls = sum(number(ep["model_inference_calls"]) or 0.0 for ep in group_episodes)
    success_calls = sum(number(ep["model_inference_calls"]) or 0.0 for ep in success_eps)
    failure_calls = sum(number(ep["model_inference_calls"]) or 0.0 for ep in failure_eps)
    total_executed_actions = sum(number(ep["total_executed_actions"]) or 0.0 for ep in group_episodes)
    success_executed_actions = sum(number(ep["total_executed_actions"]) or 0.0 for ep in success_eps)
    failure_executed_actions = sum(number(ep["total_executed_actions"]) or 0.0 for ep in failure_eps)
    return {
        "n_episodes": len(group_episodes),
        "n_success_episodes": len(success_eps),
        "n_failure_episodes": len(failure_eps),
        "accuracy": mean([ep["accuracy"] for ep in group_episodes]),
        "accuracy_percent": (mean([ep["accuracy"] for ep in group_episodes]) or 0.0) * 100
        if group_episodes
        else None,
        "avg_total_completion_time_all_episodes_s": mean(
            [ep["total_completion_time_s"] for ep in group_episodes]
        ),
        "avg_model_inference_calls_all_episodes": mean(
            [ep["model_inference_calls"] for ep in group_episodes]
        ),
        "avg_model_forward_latency_per_call_all_episodes_s": avg_model_forward_latency_per_call(group_episodes),
        "avg_action_jitter_metrics_all_episodes": avg_action_jitter_metrics(group_episodes),
        "avg_total_completion_time_success_episodes_s": mean(
            [ep["total_completion_time_s"] for ep in success_eps]
        ),
        "avg_model_inference_calls_success_episodes": mean(
            [ep["model_inference_calls"] for ep in success_eps]
        ),
        "avg_model_forward_latency_per_call_success_episodes_s": avg_model_forward_latency_per_call(success_eps),
        "avg_action_jitter_metrics_success_episodes": avg_action_jitter_metrics(success_eps),
        "avg_total_completion_time_failure_episodes_s": mean(
            [ep["total_completion_time_s"] for ep in failure_eps]
        ),
        "avg_model_inference_calls_failure_episodes": mean(
            [ep["model_inference_calls"] for ep in failure_eps]
        ),
        "avg_model_forward_latency_per_call_failure_episodes_s": avg_model_forward_latency_per_call(failure_eps),
        "avg_action_jitter_metrics_failure_episodes": avg_action_jitter_metrics(failure_eps),
        "avg_executed_action_horizon_per_call_all_episodes": (
            total_executed_actions / total_calls if total_calls > 0 else None
        ),
        "avg_model_inference_calls_by_action_horizon_all_episodes": avg_calls_by_action_horizon(group_episodes),
        "avg_executed_action_horizon_per_call_success_episodes": (
            success_executed_actions / success_calls if success_calls > 0 else None
        ),
        "avg_model_inference_calls_by_action_horizon_success_episodes": avg_calls_by_action_horizon(success_eps),
        "avg_executed_action_horizon_per_call_failure_episodes": (
            failure_executed_actions / failure_calls if failure_calls > 0 else None
        ),
        "avg_model_inference_calls_by_action_horizon_failure_episodes": avg_calls_by_action_horizon(failure_eps),
    }


task_groups = defaultdict(list)
suite_groups = defaultdict(list)
for ep in episodes:
    task_groups[f"{ep['suite']}/task_{ep['task_id']}"].append(ep)
    suite_groups[str(ep["suite"])].append(ep)

summary = {
    "base_output_dir": str(base),
    "eval_info_files": [str(path) for path in eval_infos],
    "raw_per_task": raw_per_task,
    "raw_per_suite": raw_per_suite,
    "task_level": {
        key: summarize(group_episodes) for key, group_episodes in sorted(task_groups.items())
    },
    "suite_level": {
        key: summarize(group_episodes) for key, group_episodes in sorted(suite_groups.items())
    },
    "all_suites_level": summarize(episodes),
    "video_paths": video_paths,
    "n_video_paths": len(video_paths),
}

episode_path = base / "episode_metrics.json"
summary_path = base / "metrics_summary.json"
csv_path = base / "metrics_summary.csv"

episode_path.write_text(json.dumps(episodes, indent=2))
summary_path.write_text(json.dumps(summary, indent=2))

fieldnames = [
    "level",
    "name",
    "n_episodes",
    "n_success_episodes",
    "n_failure_episodes",
    "accuracy_percent",
    "avg_total_completion_time_all_episodes_s",
    "avg_model_inference_calls_all_episodes",
    "avg_model_forward_latency_per_call_all_episodes_s",
    "avg_action_jitter_metrics_all_episodes",
    "avg_executed_action_horizon_per_call_all_episodes",
    "avg_model_inference_calls_by_action_horizon_all_episodes",
    "avg_total_completion_time_success_episodes_s",
    "avg_model_inference_calls_success_episodes",
    "avg_model_forward_latency_per_call_success_episodes_s",
    "avg_action_jitter_metrics_success_episodes",
    "avg_executed_action_horizon_per_call_success_episodes",
    "avg_model_inference_calls_by_action_horizon_success_episodes",
    "avg_total_completion_time_failure_episodes_s",
    "avg_model_inference_calls_failure_episodes",
    "avg_model_forward_latency_per_call_failure_episodes_s",
    "avg_action_jitter_metrics_failure_episodes",
    "avg_executed_action_horizon_per_call_failure_episodes",
    "avg_model_inference_calls_by_action_horizon_failure_episodes",
]


def csv_value(value):
    if isinstance(value, dict):
        return json.dumps(value)
    return value


with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for level, groups in [
        ("task", summary["task_level"]),
        ("suite", summary["suite_level"]),
        ("all_suites", {"all_suites": summary["all_suites_level"]}),
    ]:
        for name, metrics in groups.items():
            row = {"level": level, "name": name}
            row.update({key: csv_value(metrics.get(key)) for key in fieldnames if key not in row})
            writer.writerow(row)

print(f"Wrote episode metrics: {episode_path}")
print(f"Wrote metrics summary: {summary_path}")
print(f"Wrote metrics CSV: {csv_path}")
print(f"Collected {len(episodes)} episodes from {len(eval_infos)} eval_info files.")

if expected_episode_count >= 0 and len(episodes) != expected_episode_count:
    print(
        f"Expected {expected_episode_count} episodes, but collected {len(episodes)}.",
        file=sys.stderr,
    )
    sys.exit(2)
PY
}

echo "===== ws2 smoke metrics eval started at $(date) ====="
echo "Host: $(hostname)"
echo "POLICY_PATH=${POLICY_PATH}"
echo "CHECKPOINT_LABEL=${CHECKPOINT_LABEL}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "SIDECAR=${SIDECAR}"
echo "LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH}"
echo "GPU_IDS=${GPU_IDS}"
echo "SUITES_CSV=${SUITES_CSV}"
echo "TASK_IDS_ALL=${TASK_IDS_ALL}"
echo "N_EPISODES=${N_EPISODES}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "MAX_EPISODES_RENDERED=${MAX_EPISODES_RENDERED}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "USE_DURATION_HEAD=${USE_DURATION_HEAD}"
echo "BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR}"
echo "LOG_DIR=${LOG_DIR}"

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
if [[ "${#GPU_ARRAY[@]}" -lt "${#SUITES[@]}" ]]; then
  echo "Need at least ${#SUITES[@]} GPU ids for ${#SUITES[@]} suites; got ${GPU_IDS}" >&2
  exit 1
fi

pids=()
for idx in "${!SUITES[@]}"; do
  suite="${SUITES[$idx]}"
  gpu_id="${GPU_ARRAY[$idx]}"
  task_ids="$(task_ids_for "${suite}")"
  run_suite "${suite}" "${gpu_id}" "${task_ids}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "At least one suite eval failed. Logs are under ${LOG_DIR}" >&2
  exit "${status}"
fi

write_metrics_summary

echo "===== ws2 smoke metrics eval finished at $(date) ====="
echo "Outputs are under: ${BASE_OUTPUT_DIR}"
