#!/usr/bin/env bash
set -euo pipefail

# Full LIBERO eval for selected SmolVLA HiVA stage0 checkpoints on ws1.
# Runs checkpoints sequentially. Each checkpoint uses fixed policy.n_action_steps=10
# and the staged task assignment across 4 GPUs:
#   GPU0: libero_object tasks [0..9]
#   GPU1: libero_goal tasks [0..9]
#   GPU2: libero_spatial tasks [0..9]
#   GPU3: libero_10 tasks [0..6]
# After the three short suites finish, GPUs 0-2 run libero_10 tasks [7], [8], [9].

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

DEFAULT_CHECKPOINT_DIRS=(
  "/raid/jongwoopark/HiVA/HiVA_train/finetuning_stage0/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_p7_f15_bigflow_b128_g4_s0p5_steps5000_nores_nodl_20260515_044459/checkpoints/003125"
  "/raid/jongwoopark/HiVA/HiVA_train/finetuning_stage0/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_p7_f15_bigflow_b128_g4_s0p5_steps5000_nores_nodl_20260515_044459/checkpoints/003500"
  "/raid/jongwoopark/HiVA/HiVA_train/finetuning_stage0/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_p7_f15_bigflow_b128_g4_s0p5_steps5000_nores_nodl_20260515_044459/checkpoints/004375"
  "/raid/jongwoopark/HiVA/HiVA_train/finetuning_stage0/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_p7_f15_bigflow_b128_g4_s0p5_steps5000_nores_nodl_20260515_044459/checkpoints/005000"
)
if [[ -n "${CHECKPOINT_DIRS_CSV:-}" ]]; then
  IFS=',' read -r -a CHECKPOINT_DIRS <<< "${CHECKPOINT_DIRS_CSV}"
else
  CHECKPOINT_DIRS=("${DEFAULT_CHECKPOINT_DIRS[@]}")
fi
CHECKPOINT_LABEL_PREFIX="${CHECKPOINT_LABEL_PREFIX:-hiva_coeffpool_stage0}"

DATA_ROOT="${DATA_ROOT:-/raid/jongwoopark/HiVA/libero_lerobot_v3_lerobotkeys}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/raid/jongwoopark/HiVA/HiVA_sidecars}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p7_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p7_f15_canonical_lp_mt.summary.json}"
HIVA_DURATION_EXECUTION_MAP="${HIVA_DURATION_EXECUTION_MAP:-}"
HIVA_RESIDUAL_INFERENCE_WEIGHT="${HIVA_RESIDUAL_INFERENCE_WEIGHT:-}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/nfs/ws4/ssd/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_BIN}/python}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/jongwoo_hf_datasets_cache}"
mkdir -p "${HF_DATASETS_CACHE}"

LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/raid/jongwoopark/HiVA/libero_config_ws1}"
LIBERO_BENCHMARK_ROOT="${LIBERO_BENCHMARK_ROOT:-/nfs/ws4/ssd/jongwoopark/miniconda3/envs/smolvla_libero/lib/python3.12/site-packages/libero/libero}"
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
if [[ "${#GPU_ARRAY[@]}" -lt 4 ]]; then
  echo "Need at least 4 GPU ids; got GPU_IDS=${GPU_IDS}" >&2
  exit 1
fi

TASK_IDS_ALL="${TASK_IDS_ALL:-[0,1,2,3,4,5,6,7,8,9]}"
LIBERO10_INITIAL_TASK_IDS="${LIBERO10_INITIAL_TASK_IDS:-[0,1,2,3,4,5,6]}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"

N_EPISODES="${N_EPISODES:-50}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-17}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-1}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-1}"
EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT:-2000}"
EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT:-40}"
NUM_STEPS="${NUM_STEPS:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-15}"
USE_DURATION_HEAD="${USE_DURATION_HEAD:-false}"
SEED="${SEED:-1000}"

RENAME_MAP='{"observation.images.image":"observation.images.agentview","observation.images.image2":"observation.images.wrist"}'
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-/raid/jongwoopark/HiVA/HiVA_repo/lerobot/outputs/eval/ws1_full_${CHECKPOINT_LABEL_PREFIX}_na10_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-/raid/jongwoopark/HiVA/HiVA_repo/lerobot/outputs/eval_logs}"
mkdir -p "${SWEEP_OUTPUT_ROOT}" "${LOG_DIR}"

safe_name() {
  echo "$1" | tr -c '0-9A-Za-z_.-' '_'
}

run_task_sequence() {
  local base_output_dir="$1"
  local n_action_steps="$2"
  local suite="$3"
  local gpu_id="$4"
  shift 4
  local task_id
  for task_id in "$@"; do
    run_suite "${base_output_dir}" "${n_action_steps}" "${suite}" "${gpu_id}" "[${task_id}]"
  done
}

run_suite() {
  local base_output_dir="$1"
  local n_action_steps="$2"
  local suite="$3"
  local gpu_id="$4"
  local task_ids="$5"
  local safe_task_ids
  safe_task_ids="$(safe_name "${task_ids}")"
  local run_name="ws1_full_${CHECKPOINT_LABEL}_na${n_action_steps}_${suite}_taskids_${safe_task_ids}_${TIMESTAMP}"
  local output_dir="${base_output_dir}/${suite}_taskids_${safe_task_ids}"
  local log_path="${LOG_DIR}/${run_name}.log"

  if [[ -f "${output_dir}/eval_info.json" ]]; then
    echo "[$(date)] skip completed ${suite} task_ids=${task_ids}: ${output_dir}"
    return
  fi

  local policy_extra_args=(
    --policy.hiva_coeff_sidecar_path="${HIVA_COEFF_SIDECAR}"
    --policy.hiva_coeff_sidecar_summary_path="${HIVA_COEFF_SUMMARY}"
    --policy.hiva_duration_execution_map="${HIVA_DURATION_EXECUTION_MAP}"
  )
  if [[ -n "${HIVA_RESIDUAL_INFERENCE_WEIGHT}" ]]; then
    policy_extra_args+=(
      --policy.hiva_residual_inference_weight="${HIVA_RESIDUAL_INFERENCE_WEIGHT}"
    )
  fi

  echo "[$(date)] n_actions=${n_action_steps} start ${suite} task_ids=${task_ids} on GPU ${gpu_id}"
  echo "[$(date)] Log: ${log_path}"
  echo "[$(date)] Output: ${output_dir}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  MUJOCO_EGL_DEVICE_ID="${gpu_id}" \
  DATA_ROOT="${DATA_ROOT}" \
  "${PYTHON_BIN}" -m lerobot.scripts.lerobot_eval \
    --policy.path="${POLICY_PATH}" \
    --policy.device=cuda \
    --policy.num_steps="${NUM_STEPS}" \
    --policy.chunk_size="${CHUNK_SIZE}" \
    --policy.n_action_steps="${n_action_steps}" \
    --policy.use_duration_head="${USE_DURATION_HEAD}" \
    "${policy_extra_args[@]}" \
    --env.type=libero \
    --env.task="${suite}" \
    --env.task_ids="${task_ids}" \
    --env.control_mode=relative \
    --env.max_parallel_tasks="${MAX_PARALLEL_TASKS}" \
    --eval.batch_size="${EVAL_BATCH_SIZE}" \
    --eval.n_episodes="${N_EPISODES}" \
    --eval.max_episodes_rendered="${MAX_EPISODES_RENDERED}" \
    --seed="${SEED}" \
    --rename_map="${RENAME_MAP}" \
    --output_dir="${output_dir}" \
    --job_name="${run_name}" \
    > "${log_path}" 2>&1
}

write_checkpoint_summary() {
  local base_output_dir="$1"
  BASE_OUTPUT_DIR="${base_output_dir}" \
  EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT}" \
  EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT}" \
  N_ACTION_STEPS="${N_ACTION_STEPS}" \
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
expected_video_count = int(os.environ["EXPECTED_VIDEO_COUNT"])
n_action_steps = int(os.environ["N_ACTION_STEPS"])
eval_infos = sorted(base.glob("*/eval_info.json"))

per_task = []
episodes = []
video_paths = []
group_video_paths = defaultdict(list)

def number(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value

def horizon_key(value):
    value = number(value)
    if value is None:
        return None
    return str(int(value)) if value.is_integer() else str(value)

for path in eval_infos:
    info = json.loads(path.read_text())
    per_task.extend(info.get("per_task", []))
    group_names = list(info.get("per_group", {}).keys())
    if len(group_names) == 1:
        group_video_paths[group_names[0]].extend(info.get("overall", {}).get("video_paths", []))
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
            episodes.append({
                "n_action_steps": n_action_steps,
                "suite": suite,
                "task_group": suite,
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
            })

def mean(values):
    vals = [number(v) for v in values]
    vals = [v for v in vals if v is not None]
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
    n_jitter_episodes = 0
    for ep in group_episodes:
        metrics = ep.get("action_jitter_metrics")
        if not isinstance(metrics, dict):
            continue
        n_jitter_episodes += 1
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
    if n_jitter_episodes == 0:
        return None
    return {
        "n_episodes": n_jitter_episodes,
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

def summarize(group_episodes, videos=None):
    videos = videos or []
    success_eps = [ep for ep in group_episodes if ep["success"]]
    failure_eps = [ep for ep in group_episodes if not ep["success"]]
    accuracy = mean([ep["accuracy"] for ep in group_episodes])
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
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100 if accuracy is not None else None,
        "avg_total_completion_time_all_episodes_s": mean([ep["total_completion_time_s"] for ep in group_episodes]),
        "avg_model_inference_calls_all_episodes": mean([ep["model_inference_calls"] for ep in group_episodes]),
        "avg_model_forward_latency_per_call_all_episodes_s": avg_model_forward_latency_per_call(group_episodes),
        "avg_action_jitter_metrics_all_episodes": avg_action_jitter_metrics(group_episodes),
        "avg_total_completion_time_success_episodes_s": mean([ep["total_completion_time_s"] for ep in success_eps]),
        "avg_model_inference_calls_success_episodes": mean([ep["model_inference_calls"] for ep in success_eps]),
        "avg_model_forward_latency_per_call_success_episodes_s": avg_model_forward_latency_per_call(success_eps),
        "avg_action_jitter_metrics_success_episodes": avg_action_jitter_metrics(success_eps),
        "avg_total_completion_time_failure_episodes_s": mean([ep["total_completion_time_s"] for ep in failure_eps]),
        "avg_model_inference_calls_failure_episodes": mean([ep["model_inference_calls"] for ep in failure_eps]),
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
        "task_prompts": sorted({ep.get("task_prompt") for ep in group_episodes if ep.get("task_prompt")}),
        "episode_metrics": group_episodes,
        "video_paths": videos,
    }

task_groups = defaultdict(list)
suite_groups = defaultdict(list)
for ep in episodes:
    task_groups[f"{ep['suite']}/task_{ep['task_id']}"].append(ep)
    suite_groups[str(ep["suite"])].append(ep)

summary = {
    "n_action_steps": n_action_steps,
    "base_output_dir": str(base),
    "eval_info_files": [str(path) for path in eval_infos],
    "per_task": per_task,
    "per_group": {suite: summarize(eps, group_video_paths.get(suite, [])) for suite, eps in sorted(suite_groups.items())},
    "task_level": {name: summarize(eps) for name, eps in sorted(task_groups.items())},
    "suite_level": {suite: summarize(eps, group_video_paths.get(suite, [])) for suite, eps in sorted(suite_groups.items())},
    "all_suites_level": summarize(episodes, video_paths),
    "overall": {
        **summarize(episodes, video_paths),
        "pc_success": (sum(1 for ep in episodes if ep["success"]) / len(episodes) * 100) if episodes else None,
        "n_video_paths": len(video_paths),
    },
}

(base / "episode_metrics.json").write_text(json.dumps(episodes, indent=2))
(base / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
(base / "overlay_eval_summary.json").write_text(json.dumps(summary, indent=2))

csv_path = base / "metrics_summary.csv"
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
            writer.writerow({key: csv_value(metrics.get(key)) for key in fieldnames} | {"level": level, "name": name})

print(f"Wrote {base / 'overlay_eval_summary.json'}")
print(f"Collected {len(episodes)} episodes and {len(video_paths)} videos from {len(eval_infos)} eval_info files.")
if len(episodes) != expected_episode_count:
    print(f"Expected {expected_episode_count} episodes, but collected {len(episodes)}.", file=sys.stderr)
    sys.exit(2)
if len(video_paths) != expected_video_count:
    print(f"Expected {expected_video_count} videos, but collected {len(video_paths)}.", file=sys.stderr)
    sys.exit(2)
PY
}

write_sweep_manifest() {
  cat > "${SWEEP_OUTPUT_DIR}/full_eval_manifest.json" <<JSON
{
  "timestamp": "${TIMESTAMP}",
  "checkpoint_label": "${CHECKPOINT_LABEL}",
  "checkpoint_dir": "${CURRENT_CHECKPOINT_DIR}",
  "policy_path": "${POLICY_PATH}",
  "data_root": "${DATA_ROOT}",
  "hiva_coeff_sidecar": "${HIVA_COEFF_SIDECAR}",
  "hiva_coeff_summary": "${HIVA_COEFF_SUMMARY}",
  "hiva_duration_execution_map": "${HIVA_DURATION_EXECUTION_MAP}",
  "hiva_residual_inference_weight": "${HIVA_RESIDUAL_INFERENCE_WEIGHT}",
  "gpu_ids": "${GPU_IDS}",
  "task_ids_all": "${TASK_IDS_ALL}",
  "libero10_initial_task_ids": "${LIBERO10_INITIAL_TASK_IDS}",
  "n_action_steps": ${N_ACTION_STEPS},
  "n_episodes": ${N_EPISODES},
  "eval_batch_size": ${EVAL_BATCH_SIZE},
  "num_steps": ${NUM_STEPS},
  "chunk_size": ${CHUNK_SIZE},
  "max_episodes_rendered": ${MAX_EPISODES_RENDERED},
  "max_parallel_tasks": ${MAX_PARALLEL_TASKS},
  "seed": ${SEED},
  "expected_episode_count_per_action": ${EXPECTED_EPISODE_COUNT},
  "expected_video_count_per_action": ${EXPECTED_VIDEO_COUNT},
  "sweep_output_dir": "${SWEEP_OUTPUT_DIR}",
  "sweep_output_root": "${SWEEP_OUTPUT_ROOT}",
  "log_dir": "${LOG_DIR}"
}
JSON
}

run_checkpoint_tasks() {
  local base_output_dir="${SWEEP_OUTPUT_DIR}"
  mkdir -p "${base_output_dir}"

  echo "===== fixed n_action_steps=${N_ACTION_STEPS} full eval started at $(date) ====="
  echo "BASE_OUTPUT_DIR=${base_output_dir}"

  short_pids=()
  libero10_pids=()

  run_task_sequence "${base_output_dir}" "${N_ACTION_STEPS}" "libero_object" "${GPU_ARRAY[0]}" 0 1 2 3 4 5 6 7 8 9 &
  short_pids+=("$!")
  run_task_sequence "${base_output_dir}" "${N_ACTION_STEPS}" "libero_goal" "${GPU_ARRAY[1]}" 0 1 2 3 4 5 6 7 8 9 &
  short_pids+=("$!")
  run_task_sequence "${base_output_dir}" "${N_ACTION_STEPS}" "libero_spatial" "${GPU_ARRAY[2]}" 0 1 2 3 4 5 6 7 8 9 &
  short_pids+=("$!")
  run_task_sequence "${base_output_dir}" "${N_ACTION_STEPS}" "libero_10" "${GPU_ARRAY[3]}" 0 1 2 3 4 5 6 &
  libero10_pids+=("$!")

  status=0
  for pid in "${short_pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done

  if [[ "${status}" -ne 0 ]]; then
    for pid in "${libero10_pids[@]}"; do
      wait "${pid}" || true
    done
    echo "n_action_steps=${N_ACTION_STEPS} short-suite evaluation failed. Logs are under ${LOG_DIR}" >&2
    return "${status}"
  fi

  run_suite "${base_output_dir}" "${N_ACTION_STEPS}" "libero_10" "${GPU_ARRAY[0]}" "[7]" &
  libero10_pids+=("$!")
  run_suite "${base_output_dir}" "${N_ACTION_STEPS}" "libero_10" "${GPU_ARRAY[1]}" "[8]" &
  libero10_pids+=("$!")
  run_suite "${base_output_dir}" "${N_ACTION_STEPS}" "libero_10" "${GPU_ARRAY[2]}" "[9]" &
  libero10_pids+=("$!")

  for pid in "${libero10_pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done

  if [[ "${status}" -ne 0 ]]; then
    echo "n_action_steps=${N_ACTION_STEPS} evaluation failed. Logs are under ${LOG_DIR}" >&2
    return "${status}"
  fi

  write_checkpoint_summary "${base_output_dir}"
  echo "===== fixed n_action_steps=${N_ACTION_STEPS} full eval finished at $(date) ====="
}

run_one_checkpoint() {
  CURRENT_CHECKPOINT_DIR="$1"
  local checkpoint_step
  checkpoint_step="$(basename "${CURRENT_CHECKPOINT_DIR}")"
  CHECKPOINT_LABEL="${CHECKPOINT_LABEL_PREFIX}_ckpt${checkpoint_step}"
  POLICY_PATH="${CURRENT_CHECKPOINT_DIR}/pretrained_model"
  SWEEP_OUTPUT_DIR="${SWEEP_OUTPUT_ROOT}/${CHECKPOINT_LABEL}"
  mkdir -p "${SWEEP_OUTPUT_DIR}"

  echo "===== checkpoint ${CHECKPOINT_LABEL} full staged eval started at $(date) ====="
  echo "CURRENT_CHECKPOINT_DIR=${CURRENT_CHECKPOINT_DIR}"
  echo "POLICY_PATH=${POLICY_PATH}"
  echo "SWEEP_OUTPUT_DIR=${SWEEP_OUTPUT_DIR}"

  if [[ ! -d "${POLICY_PATH}" ]]; then
    echo "Missing POLICY_PATH directory: ${POLICY_PATH}" >&2
    exit 1
  fi

  write_sweep_manifest

  run_checkpoint_tasks

  echo "===== checkpoint ${CHECKPOINT_LABEL} full staged eval finished at $(date) ====="
}

echo "===== ws1 full staged SmolVLA eval started at $(date) ====="
echo "Host: $(hostname)"
echo "CHECKPOINT_LABEL_PREFIX=${CHECKPOINT_LABEL_PREFIX}"
echo "CHECKPOINT_DIRS=${CHECKPOINT_DIRS[*]}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "HIVA_COEFF_SIDECAR=${HIVA_COEFF_SIDECAR}"
echo "HIVA_COEFF_SUMMARY=${HIVA_COEFF_SUMMARY}"
echo "HIVA_DURATION_EXECUTION_MAP=${HIVA_DURATION_EXECUTION_MAP}"
echo "HIVA_RESIDUAL_INFERENCE_WEIGHT=${HIVA_RESIDUAL_INFERENCE_WEIGHT}"
echo "LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH}"
echo "GPU_IDS=${GPU_IDS}"
echo "TASK_IDS_ALL=${TASK_IDS_ALL}"
echo "LIBERO10_INITIAL_TASK_IDS=${LIBERO10_INITIAL_TASK_IDS}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "N_EPISODES=${N_EPISODES}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "NUM_STEPS=${NUM_STEPS}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "MAX_EPISODES_RENDERED=${MAX_EPISODES_RENDERED}"
echo "MAX_PARALLEL_TASKS=${MAX_PARALLEL_TASKS}"
echo "SEED=${SEED}"
echo "EXPECTED_EPISODE_COUNT=${EXPECTED_EPISODE_COUNT}"
echo "EXPECTED_VIDEO_COUNT=${EXPECTED_VIDEO_COUNT}"
echo "SWEEP_OUTPUT_ROOT=${SWEEP_OUTPUT_ROOT}"
echo "LOG_DIR=${LOG_DIR}"

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

for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"; do
  run_one_checkpoint "${checkpoint_dir}"
done

echo "===== ws1 full staged SmolVLA eval finished at $(date) ====="
echo "Outputs are under: ${SWEEP_OUTPUT_ROOT}"
