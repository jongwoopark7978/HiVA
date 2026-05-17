#!/usr/bin/env bash
set -euo pipefail

# Partial LIBERO eval sweep for every checkpoint in the stage-0 LP-MT HiVA
# coefficient run on bigbrain.
#
# Defaults:
#   - checkpoints: all numeric checkpoint directories, plus "last" if present
#   - GPUs: 6,7
#   - eval.batch_size: 10
#   - checkpoint parallelism: one checkpoint per GPU
#   - task coverage: all 4 LIBERO suites, task_ids [0..9], 10 episodes per task
#
# Example:
#   nohup setsid bash server_scripts/bigbrain/eval_lpmt_stage0_v5_all_ckpts_partial_gpu6_7_bs10_bigbrain.sh \
#     > outputs/eval_logs/eval_lpmt_stage0_v5_all_ckpts_partial_bigbrain_$(date +%Y%m%d_%H%M%S).outer.log 2>&1 < /dev/null &

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"

TRAIN_DIR="${TRAIN_DIR:-/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigcornea_b64_s0p5_20260514_182207}"
MODEL_TAG="${MODEL_TAG:-$(basename "${TRAIN_DIR}")}"
CKPTS_OVERRIDE="${CKPTS_OVERRIDE:-}"

GPU_IDS="${GPU_IDS:-6,7}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -eq 0 ]]; then
  echo "GPU_IDS produced no GPU entries: ${GPU_IDS}" >&2
  exit 1
fi

TASK_IDS_ALL="${TASK_IDS_ALL:-[0,1,2,3,4,5,6,7,8,9]}"
N_EPISODES="${N_EPISODES:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-1}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-1}"
EVAL_CHECKPOINTS_IN_PARALLEL="${EVAL_CHECKPOINTS_IN_PARALLEL:-1}"
SUITES_CSV="${SUITES_CSV:-libero_object,libero_goal,libero_spatial,libero_10}"
IFS=',' read -r -a SUITES <<< "${SUITES_CSV}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigbrain/add_disk0/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"
INCLUDE_HIVA_COEFF_ARGS="${INCLUDE_HIVA_COEFF_ARGS:-1}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/jongwoo_hf_datasets_cache}"
mkdir -p "${HF_DATASETS_CACHE}"

N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-15}"
NUM_STEPS="${NUM_STEPS:-10}"
HIVA_DURATION_EXECUTION_MAP="${HIVA_DURATION_EXECUTION_MAP:-}"
HIVA_RESIDUAL_INFERENCE_WEIGHT="${HIVA_RESIDUAL_INFERENCE_WEIGHT:-}"
RENAME_MAP='{"observation.images.image":"observation.images.agentview","observation.images.image2":"observation.images.wrist"}'

SWEEP_OUTPUT_DIR="${SWEEP_OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/lpmt_stage0_v5_all_ckpts_partial_bigbrain_bs${EVAL_BATCH_SIZE}_${TIMESTAMP}}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/eval_lpmt_stage0_v5_all_ckpts_partial_bigbrain_bs${EVAL_BATCH_SIZE}_${TIMESTAMP}.queue.log}"

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Missing required directory: ${path}" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

collect_ckpts() {
  if [[ -n "${CKPTS_OVERRIDE}" ]]; then
    printf '%s\n' ${CKPTS_OVERRIDE}
    return
  fi

  local ckpt
  local ckpt_real
  local last_real
  local seen_real_paths=""

  while IFS= read -r ckpt; do
    ckpt_real="$(readlink -f "${TRAIN_DIR}/checkpoints/${ckpt}")"
    seen_real_paths="${seen_real_paths}"$'\n'"${ckpt_real}"
    echo "${ckpt}"
  done < <(
    find "${TRAIN_DIR}/checkpoints" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' \
      | grep -E '^[0-9]+$' \
      | sort -n
  )

  if [[ -e "${TRAIN_DIR}/checkpoints/last/pretrained_model" ]]; then
    last_real="$(readlink -f "${TRAIN_DIR}/checkpoints/last")"
    if ! grep -Fxq "${last_real}" <<< "${seen_real_paths}"; then
      echo "last"
    else
      echo "Skipping duplicate checkpoint alias last -> ${last_real}" >&2
    fi
  fi
}

task_ids_for() {
  case "$1" in
    libero_object) echo "${OBJECT_TASK_IDS:-${TASK_IDS_ALL}}" ;;
    libero_goal) echo "${GOAL_TASK_IDS:-${TASK_IDS_ALL}}" ;;
    libero_spatial) echo "${SPATIAL_TASK_IDS:-${TASK_IDS_ALL}}" ;;
    libero_10) echo "${LIBERO10_TASK_IDS:-${TASK_IDS_ALL}}" ;;
    *) echo "${TASK_IDS:-${TASK_IDS_ALL}}" ;;
  esac
}

task_id_values() {
  TASK_IDS_EXPR="$1" python - <<'PY'
import ast
import os

parsed = ast.literal_eval(os.environ["TASK_IDS_EXPR"])
if isinstance(parsed, int):
    parsed = [parsed]
print(" ".join(str(int(item)) for item in parsed))
PY
}

count_eval_units() {
  SUITES_CSV="${SUITES_CSV}" \
  TASK_IDS_ALL="${TASK_IDS_ALL}" \
  OBJECT_TASK_IDS="${OBJECT_TASK_IDS:-}" \
  GOAL_TASK_IDS="${GOAL_TASK_IDS:-}" \
  SPATIAL_TASK_IDS="${SPATIAL_TASK_IDS:-}" \
  LIBERO10_TASK_IDS="${LIBERO10_TASK_IDS:-}" \
  N_EPISODES="${N_EPISODES}" \
  MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED}" \
  python - <<'PY'
import ast
import os

def ids_for(suite):
    env_name = {
        "libero_object": "OBJECT_TASK_IDS",
        "libero_goal": "GOAL_TASK_IDS",
        "libero_spatial": "SPATIAL_TASK_IDS",
        "libero_10": "LIBERO10_TASK_IDS",
    }.get(suite)
    expr = os.environ.get(env_name, "") if env_name else ""
    if not expr:
        expr = os.environ["TASK_IDS_ALL"]
    parsed = ast.literal_eval(expr)
    if isinstance(parsed, int):
        parsed = [parsed]
    return parsed

suites = [suite for suite in os.environ["SUITES_CSV"].split(",") if suite]
n_tasks = sum(len(ids_for(suite)) for suite in suites)
n_episodes = int(os.environ["N_EPISODES"])
max_videos = int(os.environ["MAX_EPISODES_RENDERED"])
print(f"{n_tasks * n_episodes} {n_tasks * max(0, min(n_episodes, max_videos))}")
PY
}

write_sweep_manifest() {
  local ckpts=("$@")
  SWEEP_OUTPUT_DIR="${SWEEP_OUTPUT_DIR}" TRAIN_DIR="${TRAIN_DIR}" CKPTS="${ckpts[*]}" python - <<'PY'
import json
import os
from pathlib import Path

base = Path(os.environ["SWEEP_OUTPUT_DIR"])
ckpts = os.environ["CKPTS"].split()
manifest = {
    "train_dir": os.environ["TRAIN_DIR"],
    "checkpoint_order": ckpts,
    "checkpoint_eval_dirs": {ckpt: str(base / f"ckpt_{ckpt}") for ckpt in ckpts},
}
base.mkdir(parents=True, exist_ok=True)
(base / "checkpoint_eval_manifest.json").write_text(json.dumps(manifest, indent=2))
PY
}

write_summary() {
  local base_output_dir="$1"
  local expected_episode_count="$2"
  local expected_video_count="$3"

  BASE_OUTPUT_DIR="${base_output_dir}" \
  EXPECTED_EPISODE_COUNT="${expected_episode_count}" \
  EXPECTED_VIDEO_COUNT="${expected_video_count}" \
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

def summarize_episodes(group_episodes, videos=None):
    videos = videos or []
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
        "video_paths": videos,
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

run_suite_task() {
  local checkpoint_label="$1"
  local policy_path="$2"
  local base_output_dir="$3"
  local suite="$4"
  local task_ids="$5"
  local gpu_id="$6"

  local safe_task_ids="${task_ids//[^0-9A-Za-z_-]/_}"
  local run_name="hiva_coeff_bigbrain_${checkpoint_label}_${suite}_taskids_${safe_task_ids}_${TIMESTAMP}"
  local log_name="${run_name}"
  if [[ "${#log_name}" -gt 180 ]]; then
    local run_hash
    local short_label
    run_hash="$(printf '%s' "${run_name}" | sha1sum | cut -c1-12)"
    short_label="$(printf '%s' "${checkpoint_label}" | cut -c1-80)"
    log_name="hiva_coeff_bigbrain_${short_label}_${suite}_${safe_task_ids}_${run_hash}"
  fi
  local output_dir="${base_output_dir}/${suite}_taskids_${safe_task_ids}"
  local log_path="${LOG_DIR}/${log_name}.log"

  if [[ -f "${output_dir}/eval_info.json" ]]; then
    echo "[${checkpoint_label}] Skipping completed ${suite} task_ids=${task_ids}: ${output_dir}"
    return
  fi

  echo "[${checkpoint_label}] Starting ${suite} task_ids=${task_ids} on GPU ${gpu_id}"
  echo "[${checkpoint_label}] Log: ${log_path}"
  echo "[${checkpoint_label}] Output dir: ${output_dir}"

  local policy_extra_args=(
    --policy.use_duration_head=false
  )
  if [[ "${INCLUDE_HIVA_COEFF_ARGS}" == "1" ]]; then
    policy_extra_args+=(
      --policy.hiva_coeff_sidecar_path="${HIVA_COEFF_SIDECAR}"
      --policy.hiva_coeff_sidecar_summary_path="${HIVA_COEFF_SUMMARY}"
      --policy.hiva_duration_execution_map="${HIVA_DURATION_EXECUTION_MAP}"
    )
    if [[ -n "${HIVA_RESIDUAL_INFERENCE_WEIGHT}" ]]; then
      policy_extra_args+=(
        --policy.hiva_residual_inference_weight="${HIVA_RESIDUAL_INFERENCE_WEIGHT}"
      )
    fi
  fi

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  MUJOCO_EGL_DEVICE_ID="${gpu_id}" \
  lerobot-eval \
    --policy.path="${policy_path}" \
    --policy.device=cuda \
    --policy.num_steps="${NUM_STEPS}" \
    --policy.chunk_size="${CHUNK_SIZE}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    "${policy_extra_args[@]}" \
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

run_worker() {
  local worker_idx="$1"
  local gpu_id="$2"
  local checkpoint_label="$3"
  local policy_path="$4"
  local base_output_dir="$5"
  shift 5
  local jobs=("$@")

  local job_idx
  for ((job_idx = worker_idx; job_idx < ${#jobs[@]}; job_idx += ${#GPU_ARRAY[@]})); do
    IFS='|' read -r suite task_ids <<< "${jobs[$job_idx]}"
    run_suite_task "${checkpoint_label}" "${policy_path}" "${base_output_dir}" "${suite}" "${task_ids}" "${gpu_id}"
  done
}

run_checkpoint_eval() {
  local ckpt="$1"
  local policy_path="${TRAIN_DIR}/checkpoints/${ckpt}/pretrained_model"
  local checkpoint_label="${MODEL_TAG}_ckpt_${ckpt}_10eps_bs${EVAL_BATCH_SIZE}"
  local base_output_dir="${SWEEP_OUTPUT_DIR}/ckpt_${ckpt}"

  if [[ -f "${base_output_dir}/overlay_eval_summary.json" ]]; then
    echo "===== $(date) skipping completed ${checkpoint_label}: ${base_output_dir} ====="
    return
  fi

  require_dir "${policy_path}"
  mkdir -p "${base_output_dir}"

  mapfile -t expected_counts < <(count_eval_units)
  local expected_episode_count="${EXPECTED_EPISODE_COUNT:-${expected_counts[0]%% *}}"
  local expected_video_count="${EXPECTED_VIDEO_COUNT:-${expected_counts[0]##* }}"

  local jobs=()
  local suite task_ids task_id
  for suite in "${SUITES[@]}"; do
    task_ids="$(task_ids_for "${suite}")"
    for task_id in $(task_id_values "${task_ids}"); do
      jobs+=("${suite}|[${task_id}]")
    done
  done

  echo "===== $(date) evaluating ${checkpoint_label} ====="
  echo "POLICY_PATH=${policy_path}"
  echo "BASE_OUTPUT_DIR=${base_output_dir}"
  echo "JOBS=${#jobs[@]}"
  echo "EXPECTED_EPISODE_COUNT=${expected_episode_count}"
  echo "EXPECTED_VIDEO_COUNT=${expected_video_count}"

  local pids=()
  local worker_idx
  for worker_idx in "${!GPU_ARRAY[@]}"; do
    run_worker "${worker_idx}" "${GPU_ARRAY[$worker_idx]}" "${checkpoint_label}" "${policy_path}" "${base_output_dir}" "${jobs[@]}" &
    pids+=("$!")
  done

  local status=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done

  if [[ "${status}" -ne 0 ]]; then
    echo "Checkpoint ${checkpoint_label} evaluation failed." >&2
    exit "${status}"
  fi

  write_summary "${base_output_dir}" "${expected_episode_count}" "${expected_video_count}"
  echo "===== $(date) finished ${checkpoint_label} ====="
}

run_checkpoint_eval_on_gpu() {
  local ckpt="$1"
  local gpu_id="$2"
  local policy_path="${TRAIN_DIR}/checkpoints/${ckpt}/pretrained_model"
  local checkpoint_label="${MODEL_TAG}_ckpt_${ckpt}_10eps_bs${EVAL_BATCH_SIZE}"
  local base_output_dir="${SWEEP_OUTPUT_DIR}/ckpt_${ckpt}"

  if [[ -f "${base_output_dir}/overlay_eval_summary.json" ]]; then
    echo "===== $(date) skipping completed ${checkpoint_label}: ${base_output_dir} ====="
    return
  fi

  require_dir "${policy_path}"
  mkdir -p "${base_output_dir}"

  mapfile -t expected_counts < <(count_eval_units)
  local expected_episode_count="${EXPECTED_EPISODE_COUNT:-${expected_counts[0]%% *}}"
  local expected_video_count="${EXPECTED_VIDEO_COUNT:-${expected_counts[0]##* }}"

  local jobs=()
  local suite task_ids task_id
  for suite in "${SUITES[@]}"; do
    task_ids="$(task_ids_for "${suite}")"
    for task_id in $(task_id_values "${task_ids}"); do
      jobs+=("${suite}|[${task_id}]")
    done
  done

  echo "===== $(date) evaluating ${checkpoint_label} on GPU ${gpu_id} ====="
  echo "POLICY_PATH=${policy_path}"
  echo "BASE_OUTPUT_DIR=${base_output_dir}"
  echo "JOBS=${#jobs[@]}"
  echo "EXPECTED_EPISODE_COUNT=${expected_episode_count}"
  echo "EXPECTED_VIDEO_COUNT=${expected_video_count}"

  local job suite_name task_ids_expr
  for job in "${jobs[@]}"; do
    IFS='|' read -r suite_name task_ids_expr <<< "${job}"
    run_suite_task "${checkpoint_label}" "${policy_path}" "${base_output_dir}" "${suite_name}" "${task_ids_expr}" "${gpu_id}"
  done

  write_summary "${base_output_dir}" "${expected_episode_count}" "${expected_video_count}"
  echo "===== $(date) finished ${checkpoint_label} on GPU ${gpu_id} ====="
}

main() {
  exec > >(tee -a "${QUEUE_LOG}") 2>&1

  require_dir "${TRAIN_DIR}/checkpoints"
  require_dir "${DATA_ROOT}"
  if [[ "${INCLUDE_HIVA_COEFF_ARGS}" == "1" ]]; then
    require_file "${HIVA_COEFF_SIDECAR}"
    require_file "${HIVA_COEFF_SUMMARY}"
  fi

  mapfile -t ckpts < <(collect_ckpts)
  if [[ "${#ckpts[@]}" -eq 0 ]]; then
    echo "No checkpoints found under ${TRAIN_DIR}/checkpoints" >&2
    exit 1
  fi
  write_sweep_manifest "${ckpts[@]}"

  echo "===== LP-MT stage0 v5 all-checkpoint partial eval started at $(date) ====="
  echo "TIMESTAMP=${TIMESTAMP}"
  echo "TRAIN_DIR=${TRAIN_DIR}"
  echo "MODEL_TAG=${MODEL_TAG}"
  echo "CKPTS=${ckpts[*]}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
  echo "EVAL_CHECKPOINTS_IN_PARALLEL=${EVAL_CHECKPOINTS_IN_PARALLEL}"
  echo "N_EPISODES=${N_EPISODES}"
  echo "TASK_IDS_ALL=${TASK_IDS_ALL}"
  echo "SUITES_CSV=${SUITES_CSV}"
  echo "MAX_PARALLEL_TASKS=${MAX_PARALLEL_TASKS}"
  echo "MAX_EPISODES_RENDERED=${MAX_EPISODES_RENDERED}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "HIVA_COEFF_SIDECAR=${HIVA_COEFF_SIDECAR}"
  echo "HIVA_COEFF_SUMMARY=${HIVA_COEFF_SUMMARY}"
  echo "INCLUDE_HIVA_COEFF_ARGS=${INCLUDE_HIVA_COEFF_ARGS}"
  echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
  echo "CHUNK_SIZE=${CHUNK_SIZE}"
  echo "NUM_STEPS=${NUM_STEPS}"
  echo "HIVA_DURATION_EXECUTION_MAP=${HIVA_DURATION_EXECUTION_MAP}"
  echo "HIVA_RESIDUAL_INFERENCE_WEIGHT=${HIVA_RESIDUAL_INFERENCE_WEIGHT}"
  echo "SWEEP_OUTPUT_DIR=${SWEEP_OUTPUT_DIR}"
  echo "QUEUE_LOG=${QUEUE_LOG}"

  local ckpt
  if [[ "${EVAL_CHECKPOINTS_IN_PARALLEL}" == "1" ]]; then
    local pids=()
    local status=0
    local idx
    local gpu_id
    for idx in "${!ckpts[@]}"; do
      ckpt="${ckpts[$idx]}"
      gpu_id="${GPU_ARRAY[$((idx % ${#GPU_ARRAY[@]}))]}"
      run_checkpoint_eval_on_gpu "${ckpt}" "${gpu_id}" &
      pids+=("$!")

      if [[ "${#pids[@]}" -ge "${#GPU_ARRAY[@]}" ]]; then
        local pid
        for pid in "${pids[@]}"; do
          if ! wait "${pid}"; then
            status=1
          fi
        done
        pids=()
        if [[ "${status}" -ne 0 ]]; then
          exit "${status}"
        fi
      fi
    done

    local pid
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        status=1
      fi
    done
    if [[ "${status}" -ne 0 ]]; then
      exit "${status}"
    fi
  else
    for ckpt in "${ckpts[@]}"; do
      run_checkpoint_eval "${ckpt}"
    done
  fi

  echo "===== LP-MT stage0 v5 all-checkpoint partial eval finished at $(date) ====="
}

main "$@"
