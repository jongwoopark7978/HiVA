#!/usr/bin/env bash
set -euo pipefail

# Resume incomplete b256 S=0.5 full LIBERO evals with dynamic split-GPU
# scheduling:
#   GPU4: libero_object + libero_spatial, three total task eval processes.
#   GPU5: libero_goal + libero_10, three total task eval processes.
#
# The script resumes into the existing eval roots and skips any task directory
# that already contains eval_info.json.
#
# Dry run:
#   DRY_RUN=1 bash server_scripts/bigbrain/resume_full_b256_s0p5_dynamic_split_gpu4_5_bigbrain.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"

TRAIN_DIR="${TRAIN_DIR:-/nfs/bigcornea.cs.stonybrook.edu/add_disk3/jongwoopark/HiVA_train/finetuning_stage0/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigbrain_b256_g2_s0p5_steps5000_20260515_233203}"
MODEL_TAG="${MODEL_TAG:-$(basename "${TRAIN_DIR}")}"
CKPTS_OVERRIDE="${CKPTS_OVERRIDE:-}"

GPU_OBJECT_SPATIAL="${GPU_OBJECT_SPATIAL:-4}"
GPU_GOAL_LIBERO10="${GPU_GOAL_LIBERO10:-5}"
PER_GPU_CONCURRENCY="${PER_GPU_CONCURRENCY:-3}"

TASK_IDS_ALL="${TASK_IDS_ALL:-0 1 2 3 4 5 6 7 8 9}"
N_EPISODES="${N_EPISODES:-50}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-50}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-1}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-1}"
EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT:-2000}"
EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT:-40}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigbrain/add_disk0/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
source "/home/jongwoopark/lerobot/server_scripts/common_hf_cache.sh"
setup_hf_datasets_cache

N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-15}"
NUM_STEPS="${NUM_STEPS:-10}"
HIVA_DURATION_EXECUTION_MAP="${HIVA_DURATION_EXECUTION_MAP:-}"
HIVA_RESIDUAL_INFERENCE_WEIGHT="${HIVA_RESIDUAL_INFERENCE_WEIGHT:-}"
RENAME_MAP='{"observation.images.image":"observation.images.agentview","observation.images.image2":"observation.images.wrist"}'

QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/resume_full_b256_s0p5_dynamic_split_gpu4_5_${TIMESTAMP}.queue.log}"

DEFAULT_CKPTS=(003125 003250 003375 003500 003625 003750 003875 004000 004375 005000)

ROOT_003125_TO_003500="${ROOT_003125_TO_003500:-${REPO_ROOT}/outputs/eval/full_bigbrain_gpu6_smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigbrain_b256_g2_s0p5_steps5000_20260515_233203_ckpts_003125_003250_003375_003500_50eps_bs50_20260517_123746}"
ROOT_003625_TO_004000="${ROOT_003625_TO_004000:-${REPO_ROOT}/outputs/eval/full_bigbrain_gpu5_smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigbrain_b256_g2_s0p5_steps5000_20260515_233203_ckpts_003625_003750_003875_004000_50eps_bs50_20260517_125234}"
ROOT_004375_TO_005000="${ROOT_004375_TO_005000:-${REPO_ROOT}/outputs/eval/full_bigbrain_gpu3_mixed_b256_p7_s0p5_ckpts_004375_005000_50eps_bs50_20260517_125708/b256_smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigbrain_b256_g2_s0p5_steps5000_20260515_233203}"

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

ckpts_to_run() {
  if [[ -n "${CKPTS_OVERRIDE}" ]]; then
    printf '%s\n' ${CKPTS_OVERRIDE}
  else
    printf '%s\n' "${DEFAULT_CKPTS[@]}"
  fi
}

eval_root_for_ckpt() {
  case "$1" in
    003125|003250|003375|003500) echo "${ROOT_003125_TO_003500}" ;;
    003625|003750|003875|004000) echo "${ROOT_003625_TO_004000}" ;;
    004375|005000) echo "${ROOT_004375_TO_005000}" ;;
    *)
      echo "No existing eval root is configured for checkpoint $1" >&2
      return 1
      ;;
  esac
}

task_output_dir() {
  local base_output_dir="$1"
  local suite="$2"
  local task_id="$3"
  echo "${base_output_dir}/${suite}_taskids__${task_id}_"
}

missing_task_ids() {
  local base_output_dir="$1"
  local suite="$2"
  local task_id
  for task_id in ${TASK_IDS_ALL}; do
    if [[ ! -f "$(task_output_dir "${base_output_dir}" "${suite}" "${task_id}")/eval_info.json" ]]; then
      echo "${task_id}"
    fi
  done
}

completed_task_count() {
  local base_output_dir="$1"
  find "${base_output_dir}" -maxdepth 2 -name eval_info.json | wc -l
}

archive_partial_output() {
  local output_dir="$1"
  if [[ ! -e "${output_dir}" || -f "${output_dir}/eval_info.json" ]]; then
    return
  fi

  local archive="${output_dir}.partial_before_resume_${TIMESTAMP}"
  local idx=1
  while [[ -e "${archive}" ]]; do
    archive="${output_dir}.partial_before_resume_${TIMESTAMP}_${idx}"
    idx=$((idx + 1))
  done

  echo "Archiving incomplete task output: ${output_dir} -> ${archive}"
  mv "${output_dir}" "${archive}"
}

write_checkpoint_manifest() {
  local eval_root="$1"
  shift
  local ckpts=("$@")

  EVAL_ROOT="${eval_root}" TRAIN_DIR="${TRAIN_DIR}" CKPTS="${ckpts[*]}" python - <<'PY'
import json
import os
from pathlib import Path

base = Path(os.environ["EVAL_ROOT"])
ckpts = os.environ["CKPTS"].split()
manifest = {
    "train_dir": os.environ["TRAIN_DIR"],
    "checkpoint_order": ckpts,
    "checkpoint_eval_dirs": {ckpt: str(base / f"ckpt_{ckpt}") for ckpt in ckpts},
}
base.mkdir(parents=True, exist_ok=True)
(base / "checkpoint_eval_manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"Wrote manifest: {base / 'checkpoint_eval_manifest.json'}")
PY
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
if len(episodes) != expected_episode_count:
    print(f"Expected {expected_episode_count} episodes, but collected {len(episodes)}.", file=sys.stderr)
    sys.exit(2)
if len(video_paths) != expected_video_count:
    print(f"Expected {expected_video_count} videos, but collected {len(video_paths)} video paths.", file=sys.stderr)
    sys.exit(2)
PY
}

run_suite_task() {
  local checkpoint_label="$1"
  local policy_path="$2"
  local base_output_dir="$3"
  local suite="$4"
  local task_id="$5"
  local gpu_id="$6"

  local task_ids="[${task_id}]"
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

  local output_dir
  output_dir="$(task_output_dir "${base_output_dir}" "${suite}" "${task_id}")"
  local log_path="${LOG_DIR}/${log_name}.log"

  if [[ -f "${output_dir}/eval_info.json" ]]; then
    echo "[${checkpoint_label}] Skipping completed ${suite} task${task_id}: ${output_dir}"
    return 0
  fi

  archive_partial_output "${output_dir}"

  echo "[${checkpoint_label}] Starting ${suite} task${task_id} on GPU ${gpu_id}"
  echo "[${checkpoint_label}] Log: ${log_path}"
  echo "[${checkpoint_label}] Output dir: ${output_dir}"

  local policy_extra_args=(
    --policy.use_duration_head=false
    --policy.hiva_coeff_sidecar_path="${HIVA_COEFF_SIDECAR}"
    --policy.hiva_coeff_sidecar_summary_path="${HIVA_COEFF_SUMMARY}"
    --policy.hiva_duration_execution_map="${HIVA_DURATION_EXECUTION_MAP}"
  )
  if [[ -n "${HIVA_RESIDUAL_INFERENCE_WEIGHT}" ]]; then
    policy_extra_args+=(
      --policy.hiva_residual_inference_weight="${HIVA_RESIDUAL_INFERENCE_WEIGHT}"
    )
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

schedule_gpu_queue() {
  local checkpoint_label="$1"
  local policy_path="$2"
  local base_output_dir="$3"
  local gpu_id="$4"
  local concurrency="$5"
  local queue_name="$6"
  shift 6
  local jobs=("$@")

  echo "[${checkpoint_label}] ${queue_name} GPU=${gpu_id} total_concurrency=${concurrency} jobs=${jobs[*]:-(none)}"
  if [[ "${DRY_RUN}" == "1" || "${#jobs[@]}" -eq 0 ]]; then
    return 0
  fi

  local active=0
  local next=0
  local status=0
  local job
  local suite
  local task_id

  while [[ "${next}" -lt "${#jobs[@]}" || "${active}" -gt 0 ]]; do
    while [[ "${active}" -lt "${concurrency}" && "${next}" -lt "${#jobs[@]}" ]]; do
      job="${jobs[$next]}"
      IFS='|' read -r suite task_id <<< "${job}"
      run_suite_task "${checkpoint_label}" "${policy_path}" "${base_output_dir}" "${suite}" "${task_id}" "${gpu_id}" &
      active=$((active + 1))
      next=$((next + 1))
    done

    if [[ "${active}" -gt 0 ]]; then
      if ! wait -n; then
        status=1
      fi
      active=$((active - 1))
    fi
  done

  return "${status}"
}

checkpoint_is_complete() {
  local base_output_dir="$1"
  [[ "$(completed_task_count "${base_output_dir}")" -eq 40 ]]
}

print_checkpoint_state() {
  local ckpt="$1"
  local eval_root
  eval_root="$(eval_root_for_ckpt "${ckpt}")"
  local base_output_dir="${eval_root}/ckpt_${ckpt}"
  local policy_path="${TRAIN_DIR}/checkpoints/${ckpt}/pretrained_model"

  require_dir "${policy_path}"
  mkdir -p "${base_output_dir}"

  mapfile -t object_missing < <(missing_task_ids "${base_output_dir}" libero_object)
  mapfile -t spatial_missing < <(missing_task_ids "${base_output_dir}" libero_spatial)
  mapfile -t goal_missing < <(missing_task_ids "${base_output_dir}" libero_goal)
  mapfile -t libero10_missing < <(missing_task_ids "${base_output_dir}" libero_10)

  local gpu4_jobs=()
  local gpu5_jobs=()
  local task_id
  local suite
  for task_id in ${TASK_IDS_ALL}; do
    for suite in libero_object libero_spatial; do
      if [[ ! -f "$(task_output_dir "${base_output_dir}" "${suite}" "${task_id}")/eval_info.json" ]]; then
        gpu4_jobs+=("${suite}|${task_id}")
      fi
    done
    for suite in libero_goal libero_10; do
      if [[ ! -f "$(task_output_dir "${base_output_dir}" "${suite}" "${task_id}")/eval_info.json" ]]; then
        gpu5_jobs+=("${suite}|${task_id}")
      fi
    done
  done

  echo "===== $(date) checkpoint ckpt_${ckpt} ====="
  echo "EVAL_ROOT=${eval_root}"
  echo "BASE_OUTPUT_DIR=${base_output_dir}"
  echo "POLICY_PATH=${policy_path}"
  echo "completed_task_dirs=$(completed_task_count "${base_output_dir}")/40"
  echo "libero_object_missing=${object_missing[*]:-(none)}"
  echo "libero_spatial_missing=${spatial_missing[*]:-(none)}"
  echo "libero_goal_missing=${goal_missing[*]:-(none)}"
  echo "libero_10_missing=${libero10_missing[*]:-(none)}"
}

build_gpu_jobs() {
  local base_output_dir="$1"
  shift
  local suites=("$@")

  local task_id
  local suite
  for task_id in ${TASK_IDS_ALL}; do
    for suite in "${suites[@]}"; do
      if [[ ! -f "$(task_output_dir "${base_output_dir}" "${suite}" "${task_id}")/eval_info.json" ]]; then
        echo "${suite}|${task_id}"
      fi
    done
  done
}

run_gpu_group_across_checkpoints() {
  local gpu_id="$1"
  local concurrency="$2"
  local queue_name="$3"
  shift 3
  local suites=("$@")

  local status=0
  local ckpt
  local eval_root
  local base_output_dir
  local policy_path
  local checkpoint_label
  local jobs

  echo "===== $(date) starting GPU ${gpu_id} queue ${queue_name} over checkpoints: ${ckpts[*]} ====="

  for ckpt in "${ckpts[@]}"; do
    eval_root="$(eval_root_for_ckpt "${ckpt}")"
    base_output_dir="${eval_root}/ckpt_${ckpt}"
    policy_path="${TRAIN_DIR}/checkpoints/${ckpt}/pretrained_model"
    checkpoint_label="${MODEL_TAG}_ckpt_${ckpt}_50eps_bs${EVAL_BATCH_SIZE}"

    require_dir "${policy_path}"
    mkdir -p "${base_output_dir}"
    mapfile -t jobs < <(build_gpu_jobs "${base_output_dir}" "${suites[@]}")

    echo "===== $(date) GPU ${gpu_id} ${queue_name} checkpoint ckpt_${ckpt} ====="
    echo "BASE_OUTPUT_DIR=${base_output_dir}"
    echo "POLICY_PATH=${policy_path}"
    echo "completed_task_dirs=$(completed_task_count "${base_output_dir}")/40"

    if [[ "${#jobs[@]}" -eq 0 ]]; then
      echo "[${checkpoint_label}] ${queue_name} has no missing jobs on GPU ${gpu_id}."
      continue
    fi

    if ! schedule_gpu_queue "${checkpoint_label}" "${policy_path}" "${base_output_dir}" "${gpu_id}" "${concurrency}" "${queue_name}" "${jobs[@]}"; then
      status=1
      echo "[${checkpoint_label}] ${queue_name} failed on GPU ${gpu_id}; continuing to next checkpoint." >&2
    fi
  done

  echo "===== $(date) finished GPU ${gpu_id} queue ${queue_name} ====="
  return "${status}"
}

write_completed_summaries() {
  local status=0
  local ckpt
  local eval_root
  local base_output_dir
  local checkpoint_label

  for ckpt in "${ckpts[@]}"; do
    eval_root="$(eval_root_for_ckpt "${ckpt}")"
    base_output_dir="${eval_root}/ckpt_${ckpt}"
    checkpoint_label="${MODEL_TAG}_ckpt_${ckpt}_50eps_bs${EVAL_BATCH_SIZE}"

    if checkpoint_is_complete "${base_output_dir}"; then
      echo "[${checkpoint_label}] All 40 task evals are present; writing summary."
      write_summary "${base_output_dir}"
    else
      echo "[${checkpoint_label}] Missing task evals remain after GPU queues; not writing summary." >&2
      status=1
    fi
  done

  return "${status}"
}

main() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "===== DRY RUN: dynamic split full eval resume ====="
  else
    exec > >(tee -a "${QUEUE_LOG}") 2>&1
    echo "===== dynamic split full eval resume started at $(date) ====="
  fi

  require_dir "${TRAIN_DIR}/checkpoints"
  require_dir "${DATA_ROOT}"
  require_file "${HIVA_COEFF_SIDECAR}"
  require_file "${HIVA_COEFF_SUMMARY}"

  mapfile -t ckpts < <(ckpts_to_run)
  echo "TIMESTAMP=${TIMESTAMP}"
  echo "DRY_RUN=${DRY_RUN}"
  echo "TRAIN_DIR=${TRAIN_DIR}"
  echo "CKPTS=${ckpts[*]}"
  echo "GPU_OBJECT_SPATIAL=${GPU_OBJECT_SPATIAL}"
  echo "GPU_GOAL_LIBERO10=${GPU_GOAL_LIBERO10}"
  echo "PER_GPU_CONCURRENCY=${PER_GPU_CONCURRENCY}"
  echo "N_EPISODES=${N_EPISODES}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
  echo "MAX_EPISODES_RENDERED=${MAX_EPISODES_RENDERED}"
  echo "QUEUE_LOG=${QUEUE_LOG}"

  if [[ "${DRY_RUN}" != "1" ]]; then
    write_checkpoint_manifest "${ROOT_003125_TO_003500}" 003125 003250 003375 003500
    write_checkpoint_manifest "${ROOT_003625_TO_004000}" 003625 003750 003875 004000
    write_checkpoint_manifest "${ROOT_004375_TO_005000}" 004375 005000
  fi

  local ckpt
  for ckpt in "${ckpts[@]}"; do
    print_checkpoint_state "${ckpt}"
  done

  local overall_status=0
  local pids=()

  run_gpu_group_across_checkpoints "${GPU_OBJECT_SPATIAL}" "${PER_GPU_CONCURRENCY}" "libero_object+libero_spatial" libero_object libero_spatial &
  pids+=("$!")
  run_gpu_group_across_checkpoints "${GPU_GOAL_LIBERO10}" "${PER_GPU_CONCURRENCY}" "libero_goal+libero_10" libero_goal libero_10 &
  pids+=("$!")

  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      overall_status=1
    fi
  done

  if [[ "${DRY_RUN}" != "1" ]]; then
    if ! write_completed_summaries; then
      overall_status=1
    fi
  fi

  if [[ "${overall_status}" -ne 0 ]]; then
    echo "One or more checkpoints failed." >&2
    exit "${overall_status}"
  fi

  echo "===== dynamic split full eval resume finished at $(date) ====="
}

main "$@"
