#!/usr/bin/env bash
set -euo pipefail

# Resume the task-sharded full LIBERO eval that was started on bigbrain for:
#   smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigbrain_b256_g2_s0p5_steps5000_20260515_233203
#
# Important: this script intentionally writes into the existing bigbrain output
# roots so another server can skip shards that are already complete.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
MODEL_TAG="smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigbrain_b256_g2_s0p5_steps5000_20260515_233203"
TRAIN_DIR="${TRAIN_DIR:-/nfs/bigcornea.cs.stonybrook.edu/add_disk3/jongwoopark/HiVA_train/finetuning_stage0/${MODEL_TAG}}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
source "/home/jongwoopark/lerobot/server_scripts/common_hf_cache.sh"
setup_hf_datasets_cache

LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"

N_EPISODES="${N_EPISODES:-50}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-1}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-1}"
EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT:-2000}"
EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT:-40}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-15}"
NUM_STEPS="${NUM_STEPS:-10}"
HIVA_DURATION_EXECUTION_MAP="${HIVA_DURATION_EXECUTION_MAP:-}"
RENAME_MAP='{"observation.images.image":"observation.images.agentview","observation.images.image2":"observation.images.wrist"}'

# The two original bigbrain roots plus the separate mixed root for ckpt_004375.
ROOT_LOW="${REPO_ROOT}/outputs/eval/full_bigbrain_gpu6_${MODEL_TAG}_ckpts_003125_003250_003375_003500_50eps_bs50_20260517_123746"
ROOT_HIGH="${REPO_ROOT}/outputs/eval/full_bigbrain_gpu5_${MODEL_TAG}_ckpts_003625_003750_003875_004000_50eps_bs50_20260517_125234"
ROOT_004375="${REPO_ROOT}/outputs/eval/full_bigbrain_gpu3_mixed_b256_p7_s0p5_ckpts_004375_005000_50eps_bs50_20260517_125708/b256_${MODEL_TAG}"

GPU0_3_CKPTS=(${GPU0_3_CKPTS:-004375 004000 003875 003750 003625})
GPU4_7_CKPTS=(${GPU4_7_CKPTS:-003500 003375 003250 003125})
GPU0_3_IDS="${GPU0_3_IDS:-0,1,2,3}"
GPU4_7_IDS="${GPU4_7_IDS:-4,5,6,7}"
WAIT_FOR_EXISTING_EVALS="${WAIT_FOR_EXISTING_EVALS:-1}"

SUITES=(libero_object libero_goal libero_spatial libero_10)
TASK_IDS=(0 1 2 3 4 5 6 7 8 9)

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

output_root_for_ckpt() {
  case "$1" in
    004375) echo "${ROOT_004375}" ;;
    004000|003875|003750|003625) echo "${ROOT_HIGH}" ;;
    003500|003375|003250|003125) echo "${ROOT_LOW}" ;;
    *) echo "Unknown checkpoint root for $1" >&2; return 1 ;;
  esac
}

is_task_complete() {
  local output_dir="$1"
  [[ -f "${output_dir}/eval_info.json" ]] || return 1
  EVAL_INFO="${output_dir}/eval_info.json" N_EPISODES="${N_EPISODES}" python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["EVAL_INFO"])
expected = int(os.environ["N_EPISODES"])
try:
    info = json.loads(path.read_text())
except Exception:
    raise SystemExit(1)
overall = info.get("overall", {})
n_episodes = overall.get("n_episodes")
if n_episodes is None:
    n_episodes = sum(
        len(task.get("metrics", {}).get("episode_metrics", []))
        for task in info.get("per_task", [])
    )
raise SystemExit(0 if int(n_episodes or 0) >= expected else 1)
PY
}

wait_for_existing_evals() {
  if [[ "${WAIT_FOR_EXISTING_EVALS}" != "1" ]]; then
    return
  fi
  echo "===== waiting for currently running lerobot-eval processes before starting resume queue ====="
  while pgrep -u "${USER}" -f '/bin/lerobot-eval| lerobot-eval ' >/dev/null; do
    pgrep -u "${USER}" -af '/bin/lerobot-eval| lerobot-eval ' | head -8
    sleep 60
  done
}

safe_log_fragment() {
  echo "$1" | tr -c '0-9A-Za-z_.-' '_'
}

run_task() {
  local ckpt="$1"
  local base_output_dir="$2"
  local suite="$3"
  local task_id="$4"
  local gpu_id="$5"
  local policy_path="${TRAIN_DIR}/checkpoints/${ckpt}/pretrained_model"
  local output_dir="${base_output_dir}/${suite}_taskids__${task_id}_"
  local label="${MODEL_TAG}_ckpt_${ckpt}_50eps_bs${EVAL_BATCH_SIZE}"
  local safe_label
  safe_label="$(safe_log_fragment "${label}_${suite}_${task_id}_${TIMESTAMP}")"
  local log_path="${LOG_DIR}/resume_bigbrain_b256_${safe_label}.log"

  if is_task_complete "${output_dir}"; then
    echo "[$(date)] skip complete ckpt_${ckpt} ${suite} task ${task_id}: ${output_dir}"
    return 0
  fi

  mkdir -p "${output_dir}"
  echo "[$(date)] start ckpt_${ckpt} ${suite} task ${task_id} on GPU ${gpu_id}"
  echo "[$(date)] log: ${log_path}"
  echo "[$(date)] output: ${output_dir}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  MUJOCO_EGL_DEVICE_ID="${gpu_id}" \
  DATA_ROOT="${DATA_ROOT}" \
  lerobot-eval \
    --policy.path="${policy_path}" \
    --policy.device=cuda \
    --policy.num_steps="${NUM_STEPS}" \
    --policy.chunk_size="${CHUNK_SIZE}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.use_duration_head=false \
    --policy.hiva_coeff_sidecar_path="${HIVA_COEFF_SIDECAR}" \
    --policy.hiva_coeff_sidecar_summary_path="${HIVA_COEFF_SUMMARY}" \
    --policy.hiva_duration_execution_map="${HIVA_DURATION_EXECUTION_MAP}" \
    --env.type=libero \
    --env.task="${suite}" \
    --env.task_ids="[${task_id}]" \
    --env.control_mode=relative \
    --env.max_parallel_tasks="${MAX_PARALLEL_TASKS}" \
    --eval.batch_size="${EVAL_BATCH_SIZE}" \
    --eval.n_episodes="${N_EPISODES}" \
    --eval.max_episodes_rendered="${MAX_EPISODES_RENDERED}" \
    --rename_map="${RENAME_MAP}" \
    --output_dir="${output_dir}" \
    --job_name="resume_bigbrain_b256_${label}_${suite}_${task_id}" \
    > "${log_path}" 2>&1
}

write_summary_if_complete() {
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
if len(eval_infos) < 40:
    print(f"{base}: only {len(eval_infos)}/40 eval_info files; not writing summary yet.")
    raise SystemExit(0)

per_task = []
episodes = []
video_paths = []
group_video_paths = {}
for path in eval_infos:
    info = json.loads(path.read_text())
    group_names = list(info.get("per_group", {}).keys())
    if len(group_names) == 1:
        group_video_paths.setdefault(group_names[0], []).extend(info.get("overall", {}).get("video_paths", []))
    video_paths.extend(info.get("overall", {}).get("video_paths", []))
    for task_info in info.get("per_task", []):
        per_task.append(task_info)
        for episode in task_info.get("metrics", {}).get("episode_metrics", []):
            episode = dict(episode)
            episode["task_group"] = task_info.get("task_group")
            episode["task_id"] = task_info.get("task_id")
            episode["task_prompt"] = task_info.get("task_prompt") or episode.get("display_metrics", {}).get("task_prompt")
            episodes.append(episode)

def mean(values):
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None

def final_duration(seq):
    return seq[-1] if seq else None

def summarize(group_episodes, videos=None):
    videos = videos or []
    successes = [bool(ep.get("success")) for ep in group_episodes]
    displays = [ep.get("display_metrics", {}) or {} for ep in group_episodes]
    return {
        "avg_sum_reward": mean([ep.get("sum_reward") for ep in group_episodes]),
        "avg_max_reward": mean([ep.get("max_reward") for ep in group_episodes]),
        "pc_success": (sum(successes) / len(successes) * 100) if successes else None,
        "n_episodes": len(group_episodes),
        "mean_final_duration": mean([final_duration(d.get("duration")) for d in displays]),
        "mean_inference_calls": mean([d.get("inference_calls") for d in displays]),
        "mean_duration": mean([d.get("mean_duration") for d in displays]),
        "mean_total_time_s": mean([d.get("total_time_s") for d in displays]),
        "task_prompts": sorted({ep.get("task_prompt") for ep in group_episodes if ep.get("task_prompt")}),
        "episode_metrics": group_episodes,
        "video_paths": videos,
    }

episodes_by_group = {}
for episode in episodes:
    episodes_by_group.setdefault(episode.get("task_group"), []).append(episode)
per_group = {
    group: summarize(group_episodes, group_video_paths.get(group, []))
    for group, group_episodes in sorted(episodes_by_group.items())
}
successes = [bool(ep.get("success")) for ep in episodes]
summary = {
    "per_task": per_task,
    "per_group": per_group,
    "overall": {
        "avg_sum_reward": mean([ep.get("sum_reward") for ep in episodes]),
        "avg_max_reward": mean([ep.get("max_reward") for ep in episodes]),
        "pc_success": (sum(successes) / len(successes) * 100) if successes else None,
        "n_episodes": len(episodes),
        "n_video_paths": len(video_paths),
        "video_paths": video_paths,
    },
}

(base / "overlay_eval_summary.json").write_text(json.dumps(summary, indent=2))
print(f"Wrote combined overlay summary: {base / 'overlay_eval_summary.json'}")
print(f"Collected {len(episodes)} episodes and {len(video_paths)} video paths from {len(eval_infos)} eval_info files.")
if len(episodes) != expected_episode_count:
    print(f"Expected {expected_episode_count} episodes, but collected {len(episodes)}.", file=sys.stderr)
    raise SystemExit(2)
if len(video_paths) != expected_video_count:
    print(f"Expected {expected_video_count} videos, but collected {len(video_paths)}.", file=sys.stderr)
    raise SystemExit(2)
PY
}

run_checkpoint() {
  local ckpt="$1"
  local gpu_ids_csv="$2"
  local policy_path="${TRAIN_DIR}/checkpoints/${ckpt}/pretrained_model"
  require_dir "${policy_path}"
  local root
  root="$(output_root_for_ckpt "${ckpt}")"
  local base_output_dir="${root}/ckpt_${ckpt}"
  mkdir -p "${base_output_dir}"

  IFS=',' read -r -a gpus <<< "${gpu_ids_csv}"
  echo "===== $(date) resume ckpt_${ckpt} on GPUs ${gpu_ids_csv} ====="
  echo "POLICY_PATH=${policy_path}"
  echo "BASE_OUTPUT_DIR=${base_output_dir}"

  local pids=()
  local gpu_idx=0
  local status=0
  for suite in "${SUITES[@]}"; do
    for task_id in "${TASK_IDS[@]}"; do
      local output_dir="${base_output_dir}/${suite}_taskids__${task_id}_"
      if is_task_complete "${output_dir}"; then
        echo "[$(date)] already complete ckpt_${ckpt} ${suite} task ${task_id}"
        continue
      fi
      local gpu_id="${gpus[$((gpu_idx % ${#gpus[@]}))]}"
      gpu_idx=$((gpu_idx + 1))
      run_task "${ckpt}" "${base_output_dir}" "${suite}" "${task_id}" "${gpu_id}" &
      pids+=("$!")
      if [[ "${#pids[@]}" -ge "${#gpus[@]}" ]]; then
        for pid in "${pids[@]}"; do
          if ! wait "${pid}"; then
            status=1
          fi
        done
        pids=()
      fi
    done
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "ckpt_${ckpt} had failed shards; see ${LOG_DIR}" >&2
    return "${status}"
  fi
  write_summary_if_complete "${base_output_dir}"
}

run_group() {
  local name="$1"
  local gpu_ids="$2"
  shift 2
  local ckpts=("$@")
  echo "===== ${name} started at $(date), GPUs ${gpu_ids}, ckpts: ${ckpts[*]} ====="
  for ckpt in "${ckpts[@]}"; do
    run_checkpoint "${ckpt}" "${gpu_ids}"
  done
  echo "===== ${name} finished at $(date) ====="
}

main() {
  require_dir "${TRAIN_DIR}"
  require_dir "${DATA_ROOT}"
  require_file "${HIVA_COEFF_SIDECAR}"
  require_file "${HIVA_COEFF_SUMMARY}"
  mkdir -p "${ROOT_LOW}" "${ROOT_HIGH}" "${ROOT_004375}"

  cat > "${LOG_DIR}/resume_bigbrain_b256_s0p5_full_eval_remaining_${TIMESTAMP}.manifest.json" <<JSON
{
  "timestamp": "${TIMESTAMP}",
  "train_dir": "${TRAIN_DIR}",
  "root_low": "${ROOT_LOW}",
  "root_high": "${ROOT_HIGH}",
  "root_004375": "${ROOT_004375}",
  "gpu0_3_ckpts": "$(printf '%s ' "${GPU0_3_CKPTS[@]}")",
  "gpu4_7_ckpts": "$(printf '%s ' "${GPU4_7_CKPTS[@]}")",
  "eval_batch_size": ${EVAL_BATCH_SIZE},
  "n_episodes": ${N_EPISODES}
}
JSON

  echo "===== bigbrain b256 S=0.5 full eval resume started at $(date) ====="
  echo "TRAIN_DIR=${TRAIN_DIR}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "HIVA_COEFF_SIDECAR=${HIVA_COEFF_SIDECAR}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
  echo "GPU0_3_CKPTS=${GPU0_3_CKPTS[*]}"
  echo "GPU4_7_CKPTS=${GPU4_7_CKPTS[*]}"

  wait_for_existing_evals

  run_group gpu0_3_group "${GPU0_3_IDS}" "${GPU0_3_CKPTS[@]}" &
  pid_a=$!
  run_group gpu4_7_group "${GPU4_7_IDS}" "${GPU4_7_CKPTS[@]}" &
  pid_b=$!

  wait "${pid_a}"
  wait "${pid_b}"
  echo "===== bigbrain b256 S=0.5 full eval resume finished at $(date) ====="
}

main "$@"
