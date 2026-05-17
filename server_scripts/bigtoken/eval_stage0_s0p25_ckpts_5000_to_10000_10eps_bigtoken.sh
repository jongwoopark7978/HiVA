#!/usr/bin/env bash
set -euo pipefail

# Partial LIBERO eval for selected stage0 LP-MT HiVA S=0.25 checkpoints.
#
# Group A runs sequentially on GPUs 0-3:
#   005000, 006000, 006250
# Group B runs sequentially on GPUs 4-7:
#   007000, 007500, 008750, 010000
#
# Each checkpoint evaluates all 4 LIBERO suites, task_ids [0..9],
# 10 episodes/task, eval.batch_size=4.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"

TRAIN_DIR="${TRAIN_DIR:-/nfs/bigcornea.cs.stonybrook.edu/add_disk3/jongwoopark/HiVA_train/finetuning_stage0/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigcornea_b64_s0p25_20260515_171520}"
MODEL_TAG="${MODEL_TAG:-$(basename "${TRAIN_DIR}")}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
N_EPISODES="${N_EPISODES:-10}"
TASK_IDS_ALL="${TASK_IDS_ALL:-[0,1,2,3,4,5,6,7,8,9]}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-1}"
EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT:-400}"
EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT:-40}"

N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-15}"
NUM_STEPS="${NUM_STEPS:-10}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/jongwoopark/hf_datasets_cache}"
mkdir -p "${HF_DATASETS_CACHE}"

SWEEP_OUTPUT_DIR="${SWEEP_OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/stage0_s0p25_${MODEL_TAG}_10eps_bs${EVAL_BATCH_SIZE}_${TIMESTAMP}}"

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

run_ckpt() {
  local ckpt="$1"
  local gpu_ids="$2"
  local policy_path="${TRAIN_DIR}/checkpoints/${ckpt}/pretrained_model"
  local base_output_dir="${SWEEP_OUTPUT_DIR}/ckpt_${ckpt}"
  local label="stage0_s0p25_${MODEL_TAG}_ckpt_${ckpt}_10eps_bs${EVAL_BATCH_SIZE}"

  if [[ -f "${base_output_dir}/overlay_eval_summary.json" ]]; then
    echo "===== $(date) skipping completed ${label}: ${base_output_dir} ====="
    return
  fi
  require_dir "${policy_path}"

  echo "===== $(date) evaluating ${label} on GPUs ${gpu_ids} ====="
  echo "POLICY_PATH=${policy_path}"
  echo "BASE_OUTPUT_DIR=${base_output_dir}"

  POLICY_PATH="${policy_path}" \
  CHECKPOINT_LABEL="${label}" \
  TIMESTAMP="${TIMESTAMP}_${ckpt}" \
  GPU_IDS="${gpu_ids}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  N_EPISODES="${N_EPISODES}" \
  TASK_IDS_ALL="${TASK_IDS_ALL}" \
  MAX_PARALLEL_TASKS=1 \
  MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED}" \
  EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT}" \
  EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT}" \
  DATA_ROOT="${DATA_ROOT}" \
  HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR}" \
  HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY}" \
  CHUNK_SIZE="${CHUNK_SIZE}" \
  N_ACTION_STEPS="${N_ACTION_STEPS}" \
  NUM_STEPS="${NUM_STEPS}" \
  HIVA_DURATION_EXECUTION_MAP="" \
  STAGED_LIBERO10_AFTER_SHORT=1 \
  SPLIT_LIBERO10_ACROSS_GPUS=0 \
  HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
  BASE_OUTPUT_DIR="${base_output_dir}" \
  bash "${SCRIPT_DIR}/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"
}

run_group() {
  local name="$1"
  local gpu_ids="$2"
  shift 2
  local ckpts=("$@")

  echo "===== ${name} started at $(date) on GPUs ${gpu_ids}: ${ckpts[*]} ====="
  for ckpt in "${ckpts[@]}"; do
    run_ckpt "${ckpt}" "${gpu_ids}"
  done
  echo "===== ${name} finished at $(date) ====="
}

main() {
  require_dir "${TRAIN_DIR}"
  require_dir "${DATA_ROOT}"
  require_file "${HIVA_COEFF_SIDECAR}"
  require_file "${HIVA_COEFF_SUMMARY}"
  mkdir -p "${SWEEP_OUTPUT_DIR}"

  cat > "${SWEEP_OUTPUT_DIR}/checkpoint_eval_manifest.json" <<JSON
{
  "train_dir": "${TRAIN_DIR}",
  "checkpoint_order_gpu0_3": ["005000", "006000", "006250"],
  "checkpoint_order_gpu4_7": ["007000", "007500", "008750", "010000"],
  "sweep_output_dir": "${SWEEP_OUTPUT_DIR}",
  "eval_batch_size": ${EVAL_BATCH_SIZE},
  "n_episodes": ${N_EPISODES}
}
JSON

  echo "===== stage0 S=0.25 selected checkpoint partial eval started at $(date) ====="
  echo "TRAIN_DIR=${TRAIN_DIR}"
  echo "SWEEP_OUTPUT_DIR=${SWEEP_OUTPUT_DIR}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "HIVA_COEFF_SIDECAR=${HIVA_COEFF_SIDECAR}"
  echo "HIVA_COEFF_SUMMARY=${HIVA_COEFF_SUMMARY}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
  echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"

  run_group "gpu0_3_group" "0,1,2,3" "005000" "006000" "006250" &
  pid_a="$!"
  run_group "gpu4_7_group" "4,5,6,7" "007000" "007500" "008750" "010000" &
  pid_b="$!"

  status=0
  for pid in "${pid_a}" "${pid_b}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done

  if [[ "${status}" -ne 0 ]]; then
    echo "One or more stage0 S=0.25 checkpoint eval groups failed." >&2
    exit "${status}"
  fi

  echo "===== stage0 S=0.25 selected checkpoint partial eval finished at $(date) ====="
}

main "$@"
