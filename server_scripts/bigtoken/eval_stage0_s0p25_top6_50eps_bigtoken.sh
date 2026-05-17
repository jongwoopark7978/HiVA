#!/usr/bin/env bash
set -euo pipefail

# Full 50-episode LIBERO eval for selected S=0.25 stage0 LP-MT HiVA checkpoints.
# Runs two independent sequential queues:
#   GPUs 0-3: 007500 -> 007000 -> 006250
#   GPUs 4-7: 010000 -> 008750 -> 006000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
TRAIN_DIR="${TRAIN_DIR:-/nfs/bigcornea.cs.stonybrook.edu/add_disk3/jongwoopark/HiVA_train/finetuning_stage0/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigcornea_b64_s0p25_20260515_171520}"
MODEL_TAG="$(basename "${TRAIN_DIR}")"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark}"

DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/jongwoopark/hf_datasets_cache}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
N_EPISODES="${N_EPISODES:-50}"
TASK_IDS_ALL="${TASK_IDS_ALL:-[0,1,2,3,4,5,6,7,8,9]}"
EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT:-2000}"
EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT:-40}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-15}"
NUM_STEPS="${NUM_STEPS:-10}"

SWEEP_OUTPUT_DIR="${SWEEP_OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/full_bigtoken_${MODEL_TAG}_top6_50eps_bs${EVAL_BATCH_SIZE}_${TIMESTAMP}}"
LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${SWEEP_OUTPUT_DIR}" "${LOG_DIR}" "${HF_DATASETS_CACHE}"

HELPER="${REPO_ROOT}/server_scripts/bigtoken/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"

cat > "${SWEEP_OUTPUT_DIR}/checkpoint_eval_manifest.json" <<JSON
{
  "train_dir": "${TRAIN_DIR}",
  "checkpoint_order_gpu0_3": ["007500", "007000", "006250"],
  "checkpoint_order_gpu4_7": ["010000", "008750", "006000"],
  "sweep_output_dir": "${SWEEP_OUTPUT_DIR}",
  "eval_batch_size": ${EVAL_BATCH_SIZE},
  "n_episodes": ${N_EPISODES},
  "expected_episode_count": ${EXPECTED_EPISODE_COUNT}
}
JSON

run_checkpoint() {
  local ckpt="$1"
  local gpu_ids="$2"
  local policy_path="${TRAIN_DIR}/checkpoints/${ckpt}/pretrained_model"
  local label="stage0_s0p25_${MODEL_TAG}_ckpt_${ckpt}_50eps_bs${EVAL_BATCH_SIZE}"
  local base_output_dir="${SWEEP_OUTPUT_DIR}/ckpt_${ckpt}"

  if [[ ! -d "${policy_path}" ]]; then
    echo "Missing checkpoint ${policy_path}" >&2
    return 1
  fi

  echo "===== $(date) evaluating ${label} on GPUs ${gpu_ids} ====="
  echo "POLICY_PATH=${policy_path}"
  echo "BASE_OUTPUT_DIR=${base_output_dir}"

  TIMESTAMP="${TIMESTAMP}_${ckpt}" \
  POLICY_PATH="${policy_path}" \
  CHECKPOINT_LABEL="${label}" \
  GPU_IDS="${gpu_ids}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  N_EPISODES="${N_EPISODES}" \
  TASK_IDS_ALL="${TASK_IDS_ALL}" \
  MAX_PARALLEL_TASKS=1 \
  MAX_EPISODES_RENDERED=1 \
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
  bash "${HELPER}"
}

run_group() {
  local group_name="$1"
  local gpu_ids="$2"
  shift 2
  echo "===== ${group_name} started at $(date) on GPUs ${gpu_ids}: $* ====="
  for ckpt in "$@"; do
    run_checkpoint "${ckpt}" "${gpu_ids}"
  done
  echo "===== ${group_name} finished at $(date) ====="
}

echo "===== stage0 S=0.25 top6 full eval started at $(date) ====="
echo "TRAIN_DIR=${TRAIN_DIR}"
echo "SWEEP_OUTPUT_DIR=${SWEEP_OUTPUT_DIR}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "HIVA_COEFF_SIDECAR=${HIVA_COEFF_SIDECAR}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "N_EPISODES=${N_EPISODES}"

run_group gpu0_3_group "0,1,2,3" "007500" "007000" "006250" &
pid_a=$!
run_group gpu4_7_group "4,5,6,7" "010000" "008750" "006000" &
pid_b=$!

wait "${pid_a}"
wait "${pid_b}"

echo "===== stage0 S=0.25 top6 full eval finished at $(date) ====="
