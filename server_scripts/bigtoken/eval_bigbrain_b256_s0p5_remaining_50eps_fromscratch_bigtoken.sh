#!/usr/bin/env bash
set -euo pipefail

# Fresh full 50-episode LIBERO eval for the remaining b256 S=0.5 stage0
# LP-MT HiVA checkpoints. This intentionally does not reuse the partially
# completed bigbrain task-sharded output directories.
#
# Uses the same staged LIBERO-10 assignment style as the current bigtoken full
# eval helper:
#   GPU0/4: libero_object [0..9]
#   GPU1/5: libero_goal [0..9]
#   GPU2/6: libero_spatial [0..9]
#   GPU3/7: libero_10 [0..6]
# After the short suites finish, GPUs 0/1/2 or 4/5/6 run libero_10 [7], [8],
# [9] while the long libero_10 [0..6] shard continues.

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
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/jongwoopark/hf_datasets_cache}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
N_EPISODES="${N_EPISODES:-50}"
TASK_IDS_ALL="${TASK_IDS_ALL:-[0,1,2,3,4,5,6,7,8,9]}"
EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT:-2000}"
EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT:-40}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-15}"
NUM_STEPS="${NUM_STEPS:-10}"
WAIT_FOR_EXISTING_EVALS="${WAIT_FOR_EXISTING_EVALS:-1}"

GPU0_3_CKPTS=(${GPU0_3_CKPTS:-004375 004000 003875 003750 003625})
GPU4_7_CKPTS=(${GPU4_7_CKPTS:-003500 003375 003250 003125})

SWEEP_OUTPUT_DIR="${SWEEP_OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/full_bigtoken_${MODEL_TAG}_remaining_fromscratch_50eps_bs${EVAL_BATCH_SIZE}_${TIMESTAMP}}"
LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
HELPER="${REPO_ROOT}/server_scripts/bigtoken/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"
mkdir -p "${SWEEP_OUTPUT_DIR}" "${LOG_DIR}" "${HF_DATASETS_CACHE}"

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

gpu_group_has_python_processes() {
  local gpu_ids="$1"
  nvidia-smi | awk -v ids="${gpu_ids}" '
    BEGIN {
      split(ids, arr, ",")
      for (i in arr) want[arr[i]] = 1
    }
    $1 == "|" && ($2 in want) && /python/ { found = 1 }
    END { exit found ? 0 : 1 }
  '
}

wait_for_gpu_group() {
  local name="$1"
  local gpu_ids="$2"
  if [[ "${WAIT_FOR_EXISTING_EVALS}" != "1" ]]; then
    return
  fi
  echo "===== ${name} waiting for GPUs ${gpu_ids} to be free before starting fresh queue ====="
  while gpu_group_has_python_processes "${gpu_ids}"; do
    echo "[$(date)] ${name} still waiting; active python processes on GPUs ${gpu_ids}:"
    nvidia-smi | awk -v ids="${gpu_ids}" '
      BEGIN {
        split(ids, arr, ",")
        for (i in arr) want[arr[i]] = 1
      }
      $1 == "|" && ($2 in want) && /python/ { print }
    '
    sleep 60
  done
  echo "===== ${name} GPUs ${gpu_ids} are free at $(date); starting fresh queue ====="
}

run_group_when_free() {
  local name="$1"
  local gpu_ids="$2"
  shift 2
  wait_for_gpu_group "${name}" "${gpu_ids}"
  run_group "${name}" "${gpu_ids}" "$@"
}

write_manifest() {
  cat > "${SWEEP_OUTPUT_DIR}/checkpoint_eval_manifest.json" <<JSON
{
  "train_dir": "${TRAIN_DIR}",
  "checkpoint_order_gpu0_3": [$(printf '"%s",' "${GPU0_3_CKPTS[@]}" | sed 's/,$//')],
  "checkpoint_order_gpu4_7": [$(printf '"%s",' "${GPU4_7_CKPTS[@]}" | sed 's/,$//')],
  "sweep_output_dir": "${SWEEP_OUTPUT_DIR}",
  "eval_batch_size": ${EVAL_BATCH_SIZE},
  "n_episodes": ${N_EPISODES},
  "expected_episode_count": ${EXPECTED_EPISODE_COUNT},
  "expected_video_count": ${EXPECTED_VIDEO_COUNT},
  "staged_libero10_after_short": true,
  "fresh_from_scratch": true
}
JSON
}

run_checkpoint() {
  local ckpt="$1"
  local gpu_ids="$2"
  local policy_path="${TRAIN_DIR}/checkpoints/${ckpt}/pretrained_model"
  local label="bigbrain_b256_s0p5_${MODEL_TAG}_ckpt_${ckpt}_50eps_bs${EVAL_BATCH_SIZE}"
  local base_output_dir="${SWEEP_OUTPUT_DIR}/ckpt_${ckpt}"

  require_dir "${policy_path}"

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
  local name="$1"
  local gpu_ids="$2"
  shift 2
  local ckpts=("$@")

  echo "===== ${name} started at $(date) on GPUs ${gpu_ids}: ${ckpts[*]} ====="
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
  require_file "${HELPER}"

  write_manifest

  echo "===== bigbrain b256 S=0.5 fresh full eval queue started at $(date) ====="
  echo "TRAIN_DIR=${TRAIN_DIR}"
  echo "SWEEP_OUTPUT_DIR=${SWEEP_OUTPUT_DIR}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "HIVA_COEFF_SIDECAR=${HIVA_COEFF_SIDECAR}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
  echo "GPU0_3_CKPTS=${GPU0_3_CKPTS[*]}"
  echo "GPU4_7_CKPTS=${GPU4_7_CKPTS[*]}"

  run_group_when_free gpu0_3_group "0,1,2,3" "${GPU0_3_CKPTS[@]}" &
  pid_a=$!
  run_group_when_free gpu4_7_group "4,5,6,7" "${GPU4_7_CKPTS[@]}" &
  pid_b=$!

  wait "${pid_a}"
  wait "${pid_b}"

  echo "===== bigbrain b256 S=0.5 fresh full eval queue finished at $(date) ====="
}

main "$@"
