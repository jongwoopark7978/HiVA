#!/usr/bin/env bash
set -euo pipefail

# Full 50-episode LIBERO eval for all saved checkpoints from the BigCornea
# stage0 LP-MT K=6 F=15 s0p25 HiVA coefficient run, executed on bigtoken GPUs 4-7.
#
# "Full" means all 4 LIBERO suites, task_ids [0..9], 50 episodes per task:
# 2000 episodes and 40 rendered videos per checkpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
TRAIN_DIR="${TRAIN_DIR:-/nfs/bigcornea.cs.stonybrook.edu/add_disk3/jongwoopark/HiVA_train/finetuning_stage0/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k6_f15_bigcornea_b64_s0p25_20260523_034355}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${TRAIN_DIR}/checkpoints}"
MODEL_TAG="${MODEL_TAG:-$(basename "${TRAIN_DIR}")}"

GPU_IDS="${GPU_IDS:-4,5,6,7}"
GPU_GROUP="${GPU_GROUP:-gpu4_7}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
N_EPISODES="${N_EPISODES:-50}"
TASK_IDS_ALL="${TASK_IDS_ALL:-[0,1,2,3,4,5,6,7,8,9]}"
EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT:-2000}"
EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT:-40}"

N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-15}"
NUM_STEPS="${NUM_STEPS:-10}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-1}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-1}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k6_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k6_f15_canonical_lp_mt.summary.json}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/jongwoopark/hf_datasets_cache}"

HELPER="${HELPER:-${REPO_ROOT}/server_scripts/bigtoken/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh}"
LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
SWEEP_OUTPUT_DIR="${SWEEP_OUTPUT_DIR:-${TRAIN_DIR}/eval/full_bigtoken_${MODEL_TAG}_${GPU_GROUP}_all_ckpts_50eps_bs${EVAL_BATCH_SIZE}_${TIMESTAMP}}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/eval_stage0_lpmt_k6_f15_s0p25_all_ckpts_${GPU_GROUP}_50eps_${TIMESTAMP}.queue.log}"

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

collect_ckpts() {
  if [[ -n "${CKPTS_OVERRIDE:-}" ]]; then
    printf '%s\n' ${CKPTS_OVERRIDE}
    return
  fi

  find "${CHECKPOINT_ROOT}" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' \
    | grep -E '^[0-9]+$' \
    | while read -r ckpt; do
        if [[ -d "${CHECKPOINT_ROOT}/${ckpt}/pretrained_model" ]]; then
          echo "${ckpt}"
        fi
      done \
    | sort -n
}

write_manifest() {
  local ckpts=("$@")
  CKPTS_JOINED="$(printf '%s\n' "${ckpts[@]}")" \
  TIMESTAMP="${TIMESTAMP}" \
  TRAIN_DIR="${TRAIN_DIR}" \
  CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" \
  MODEL_TAG="${MODEL_TAG}" \
  GPU_IDS="${GPU_IDS}" \
  GPU_GROUP="${GPU_GROUP}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  N_EPISODES="${N_EPISODES}" \
  EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT}" \
  EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT}" \
  SWEEP_OUTPUT_DIR="${SWEEP_OUTPUT_DIR}" \
  DATA_ROOT="${DATA_ROOT}" \
  HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR}" \
  HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY}" \
  CHUNK_SIZE="${CHUNK_SIZE}" \
  N_ACTION_STEPS="${N_ACTION_STEPS}" \
  NUM_STEPS="${NUM_STEPS}" \
  python - <<'PY'
import json
import os
from pathlib import Path

ckpts = [line for line in os.environ["CKPTS_JOINED"].splitlines() if line]
base = Path(os.environ["SWEEP_OUTPUT_DIR"])
manifest = {
    "timestamp": os.environ["TIMESTAMP"],
    "train_dir": os.environ["TRAIN_DIR"],
    "checkpoint_root": os.environ["CHECKPOINT_ROOT"],
    "model_tag": os.environ["MODEL_TAG"],
    "checkpoint_order": ckpts,
    "checkpoint_eval_dirs": {ckpt: str(base / f"ckpt_{ckpt}") for ckpt in ckpts},
    "full_eval_definition": {
        "suites": ["libero_object", "libero_goal", "libero_spatial", "libero_10"],
        "task_ids_per_suite": list(range(10)),
        "episodes_per_task": int(os.environ["N_EPISODES"]),
        "expected_episodes_per_checkpoint": int(os.environ["EXPECTED_EPISODE_COUNT"]),
        "expected_videos_per_checkpoint": int(os.environ["EXPECTED_VIDEO_COUNT"]),
    },
    "gpu_ids": os.environ["GPU_IDS"],
    "gpu_group": os.environ["GPU_GROUP"],
    "eval_batch_size": int(os.environ["EVAL_BATCH_SIZE"]),
    "data_root": os.environ["DATA_ROOT"],
    "hiva_coeff_sidecar": os.environ["HIVA_COEFF_SIDECAR"],
    "hiva_coeff_summary": os.environ["HIVA_COEFF_SUMMARY"],
    "chunk_size": int(os.environ["CHUNK_SIZE"]),
    "n_action_steps": int(os.environ["N_ACTION_STEPS"]),
    "num_steps": int(os.environ["NUM_STEPS"]),
}
base.mkdir(parents=True, exist_ok=True)
(base / "checkpoint_eval_manifest.json").write_text(json.dumps(manifest, indent=2))
PY
}

run_checkpoint() {
  local ckpt="$1"
  local policy_path="${CHECKPOINT_ROOT}/${ckpt}/pretrained_model"
  local label="stage0_lpmt_k6_f15_s0p25_ckpt_${ckpt}_50eps_bs${EVAL_BATCH_SIZE}"
  local base_output_dir="${SWEEP_OUTPUT_DIR}/ckpt_${ckpt}"

  if [[ -f "${base_output_dir}/overlay_eval_summary.json" ]]; then
    echo "===== $(date) skipping completed ${label}: ${base_output_dir} ====="
    return
  fi

  require_dir "${policy_path}"
  echo "===== $(date) evaluating ${label} on GPUs ${GPU_IDS} ====="
  echo "POLICY_PATH=${policy_path}"
  echo "BASE_OUTPUT_DIR=${base_output_dir}"

  TIMESTAMP="${TIMESTAMP}_ckpt${ckpt}" \
  POLICY_PATH="${policy_path}" \
  CHECKPOINT_LABEL="${label}" \
  GPU_IDS="${GPU_IDS}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  N_EPISODES="${N_EPISODES}" \
  TASK_IDS_ALL="${TASK_IDS_ALL}" \
  MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS}" \
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
  bash "${HELPER}"
}

main() {
  exec > >(tee -a "${QUEUE_LOG}") 2>&1

  require_file "${HELPER}"
  require_dir "${TRAIN_DIR}"
  require_dir "${CHECKPOINT_ROOT}"
  require_dir "${DATA_ROOT}"
  require_file "${HIVA_COEFF_SIDECAR}"
  require_file "${HIVA_COEFF_SUMMARY}"

  mapfile -t ckpts < <(collect_ckpts)
  if [[ "${#ckpts[@]}" -eq 0 ]]; then
    echo "No checkpoints with pretrained_model found under ${CHECKPOINT_ROOT}" >&2
    exit 1
  fi

  write_manifest "${ckpts[@]}"

  echo "===== stage0 LP-MT K6 F15 s0p25 all-checkpoint full eval started at $(date) ====="
  echo "TIMESTAMP=${TIMESTAMP}"
  echo "TRAIN_DIR=${TRAIN_DIR}"
  echo "CHECKPOINTS=${ckpts[*]}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "SWEEP_OUTPUT_DIR=${SWEEP_OUTPUT_DIR}"
  echo "QUEUE_LOG=${QUEUE_LOG}"
  echo "HIVA_COEFF_SIDECAR=${HIVA_COEFF_SIDECAR}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
  echo "N_EPISODES=${N_EPISODES}"
  echo "CHUNK_SIZE=${CHUNK_SIZE}"
  echo "N_ACTION_STEPS=${N_ACTION_STEPS}"

  local ckpt
  for ckpt in "${ckpts[@]}"; do
    run_checkpoint "${ckpt}"
  done

  echo "===== stage0 LP-MT K6 F15 s0p25 all-checkpoint full eval finished at $(date) ====="
}

main "$@"
