#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-20260513_180325_earlyckpt_gpu0_3}"
TRAIN_DIR="${TRAIN_DIR:-/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_stage1_xattn_b15_v5_d4_6_10_k10_f15_tr3p0_rot3p0_grip0p0_daw1p0_trb0p1_rotb0p05_gripb0p1_b1024_g4_s1_earlyckpt_20260513_180325}"
MODEL_TAG="${MODEL_TAG:-b15_s1_earlyckpt_tr3_rot3_grip0_rotb0p05}"
CKPTS="${CKPTS:-000032 000044 000050 000057 000063 000069 000075}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
N_EPISODES="${N_EPISODES:-10}"
TASK_IDS_ALL="${TASK_IDS_ALL:-[0,1,2,3,4,5,6,7,8,9]}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-1}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-1}"

DATA_ROOT="${DATA_ROOT:-/nfs/bigcornea/add_disk2/jongwoopark/libero_lerobot_v3_lerobotkeys}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea/add_disk2/jongwoopark}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"

LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
MODEL_EVAL_DIR="${REPO_ROOT}/outputs/eval/stage1_xattn_earlyckpt_sweep_${MODEL_TAG}_${TIMESTAMP}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/eval_stage1_xattn_earlyckpt_${MODEL_TAG}_${TIMESTAMP}.queue.log}"
mkdir -p "${LOG_DIR}" "${MODEL_EVAL_DIR}"

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

write_manifest() {
  MODEL_EVAL_DIR="${MODEL_EVAL_DIR}" TRAIN_DIR="${TRAIN_DIR}" CKPTS="${CKPTS}" python - <<'PY'
import json
import os
from pathlib import Path

base = Path(os.environ["MODEL_EVAL_DIR"])
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

run_ckpt() {
  local ckpt="$1"
  local policy_path="${TRAIN_DIR}/checkpoints/${ckpt}/pretrained_model"
  local checkpoint_label="${MODEL_TAG}_ckpt_${ckpt}_10eps_bs${EVAL_BATCH_SIZE}"
  local base_output_dir="${MODEL_EVAL_DIR}/ckpt_${ckpt}"

  if [[ -f "${base_output_dir}/overlay_eval_summary.json" ]]; then
    echo "===== $(date) skipping completed ${ckpt}: ${base_output_dir} ====="
    return
  fi

  require_dir "${policy_path}"
  echo "===== $(date) evaluating ${ckpt} ====="
  echo "POLICY_PATH=${policy_path}"
  echo "BASE_OUTPUT_DIR=${base_output_dir}"

  POLICY_PATH="${policy_path}" \
  CHECKPOINT_LABEL="${checkpoint_label}" \
  TIMESTAMP="${TIMESTAMP}_${MODEL_TAG}_${ckpt}" \
  GPU_IDS="${GPU_IDS}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  N_EPISODES="${N_EPISODES}" \
  TASK_IDS_ALL="${TASK_IDS_ALL}" \
  MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS}" \
  MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED}" \
  EXPECTED_EPISODE_COUNT=400 \
  EXPECTED_VIDEO_COUNT=40 \
  DATA_ROOT="${DATA_ROOT}" \
  HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR}" \
  HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY}" \
  CHUNK_SIZE=15 \
  N_ACTION_STEPS=10 \
  NUM_STEPS=10 \
  HIVA_DURATION_EXECUTION_MAP="" \
  BASE_OUTPUT_DIR="${base_output_dir}" \
  bash "${REPO_ROOT}/server_scripts/bigtoken/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"
}

main() {
  exec > >(tee -a "${QUEUE_LOG}") 2>&1

  require_dir "${TRAIN_DIR}/checkpoints"
  require_dir "${DATA_ROOT}"
  require_file "${HIVA_COEFF_SIDECAR}"
  require_file "${HIVA_COEFF_SUMMARY}"
  write_manifest

  echo "===== early checkpoint eval queue started at $(date) ====="
  echo "TRAIN_DIR=${TRAIN_DIR}"
  echo "MODEL_EVAL_DIR=${MODEL_EVAL_DIR}"
  echo "CKPTS=${CKPTS}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
  echo "N_EPISODES=${N_EPISODES}"
  echo "QUEUE_LOG=${QUEUE_LOG}"

  for ckpt in ${CKPTS}; do
    run_ckpt "${ckpt}"
  done

  echo "===== early checkpoint eval queue finished at $(date) ====="
}

main "$@"
