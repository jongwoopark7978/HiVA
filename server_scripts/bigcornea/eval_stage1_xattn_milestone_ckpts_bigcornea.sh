#!/usr/bin/env bash
set -euo pipefail

# Evaluate all saved intermediate checkpoints plus last for the requested
# stage-1 LP-MT residual xattn models. Each model gets one parent eval
# directory, with checkpoint-specific evals nested under it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
MODEL_SET="${MODEL_SET:-full}"  # full -> b15,b12; b12_only -> real b12 only; smoke -> smoke b15,b12
CKPTS_OVERRIDE="${CKPTS_OVERRIDE:-}"
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
mkdir -p "${LOG_DIR}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/eval_stage1_xattn_milestone_ckpts_${MODEL_SET}_${TIMESTAMP}.queue.log}"

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

sanitize() {
  local value="$1"
  value="${value//./p}"
  value="${value//[^0-9A-Za-z_+-]/_}"
  echo "${value}"
}

write_model_manifest() {
  local model_eval_dir="$1"
  local train_dir="$2"
  shift 2
  local ckpts=("$@")

  MODEL_EVAL_DIR="${model_eval_dir}" TRAIN_DIR="${train_dir}" CKPTS="${ckpts[*]}" python - <<'PY'
import json
import os
from pathlib import Path

base = Path(os.environ["MODEL_EVAL_DIR"])
train_dir = os.environ["TRAIN_DIR"]
ckpts = os.environ["CKPTS"].split()
manifest = {
    "train_dir": train_dir,
    "checkpoint_order": ckpts,
    "checkpoint_eval_dirs": {
        ckpt: str(base / f"ckpt_{ckpt}") for ckpt in ckpts
    },
}
base.mkdir(parents=True, exist_ok=True)
(base / "checkpoint_eval_manifest.json").write_text(json.dumps(manifest, indent=2))
PY
}

run_checkpoint_eval() {
  local train_dir="$1"
  local model_tag="$2"
  local ckpt="$3"
  local model_eval_dir="$4"
  local checkpoint_label="${model_tag}_ckpt_${ckpt}_10eps_bs${EVAL_BATCH_SIZE}"
  local policy_path="${train_dir}/checkpoints/${ckpt}/pretrained_model"
  local base_output_dir="${model_eval_dir}/ckpt_${ckpt}"

  require_dir "${policy_path}"

  echo "===== $(date) evaluating ${model_tag} ${ckpt} ====="
  echo "POLICY_PATH=${policy_path}"
  echo "BASE_OUTPUT_DIR=${base_output_dir}"

  POLICY_PATH="${policy_path}" \
  CHECKPOINT_LABEL="${checkpoint_label}" \
  TIMESTAMP="${TIMESTAMP}_${model_tag}_${ckpt}" \
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

run_model() {
  local train_dir="$1"
  local model_tag="$2"
  shift 2
  local ckpts=("$@")
  if [[ -n "${CKPTS_OVERRIDE}" ]]; then
    read -r -a ckpts <<< "${CKPTS_OVERRIDE}"
  fi
  local model_eval_dir="${REPO_ROOT}/outputs/eval/stage1_xattn_milestone_ckpt_sweep_${model_tag}_${TIMESTAMP}"

  require_dir "${train_dir}/checkpoints"
  write_model_manifest "${model_eval_dir}" "${train_dir}" "${ckpts[@]}"

  echo "===== $(date) starting model ${model_tag} ====="
  echo "TRAIN_DIR=${train_dir}"
  echo "MODEL_EVAL_DIR=${model_eval_dir}"
  echo "CHECKPOINTS=${ckpts[*]}"

  for ckpt in "${ckpts[@]}"; do
    run_checkpoint_eval "${train_dir}" "${model_tag}" "${ckpt}" "${model_eval_dir}"
  done

  echo "===== $(date) finished model ${model_tag} ====="
}

main() {
  exec > >(tee -a "${QUEUE_LOG}") 2>&1

  require_dir "${DATA_ROOT}"
  require_file "${HIVA_COEFF_SIDECAR}"
  require_file "${HIVA_COEFF_SUMMARY}"

  echo "===== stage1 xattn milestone checkpoint eval queue started at $(date) ====="
  echo "MODEL_SET=${MODEL_SET}"
  echo "TIMESTAMP=${TIMESTAMP}"
  echo "CKPTS_OVERRIDE=${CKPTS_OVERRIDE}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
  echo "N_EPISODES=${N_EPISODES}"
  echo "TASK_IDS_ALL=${TASK_IDS_ALL}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "HIVA_COEFF_SIDECAR=${HIVA_COEFF_SIDECAR}"
  echo "HIVA_COEFF_SUMMARY=${HIVA_COEFF_SUMMARY}"
  echo "QUEUE_LOG=${QUEUE_LOG}"

  case "${MODEL_SET}" in
    full)
      run_model \
        "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_stage1_xattn_b15_v5_d4_6_10_k10_f15_tr3p0_rot3p0_grip0p0_daw1p0_trb0p1_rotb0p05_gripb0p1_b128_s1_20260512_235751" \
        "b15_s1_tr3_rot3_grip0_rotb0p05" \
        000250 000313 000375 000438 last

      run_model \
        "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_stage1_xattn_b12_v5_d4_6_10_k10_f15_tr3p0_rot3p0_grip0p0_daw1p0_trb0p1_rotb0p05_gripb0p1_b128_s1_20260512_235751" \
        "b12_s1_tr3_rot3_grip0_rotb0p05" \
        000250 000313 000375 000438 last
      ;;
    smoke)
      run_model \
        "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_stage1_xattn_b15_v5_d4_6_10_k10_f15_tr3p0_rot3p0_grip0p0_daw1p0_trb0p1_rotb0p05_gripb0p1_b128_s1_smoke_20260512_234739" \
        "b15_s1_smoke_tr3_rot3_grip0_rotb0p05" \
        000001 last

      run_model \
        "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_stage1_xattn_b12_v5_d4_6_10_k10_f15_tr3p0_rot3p0_grip0p0_daw1p0_trb0p1_rotb0p05_gripb0p1_b128_s1_smoke_20260512_234739" \
        "b12_s1_smoke_tr3_rot3_grip0_rotb0p05" \
        000001 last
      ;;
    b12_only)
      run_model \
        "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_stage1_xattn_b12_v5_d4_6_10_k10_f15_tr3p0_rot3p0_grip0p0_daw1p0_trb0p1_rotb0p05_gripb0p1_b128_s1_20260512_235751" \
        "b12_s1_tr3_rot3_grip0_rotb0p05" \
        000250 000313 000375 000438 last
      ;;
    b15_only)
      run_model \
        "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_stage1_xattn_b15_v5_d4_6_10_k10_f15_tr3p0_rot3p0_grip0p0_daw1p0_trb0p1_rotb0p05_gripb0p1_b128_s1_20260512_235751" \
        "b15_s1_tr3_rot3_grip0_rotb0p05" \
        000250 000313 000375 000438 last
      ;;
    *)
      echo "Unknown MODEL_SET=${MODEL_SET}; expected full, b15_only, b12_only, or smoke." >&2
      exit 2
      ;;
  esac

  echo "===== stage1 xattn milestone checkpoint eval queue finished at $(date) ====="
}

main "$@"
