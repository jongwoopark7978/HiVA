#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark}"

DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"

TRAIN_DUALEXPERT="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_residual_flow_stage1_dualexpert_v5_d4_6_10_trrot3_grip0p5_b768_g4_s2_20260514_040613"
TRAIN_LPMT="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_residual_flow_stage1_v5_d4_6_10_k10_f15_tr3p0_rot3p0_grip0p5_daw1p0_b128_g8_s2_20260514_040554"

CKPTS_DUALEXPERT=(000021 000053 000105 000157 last)
CKPTS_LPMT=(000063 000157 000313 000469 last)

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
  MODEL_EVAL_DIR="${MODEL_EVAL_DIR}" TRAIN_DIR="${TRAIN_DIR}" CKPTS="${CKPTS[*]}" python - <<'PY'
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
  local train_dir="$1"
  local model_tag="$2"
  local model_eval_dir="$3"
  local ckpt="$4"
  local policy_path="${train_dir}/checkpoints/${ckpt}/pretrained_model"
  local base_output_dir="${model_eval_dir}/ckpt_${ckpt}"
  local checkpoint_label="${model_tag}_ckpt_${ckpt}_10eps_bs${EVAL_BATCH_SIZE}"

  if [[ -f "${base_output_dir}/overlay_eval_summary.json" ]]; then
    echo "===== $(date) skipping completed ${checkpoint_label}: ${base_output_dir} ====="
    return
  fi

  require_dir "${policy_path}"
  echo "===== $(date) evaluating ${checkpoint_label} ====="
  echo "POLICY_PATH=${policy_path}"
  echo "BASE_OUTPUT_DIR=${base_output_dir}"

  POLICY_PATH="${policy_path}" \
  CHECKPOINT_LABEL="${checkpoint_label}" \
  TIMESTAMP="${TIMESTAMP}_${model_tag}_${ckpt}" \
  GPU_IDS="${GPU_IDS}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  N_EPISODES=10 \
  TASK_IDS_ALL="[0,1,2,3,4,5,6,7,8,9]" \
  MAX_PARALLEL_TASKS=1 \
  MAX_EPISODES_RENDERED=1 \
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
  bash "${SCRIPT_DIR}/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"
}

run_group() {
  local train_dir="$1"
  local model_tag="$2"
  shift 2
  local ckpts=("$@")
  local model_eval_dir="${REPO_ROOT}/outputs/eval/residual_flow_${model_tag}_10eps_bs${EVAL_BATCH_SIZE}_${TIMESTAMP}"

  require_dir "${train_dir}/checkpoints"
  require_dir "${DATA_ROOT}"
  require_file "${HIVA_COEFF_SIDECAR}"
  require_file "${HIVA_COEFF_SUMMARY}"

  local ckpts_joined="${ckpts[*]}"
  TRAIN_DIR="${train_dir}" MODEL_EVAL_DIR="${model_eval_dir}" CKPTS="${ckpts_joined}" write_manifest

  echo "===== Residual-flow group ${model_tag} ====="
  echo "TRAIN_DIR=${train_dir}"
  echo "MODEL_EVAL_DIR=${model_eval_dir}"
  echo "CKPTS=${ckpts[*]}"

  for ckpt in "${ckpts[@]}"; do
    run_ckpt "${train_dir}" "${model_tag}" "${model_eval_dir}" "${ckpt}"
  done
}

echo "===== residual-flow eval queue started at $(date) ====="
echo "TIMESTAMP=${TIMESTAMP}"
echo "GPU_IDS=${GPU_IDS}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"

run_group \
  "${TRAIN_DUALEXPERT}" \
  "dualexpert_v5_d4_6_10_trrot3_grip0p5_b768_g4_s2" \
  "${CKPTS_DUALEXPERT[@]}"

echo "===== Skipping LPMT residual-flow group in this GPU4-7 queue; it is scheduled separately on GPU0-3. ====="

echo "===== residual-flow eval queue finished at $(date) ====="
