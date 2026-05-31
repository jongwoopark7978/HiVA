#!/usr/bin/env bash
set -euo pipefail

# Wait for GPUs 0-3 to be free, then full-evaluate all saved checkpoints from
# two currently-running stage0 train directories. Checkpoints are discovered
# after the wait so newly saved checkpoints can be picked up before eval starts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
N_EPISODES="${N_EPISODES:-50}"
REPEATS="${REPEATS:-1}"
WAIT_FOR_GPU_GROUP="${WAIT_FOR_GPU_GROUP:-1}"

TRAIN_DIRS=(
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_p3trrot_p7grip_bigbrain_b256_g2_s0p25_steps10000_20260519_212637_resume_gpu8_9"
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigcornea_b64_s0p125_20260519_235507"
)

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"

HELPER_QUEUE="${REPO_ROOT}/server_scripts/bigtoken/eval_best_s0p25_extra_ckpts_6300_7500_repeat2_50eps_bigtoken.sh"
SWEEP_ROOT="${SWEEP_ROOT:-${REPO_ROOT}/outputs/eval/full_bigtoken_stage0_saved_ckpts_two_running_models_gpu0_3_50eps_bs${EVAL_BATCH_SIZE}_${TIMESTAMP}}"
LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${SWEEP_ROOT}" "${LOG_DIR}"

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
  if [[ "${WAIT_FOR_GPU_GROUP}" != "1" ]]; then
    return
  fi
  echo "===== waiting for GPUs ${GPU_IDS} to be free at $(date) ====="
  while gpu_group_has_python_processes "${GPU_IDS}"; do
    echo "[$(date)] still waiting; active python processes on GPUs ${GPU_IDS}:"
    nvidia-smi | awk -v ids="${GPU_IDS}" '
      BEGIN {
        split(ids, arr, ",")
        for (i in arr) want[arr[i]] = 1
      }
      $1 == "|" && ($2 in want) && /python/ { print }
    '
    sleep 60
  done
  echo "===== GPUs ${GPU_IDS} are free at $(date) ====="
}

list_saved_ckpts() {
  local train_dir="$1"
  TRAIN_DIR_FOR_LIST="${train_dir}" python - <<'PY'
import os
from pathlib import Path

train_dir = Path(os.environ["TRAIN_DIR_FOR_LIST"])
ckpt_root = train_dir / "checkpoints"
items = []
for path in ckpt_root.iterdir():
    if path.name == "last" or path.is_symlink() or not path.is_dir():
        continue
    if (path / "pretrained_model").is_dir():
        items.append(path.name)
print(" ".join(sorted(items)))
PY
}

write_manifest() {
  python - <<'PY' > "${SWEEP_ROOT}/queue_manifest.json"
import json
import os

manifest = {
    "timestamp": os.environ["TIMESTAMP"],
    "gpu_ids": os.environ["GPU_IDS"],
    "eval_batch_size": int(os.environ["EVAL_BATCH_SIZE"]),
    "n_episodes": int(os.environ["N_EPISODES"]),
    "repeats": int(os.environ["REPEATS"]),
    "sweep_root": os.environ["SWEEP_ROOT"],
    "hiva_coeff_sidecar": os.environ["HIVA_COEFF_SIDECAR"],
    "hiva_coeff_summary": os.environ["HIVA_COEFF_SUMMARY"],
    "train_dirs": os.environ["TRAIN_DIRS_JOINED"].split("\n"),
}
print(json.dumps(manifest, indent=2))
PY
}

run_train_dir() {
  local train_dir="$1"
  local model_tag
  model_tag="$(basename "${train_dir}")"
  local ckpts
  ckpts="$(list_saved_ckpts "${train_dir}")"
  if [[ -z "${ckpts}" ]]; then
    echo "No saved checkpoints found under ${train_dir}/checkpoints; skipping."
    return
  fi

  local output_dir="${SWEEP_ROOT}/${model_tag}"
  echo "===== evaluating ${model_tag} at $(date) ====="
  echo "TRAIN_DIR=${train_dir}"
  echo "CKPTS=${ckpts}"
  echo "OUTPUT_DIR=${output_dir}"

  TRAIN_DIR="${train_dir}" \
  CHECKPOINT_ROOT="${train_dir}/checkpoints" \
  CKPTS_OVERRIDE="${ckpts}" \
  GROUP=gpu0_3 \
  GPU_IDS="${GPU_IDS}" \
  REPEATS="${REPEATS}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  N_EPISODES="${N_EPISODES}" \
  HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR}" \
  HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY}" \
  SWEEP_OUTPUT_DIR="${output_dir}" \
  TIMESTAMP="${TIMESTAMP}_${model_tag}" \
  bash "${HELPER_QUEUE}"
}

main() {
  require_file "${HELPER_QUEUE}"
  require_file "${HIVA_COEFF_SIDECAR}"
  require_file "${HIVA_COEFF_SUMMARY}"
  for train_dir in "${TRAIN_DIRS[@]}"; do
    require_dir "${train_dir}/checkpoints"
  done

  TRAIN_DIRS_JOINED="$(printf '%s\n' "${TRAIN_DIRS[@]}")" \
  TIMESTAMP="${TIMESTAMP}" \
  GPU_IDS="${GPU_IDS}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  N_EPISODES="${N_EPISODES}" \
  REPEATS="${REPEATS}" \
  SWEEP_ROOT="${SWEEP_ROOT}" \
  HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR}" \
  HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY}" \
  write_manifest

  echo "===== two-model saved-checkpoint full eval queue started at $(date) ====="
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
  echo "N_EPISODES=${N_EPISODES}"

  wait_for_gpu_group
  for train_dir in "${TRAIN_DIRS[@]}"; do
    run_train_dir "${train_dir}"
  done
  echo "===== two-model saved-checkpoint full eval queue finished at $(date) ====="
}

main "$@"
