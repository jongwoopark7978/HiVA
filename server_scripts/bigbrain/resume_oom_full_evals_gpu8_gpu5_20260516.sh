#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
mkdir -p "${LOG_DIR}"

RUNNER="${SCRIPT_DIR}/eval_lpmt_stage0_v5_all_ckpts_partial_gpu6_7_bs10_bigbrain.sh"
RESUME_LOG="${LOG_DIR}/resume_oom_full_evals_gpu8_gpu5_${TIMESTAMP}.queue.log"

wait_for_gpu_free() {
  local gpu_id="$1"
  local min_free_mb="$2"
  local label="$3"
  local free_mb

  while true; do
    free_mb="$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
    echo "[$(date)] ${label}: GPU ${gpu_id} free=${free_mb} MiB, need >= ${min_free_mb} MiB"
    if [[ "${free_mb}" =~ ^[0-9]+$ ]] && (( free_mb >= min_free_mb )); then
      break
    fi
    sleep 120
  done
}

archive_partial_dir() {
  local path="$1"
  if [[ -d "${path}" && ! -f "${path}/eval_info.json" ]]; then
    local archived="${path}.failed_oom_${TIMESTAMP}"
    if [[ ! -e "${archived}" ]]; then
      echo "[$(date)] Archiving incomplete eval dir: ${path} -> ${archived}"
      mv "${path}" "${archived}"
    fi
  fi
}

resume_original_smolvla_ckpt003500() {
  local train_dir="/home/jongwoopark/lerobot/outputs/train/smolvla_original_bigcornea_s0p5_b64_g8_20260515_223843_888072651_pid2567557"
  local eval_root="/home/jongwoopark/lerobot/outputs/eval/full_bigbrain_gpu8_smolvla_original_bigcornea_s0p5_b64_g8_20260515_223843_888072651_pid2567557_ckpts_003000_003125_003500_003750_50eps_bs50_20260516_172454"
  local queue_log="${LOG_DIR}/resume_original_smolvla_ckpt003500_gpu8_${TIMESTAMP}.queue.log"

  wait_for_gpu_free 8 45000 "original_smolvla_ckpt003500"
  archive_partial_dir "${eval_root}/ckpt_003500/libero_object_taskids__0_"

  TIMESTAMP="${TIMESTAMP}" \
  TRAIN_DIR="${train_dir}" \
  MODEL_TAG="$(basename "${train_dir}")" \
  CKPTS_OVERRIDE="003500" \
  GPU_IDS="8" \
  EVAL_CHECKPOINTS_IN_PARALLEL=1 \
  N_EPISODES=50 \
  EVAL_BATCH_SIZE=50 \
  INCLUDE_HIVA_COEFF_ARGS=0 \
  N_ACTION_STEPS=1 \
  CHUNK_SIZE=50 \
  NUM_STEPS=10 \
  SWEEP_OUTPUT_DIR="${eval_root}" \
  QUEUE_LOG="${queue_log}" \
  bash "${RUNNER}"
}

resume_hiva_b256_ckpt002750() {
  local train_dir="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigbrain_b256_g2_s0p5_steps5000_20260515_233203"
  local eval_root="/home/jongwoopark/lerobot/outputs/eval/full_bigbrain_gpu5_smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigbrain_b256_g2_s0p5_steps5000_20260515_233203_ckpts_002500_002750_50eps_bs50_20260516_172454"
  local queue_log="${LOG_DIR}/resume_hiva_b256_ckpt002750_gpu5_${TIMESTAMP}.queue.log"

  wait_for_gpu_free 5 35000 "hiva_b256_ckpt002750"
  archive_partial_dir "${eval_root}/ckpt_002750/libero_object_taskids__2_"

  TIMESTAMP="${TIMESTAMP}" \
  TRAIN_DIR="${train_dir}" \
  MODEL_TAG="$(basename "${train_dir}")" \
  CKPTS_OVERRIDE="002750" \
  GPU_IDS="5" \
  EVAL_CHECKPOINTS_IN_PARALLEL=1 \
  N_EPISODES=50 \
  EVAL_BATCH_SIZE=50 \
  INCLUDE_HIVA_COEFF_ARGS=1 \
  HIVA_COEFF_SIDECAR="/nfs/bigbrain/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet" \
  HIVA_COEFF_SUMMARY="/nfs/bigbrain/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json" \
  SWEEP_OUTPUT_DIR="${eval_root}" \
  QUEUE_LOG="${queue_log}" \
  bash "${RUNNER}"
}

main() {
  exec > >(tee -a "${RESUME_LOG}") 2>&1

  echo "===== guarded OOM resume started at $(date) ====="
  echo "TIMESTAMP=${TIMESTAMP}"
  echo "RUNNER=${RUNNER}"
  echo "RESUME_LOG=${RESUME_LOG}"

  resume_original_smolvla_ckpt003500 &
  local original_pid="$!"
  resume_hiva_b256_ckpt002750 &
  local hiva_pid="$!"

  local status=0
  wait "${original_pid}" || status=1
  wait "${hiva_pid}" || status=1

  if [[ "${status}" -ne 0 ]]; then
    echo "===== guarded OOM resume failed at $(date) =====" >&2
    exit "${status}"
  fi
  echo "===== guarded OOM resume finished at $(date) ====="
}

main "$@"
