#!/usr/bin/env bash
set -euo pipefail

# Sequential partial LIBERO evals for residual-FFN K=10 HiVA coefficient
# checkpoints. Each checkpoint runs 10 episodes x 10 task ids x 4 suites with
# four suites in parallel over GPU_IDS and eval.batch_size=4.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HELPER="${SCRIPT_DIR}/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
N_EPISODES="${N_EPISODES:-10}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-1}"
EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT:-400}"
EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT:-40}"

BIGCORNEA_ROOT="${BIGCORNEA_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${BIGCORNEA_ROOT}/libero_lerobot_v3_lerobotkeys}"
MT_K10_SIDECAR="${MT_K10_SIDECAR:-${BIGCORNEA_ROOT}/libero_hiva_coeff_sidecar_d6_10_15_k10_canonical_mt.parquet}"
MT_K10_SUMMARY="${MT_K10_SUMMARY:-${BIGCORNEA_ROOT}/libero_hiva_coeff_sidecar_d6_10_15_k10_canonical_mt.summary.json}"
HP_K10_SIDECAR="${HP_K10_SIDECAR:-${BIGCORNEA_ROOT}/libero_hiva_coeff_sidecar_d6_10_15_k10_canonical_hp.parquet}"
HP_K10_SUMMARY="${HP_K10_SUMMARY:-${BIGCORNEA_ROOT}/libero_hiva_coeff_sidecar_d6_10_15_k10_canonical_hp.summary.json}"

if [[ ! -x "${HELPER}" ]]; then
  echo "Missing executable helper: ${HELPER}" >&2
  exit 1
fi

run_checkpoint() {
  local label="$1"
  local policy_path="$2"
  local sidecar="$3"
  local summary="$4"

  echo
  echo "======================================================================"
  echo "Starting ${label} at $(date)"
  echo "POLICY_PATH=${policy_path}"
  echo "HIVA_COEFF_SIDECAR=${sidecar}"
  echo "HIVA_COEFF_SUMMARY=${summary}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
  echo "======================================================================"

  TIMESTAMP="${TIMESTAMP}" \
  GPU_IDS="${GPU_IDS}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  N_EPISODES="${N_EPISODES}" \
  MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED}" \
  EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT}" \
  EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT}" \
  DATA_ROOT="${DATA_ROOT}" \
  POLICY_PATH="${policy_path}" \
  CHECKPOINT_LABEL="${label}" \
  HIVA_COEFF_SIDECAR="${sidecar}" \
  HIVA_COEFF_SUMMARY="${summary}" \
  bash "${HELPER}"

  echo "Finished ${label} at $(date)"
}

run_checkpoint \
  "smolvla_hiva_coeff_mt_k10_bigcornea_job1_residual_ffn_ce_mean_full_w1p0_sigma0p25_b64_s4_20260506_030413_10eps_bs4" \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_mt_k10_bigcornea_job1_residual_ffn_ce_mean_full_w1p0_sigma0p25_b64_s4_20260506_030413/checkpoints/last/pretrained_model" \
  "${MT_K10_SIDECAR}" \
  "${MT_K10_SUMMARY}"

run_checkpoint \
  "smolvla_hiva_coeff_mt_k10_bigcornea_job2_residual_ffn_ce_mean_duration_prefix_w1p0_sigma0p25_b64_s4_20260506_030413_10eps_bs4" \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_mt_k10_bigcornea_job2_residual_ffn_ce_mean_duration_prefix_w1p0_sigma0p25_b64_s4_20260506_030413/checkpoints/last/pretrained_model" \
  "${MT_K10_SIDECAR}" \
  "${MT_K10_SUMMARY}"

run_checkpoint \
  "smolvla_hiva_coeff_hp_k10_residual_ffn_bigflow_job1_ce_mean_full_w1p0_sigma0p25_b160_s4_20260506_032603_10eps_bs4" \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_hp_k10_residual_ffn_bigflow_job1_ce_mean_full_w1p0_sigma0p25_b160_s4_20260506_032603/checkpoints/last/pretrained_model" \
  "${HP_K10_SIDECAR}" \
  "${HP_K10_SUMMARY}"

run_checkpoint \
  "smolvla_hiva_coeff_hp_k10_residual_ffn_bigflow_job2_ce_mean_duration_prefix_w1p0_sigma0p25_b160_s4_20260506_032603_10eps_bs4" \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_hp_k10_residual_ffn_bigflow_job2_ce_mean_duration_prefix_w1p0_sigma0p25_b160_s4_20260506_032603/checkpoints/last/pretrained_model" \
  "${HP_K10_SIDECAR}" \
  "${HP_K10_SUMMARY}"

echo "All residual-FFN K=10 partial evaluations finished at $(date)."
