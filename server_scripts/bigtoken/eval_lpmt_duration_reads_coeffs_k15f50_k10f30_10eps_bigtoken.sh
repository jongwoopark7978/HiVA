#!/usr/bin/env bash
set -euo pipefail

# Sequential partial LIBERO evals for LP-MT coefficient HiVA checkpoints.
# Runs 10 episodes x 10 task ids x 4 suites, with suites in parallel over
# GPU_IDS and eval.batch_size=4 by default.

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
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${BIGCORNEA_ROOT}/hf_datasets_cache}"

LPMT_K15_F50_SIDECAR="${LPMT_K15_F50_SIDECAR:-${BIGCORNEA_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k15_f50_canonical_lp_mt.parquet}"
LPMT_K15_F50_SUMMARY="${LPMT_K15_F50_SUMMARY:-${BIGCORNEA_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k15_f50_canonical_lp_mt.summary.json}"
LPMT_K10_F30_SIDECAR="${LPMT_K10_F30_SIDECAR:-${BIGCORNEA_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_f30_canonical_lp_mt.parquet}"
LPMT_K10_F30_SUMMARY="${LPMT_K10_F30_SUMMARY:-${BIGCORNEA_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_f30_canonical_lp_mt.summary.json}"

mkdir -p "${HF_DATASETS_CACHE}"

if [[ ! -f "${HELPER}" ]]; then
  echo "Missing helper: ${HELPER}" >&2
  exit 1
fi

run_checkpoint() {
  local label="$1"
  local policy_path="$2"
  local sidecar="$3"
  local summary="$4"
  local chunk_size="$5"

  echo
  echo "======================================================================"
  echo "Starting ${label} at $(date)"
  echo "POLICY_PATH=${policy_path}"
  echo "HIVA_COEFF_SIDECAR=${sidecar}"
  echo "HIVA_COEFF_SUMMARY=${summary}"
  echo "CHUNK_SIZE=${chunk_size}"
  echo "N_ACTION_STEPS=15"
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
  HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
  POLICY_PATH="${policy_path}" \
  CHECKPOINT_LABEL="${label}" \
  HIVA_COEFF_SIDECAR="${sidecar}" \
  HIVA_COEFF_SUMMARY="${summary}" \
  CHUNK_SIZE="${chunk_size}" \
  N_ACTION_STEPS="15" \
  NUM_STEPS="10" \
  bash "${HELPER}"

  echo "Finished ${label} at $(date)"
}

run_checkpoint \
  "smolvla_hiva_coeff_lpmt_d2_15_bigcornea_job1_k15_f50_residual_ffn_ce_mean_duration_reads_coeffs_b64_s2_20260507_190359_10eps_bs4" \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_d2_15_bigcornea_job1_k15_f50_residual_ffn_ce_mean_duration_reads_coeffs_b64_s2_20260507_190359/checkpoints/last/pretrained_model" \
  "${LPMT_K15_F50_SIDECAR}" \
  "${LPMT_K15_F50_SUMMARY}" \
  "50"

run_checkpoint \
  "smolvla_hiva_coeff_lpmt_d2_15_w1_10_w3_0_residual_ffn_duration_reads_coeffs_ce_mean_k10_f30_bigflow_b160_s2_20260507_165638_10eps_bs4" \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_d2_15_w1_10_w3_0_residual_ffn_duration_reads_coeffs_ce_mean_k10_f30_bigflow_b160_s2_20260507_165638/checkpoints/last/pretrained_model" \
  "${LPMT_K10_F30_SIDECAR}" \
  "${LPMT_K10_F30_SUMMARY}" \
  "30"

echo "All LP-MT partial evaluations finished at $(date)."
