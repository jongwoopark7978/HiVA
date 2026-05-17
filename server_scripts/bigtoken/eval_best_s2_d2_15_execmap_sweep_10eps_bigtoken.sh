#!/usr/bin/env bash
set -euo pipefail

# Re-evaluate the BestS2 D={2,15} canonical-MT checkpoint with per-env execution
# horizons, sweeping the inference-only mapping for predicted duration 15.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"

POLICY_PATH="${POLICY_PATH:-/home/jongwoopark/lerobot/outputs/train/BestS2_2-15_smolvla_hiva_coeff_mt_d2_15_w1_10_w3_0_residual_ffn_full_ce_mean_k10_bigflow_b160_s2_20260506_232204/checkpoints/last/pretrained_model}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_canonical_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_canonical_mt.summary.json}"

BASE_LABEL="BestS2_2-15_smolvla_hiva_coeff_mt_d2_15_w1_10_w3_0_residual_ffn_full_ce_mean_k10_bigflow_b160_s2_20260506_232204"
MAPPED_DURATIONS=(${MAPPED_DURATIONS:-2 6 10 15})

echo "===== Starting BestS2 D={2,15} duration-execution-map sweep at $(date) ====="
echo "TIMESTAMP=${TIMESTAMP}"
echo "GPU_IDS=${GPU_IDS}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "POLICY_PATH=${POLICY_PATH}"
echo "HIVA_COEFF_SIDECAR=${HIVA_COEFF_SIDECAR}"
echo "HIVA_COEFF_SUMMARY=${HIVA_COEFF_SUMMARY}"
echo "MAPPED_DURATIONS=${MAPPED_DURATIONS[*]}"

for mapped_duration in "${MAPPED_DURATIONS[@]}"; do
  run_timestamp="${TIMESTAMP}_exec15to${mapped_duration}"
  checkpoint_label="${BASE_LABEL}_10eps_bs${EVAL_BATCH_SIZE}_perenv_exec15to${mapped_duration}"
  echo "===== Running mapping 15:${mapped_duration} at $(date) ====="
  POLICY_PATH="${POLICY_PATH}" \
  CHECKPOINT_LABEL="${checkpoint_label}" \
  TIMESTAMP="${run_timestamp}" \
  GPU_IDS="${GPU_IDS}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR}" \
  HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY}" \
  HIVA_DURATION_EXECUTION_MAP="15:${mapped_duration}" \
  bash "${SCRIPT_DIR}/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"
done

echo "===== Finished BestS2 D={2,15} duration-execution-map sweep at $(date) ====="
