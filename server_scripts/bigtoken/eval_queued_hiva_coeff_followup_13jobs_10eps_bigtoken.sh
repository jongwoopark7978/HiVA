#!/usr/bin/env bash
set -euo pipefail

# Queue follow-up partial LIBERO evaluations behind the currently running GPU4-7
# eval sweep. Each eval uses the common 4-suite helper with 10 episodes x 10
# tasks, eval.batch_size=4, and GPUs 4,5,6,7.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
WAIT_PATTERN="${WAIT_PATTERN:-eval_best_s2_d2_15_execmap_sweep_10eps_bigtoken.sh}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark}"
SIDECAR_MT_D2_15_K10="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_canonical_mt.parquet"
SUMMARY_MT_D2_15_K10="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_canonical_mt.summary.json"
SIDECAR_MT_D6_10_15_K10="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d6_10_15_k10_canonical_mt.parquet"
SUMMARY_MT_D6_10_15_K10="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d6_10_15_k10_canonical_mt.summary.json"
SIDECAR_LPMT_D2_15_K10_F30="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_f30_canonical_lp_mt.parquet"
SUMMARY_LPMT_D2_15_K10_F30="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_f30_canonical_lp_mt.summary.json"
SIDECAR_LPMT_D2_15_K20_F30="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k20_f30_canonical_lp_mt.parquet"
SUMMARY_LPMT_D2_15_K20_F30="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k20_f30_canonical_lp_mt.summary.json"
SIDECAR_LPMT_D2_15_K10_F50="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_f50_canonical_lp_mt.parquet"
SUMMARY_LPMT_D2_15_K10_F50="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_f50_canonical_lp_mt.summary.json"
SIDECAR_LPMT_D2_15_K15_F50="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k15_f50_canonical_lp_mt.parquet"
SUMMARY_LPMT_D2_15_K15_F50="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k15_f50_canonical_lp_mt.summary.json"
SIDECAR_LPMT_D2_15_K20_F50="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k20_f50_canonical_lp_mt.parquet"
SUMMARY_LPMT_D2_15_K20_F50="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k20_f50_canonical_lp_mt.summary.json"

wait_for_existing_queue() {
  echo "Waiting for existing GPU4-7 queue pattern: ${WAIT_PATTERN}"
  while pgrep -f "${WAIT_PATTERN}" >/dev/null; do
    date
    pgrep -af "${WAIT_PATTERN}|lerobot-eval.*exec15to" || true
    sleep 300
  done
  echo "Existing queue pattern is clear at $(date)."
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Missing required directory: ${path}" >&2
    exit 1
  fi
}

run_eval() {
  local job_id="$1"
  local policy_path="$2"
  local checkpoint_label="$3"
  local sidecar="$4"
  local summary="$5"
  local chunk_size="$6"
  local duration_map="${7:-}"

  require_dir "${policy_path}"
  require_file "${sidecar}"
  require_file "${summary}"

  echo "===== ${job_id}: ${checkpoint_label} at $(date) ====="
  echo "POLICY_PATH=${policy_path}"
  echo "HIVA_COEFF_SIDECAR=${sidecar}"
  echo "HIVA_COEFF_SUMMARY=${summary}"
  echo "CHUNK_SIZE=${chunk_size}"
  echo "HIVA_DURATION_EXECUTION_MAP=${duration_map}"

  POLICY_PATH="${policy_path}" \
  CHECKPOINT_LABEL="${checkpoint_label}" \
  TIMESTAMP="${TIMESTAMP}_${job_id}" \
  GPU_IDS="${GPU_IDS}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  HIVA_COEFF_SIDECAR="${sidecar}" \
  HIVA_COEFF_SUMMARY="${summary}" \
  CHUNK_SIZE="${chunk_size}" \
  N_ACTION_STEPS=15 \
  HIVA_DURATION_EXECUTION_MAP="${duration_map}" \
  bash "${SCRIPT_DIR}/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"
}

BEST_S2_D2_15="/home/jongwoopark/lerobot/outputs/train/BestS2_2-15_smolvla_hiva_coeff_mt_d2_15_w1_10_w3_0_residual_ffn_full_ce_mean_k10_bigflow_b160_s2_20260506_232204/checkpoints/last/pretrained_model"
BEST_S4_READS="/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/HiVA_train/bestS4_smolvla_hiva_coeff_mt_k10_bigcornea_job1_residual_ffn_ce_mean_duration_reads_coeffs_clean0p0_w1p0_sigma0p25_b64_s4_20260506_201329/checkpoints/last/pretrained_model"
BEST_S2_D6_10_15="/home/jongwoopark/lerobot/outputs/train/BestS2_2-6-15_smolvla_hiva_coeff_mt_k10_bigcornea_job3a_residual_ffn_ce_mean_full_w1p0_sigma0p25_b64_s2_20260506_030413/checkpoints/last/pretrained_model"
MT_DECODED_ROT10="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_mt_decodedloss_d2_15_residual_ffn_duration_reads_coeffs_ce_mean_k10_f15_bigcornea_b64_s2_rot10_20260508_142308/checkpoints/last/pretrained_model"
LPMT_K10_F30="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_d2_15_w1_10_w3_0_residual_ffn_duration_reads_coeffs_ce_mean_k10_f30_bigflow_b160_s2_20260507_165638/checkpoints/last/pretrained_model"
MT_S4_FULL="/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/HiVA_train/smolvla_hiva_coeff_mt_k10_bigcornea_job1_residual_ffn_ce_mean_full_w1p0_sigma0p25_b64_s4_20260506_030413/checkpoints/last/pretrained_model"
LPMT_DECODED_K10_F30="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_decodedloss_d2_15_residual_ffn_duration_reads_coeffs_ce_mean_k10_f30_bigcornea_b64_s2_rot10_job2resume_20260508_194502/checkpoints/last/pretrained_model"
LPMT_DECODED_K20_F30="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_decodedloss_d2_15_residual_ffn_duration_reads_coeffs_ce_mean_k20_f30_rotw10_bigflow_b128_s2_20260508_160755/checkpoints/last/pretrained_model"
LPMT_K10_F50="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_d2_15_w1_10_w3_0_residual_ffn_duration_reads_coeffs_ce_mean_k10_f50_bigflow_b160_s2_20260508_040209/checkpoints/last/pretrained_model"
LPMT_K15_F50="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_d2_15_bigcornea_job1_k15_f50_residual_ffn_ce_mean_duration_reads_coeffs_b64_s2_20260507_190359/checkpoints/last/pretrained_model"
LPMT_K20_F50="/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_d2_15_bigcornea_job2_k20_f50_residual_ffn_ce_mean_duration_reads_coeffs_b64_s2_20260507_190359/checkpoints/last/pretrained_model"

echo "===== Queued HiVA coefficient follow-up evals started at $(date) ====="
echo "TIMESTAMP=${TIMESTAMP}"
echo "GPU_IDS=${GPU_IDS}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"

wait_for_existing_queue

run_eval "job1_map2to1_15to2" "${BEST_S2_D2_15}" "job1_BestS2_d2_15_perenv_map2to1_15to2_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_MT_D2_15_K10}" "${SUMMARY_MT_D2_15_K10}" 15 "2:1,15:2"
run_eval "job1_map2to1_15to6" "${BEST_S2_D2_15}" "job1_BestS2_d2_15_perenv_map2to1_15to6_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_MT_D2_15_K10}" "${SUMMARY_MT_D2_15_K10}" 15 "2:1,15:6"
run_eval "job1_map2to1_15to10" "${BEST_S2_D2_15}" "job1_BestS2_d2_15_perenv_map2to1_15to10_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_MT_D2_15_K10}" "${SUMMARY_MT_D2_15_K10}" 15 "2:1,15:10"
run_eval "job1_map2to1_15to15" "${BEST_S2_D2_15}" "job1_BestS2_d2_15_perenv_map2to1_15to15_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_MT_D2_15_K10}" "${SUMMARY_MT_D2_15_K10}" 15 "2:1,15:15"

run_eval "job2_map2to4_15to4" "${BEST_S2_D2_15}" "job2_BestS2_d2_15_perenv_map2to4_15to4_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_MT_D2_15_K10}" "${SUMMARY_MT_D2_15_K10}" 15 "2:4,15:4"
run_eval "job2_map2to4_15to6" "${BEST_S2_D2_15}" "job2_BestS2_d2_15_perenv_map2to4_15to6_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_MT_D2_15_K10}" "${SUMMARY_MT_D2_15_K10}" 15 "2:4,15:6"
run_eval "job2_map2to4_15to10" "${BEST_S2_D2_15}" "job2_BestS2_d2_15_perenv_map2to4_15to10_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_MT_D2_15_K10}" "${SUMMARY_MT_D2_15_K10}" 15 "2:4,15:10"
run_eval "job2_map2to4_15to15" "${BEST_S2_D2_15}" "job2_BestS2_d2_15_perenv_map2to4_15to15_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_MT_D2_15_K10}" "${SUMMARY_MT_D2_15_K10}" 15 "2:4,15:15"

run_eval "job3" "${BEST_S4_READS}" "job3_bestS4_reads_coeffs_clean0p0_s4_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_MT_D6_10_15_K10}" "${SUMMARY_MT_D6_10_15_K10}" 15 ""
run_eval "job4" "${BEST_S2_D6_10_15}" "job4_BestS2_d6_10_15_full_s2_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_MT_D6_10_15_K10}" "${SUMMARY_MT_D6_10_15_K10}" 15 ""
run_eval "job5" "${MT_DECODED_ROT10}" "job5_mt_decodedloss_d2_15_k10_f15_rot10_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_MT_D2_15_K10}" "${SUMMARY_MT_D2_15_K10}" 15 ""
run_eval "job7" "${LPMT_K10_F30}" "job7_lpmt_d2_15_k10_f30_reads_coeffs_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_LPMT_D2_15_K10_F30}" "${SUMMARY_LPMT_D2_15_K10_F30}" 30 ""
run_eval "job8" "${MT_S4_FULL}" "job8_s4_mt_k10_full_bigcornea_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_MT_D6_10_15_K10}" "${SUMMARY_MT_D6_10_15_K10}" 15 ""
run_eval "job9" "${LPMT_DECODED_K10_F30}" "job9_lpmt_decodedloss_d2_15_k10_f30_rot10_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_LPMT_D2_15_K10_F30}" "${SUMMARY_LPMT_D2_15_K10_F30}" 30 ""
run_eval "job10" "${LPMT_DECODED_K20_F30}" "job10_lpmt_decodedloss_d2_15_k20_f30_rotw10_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_LPMT_D2_15_K20_F30}" "${SUMMARY_LPMT_D2_15_K20_F30}" 30 ""
run_eval "job11" "${LPMT_K10_F50}" "job11_lpmt_d2_15_k10_f50_reads_coeffs_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_LPMT_D2_15_K10_F50}" "${SUMMARY_LPMT_D2_15_K10_F50}" 50 ""
run_eval "job12" "${LPMT_K15_F50}" "job12_lpmt_d2_15_k15_f50_bigcornea_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_LPMT_D2_15_K15_F50}" "${SUMMARY_LPMT_D2_15_K15_F50}" 50 ""
run_eval "job13" "${LPMT_K20_F50}" "job13_lpmt_d2_15_k20_f50_bigcornea_10eps_bs${EVAL_BATCH_SIZE}" "${SIDECAR_LPMT_D2_15_K20_F50}" "${SUMMARY_LPMT_D2_15_K20_F50}" 50 ""

echo "===== Queued HiVA coefficient follow-up evals finished at $(date) ====="
