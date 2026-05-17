#!/usr/bin/env bash
set -euo pipefail

# Sequential partial LIBERO evals on bigtoken for:
#   1. HP k10 coeff HiVA, duration_noisy_weights, full suffix attention.
#   2. k6 coeff HiVA lambda0p5, ce_mean, duration_prefix attention.
#
# Each checkpoint evaluates 10 episodes x 10 task ids x 4 suites with
# eval.batch_size=4. The helper runs the 4 suites in parallel over GPU_IDS.
# Dataset and sidecars intentionally come from bigcornea add_disk2.

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
CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_BIN}/python}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

BIGCORNEA_ROOT="${BIGCORNEA_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${BIGCORNEA_ROOT}/libero_lerobot_v3_lerobotkeys}"
DURATION_SIDECAR="${DURATION_SIDECAR:-${BIGCORNEA_ROOT}/libero_duration_sidecar_all_episodes.parquet}"
HP_K10_SIDECAR="${HP_K10_SIDECAR:-${BIGCORNEA_ROOT}/libero_hiva_coeff_sidecar_d6_10_15_k10_canonical_hp.parquet}"
HP_K10_SUMMARY="${HP_K10_SUMMARY:-${BIGCORNEA_ROOT}/libero_hiva_coeff_sidecar_d6_10_15_k10_canonical_hp.summary.json}"
K6_SIDECAR="${K6_SIDECAR:-${BIGCORNEA_ROOT}/libero_hiva_coeff_sidecar_d6_10_15_k6_all_episodes.parquet}"
K6_SUMMARY="${K6_SUMMARY:-${BIGCORNEA_ROOT}/libero_hiva_coeff_sidecar_d6_10_15_k6_all_episodes.summary.json}"

if [[ ! -x "${HELPER}" ]]; then
  echo "Missing executable helper: ${HELPER}" >&2
  exit 1
fi

ensure_hp_k10_sidecar() {
  if [[ -f "${HP_K10_SIDECAR}" && -f "${HP_K10_SUMMARY}" ]]; then
    echo "Using existing bigcornea HP k10 sidecar: ${HP_K10_SIDECAR}"
    return 0
  fi

  echo "Missing bigcornea HP k10 sidecar; building it now from bigcornea dataset."
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "DURATION_SIDECAR=${DURATION_SIDECAR}"
  echo "HP_K10_SIDECAR=${HP_K10_SIDECAR}"
  "${PYTHON_BIN}" "${REPO_ROOT}/src/lerobot/scripts/build_hiva_coeff_sidecar_hp.py" \
    --data-root "${DATA_ROOT}" \
    --duration-sidecar "${DURATION_SIDECAR}" \
    --output "${HP_K10_SIDECAR}" \
    --summary-json "${HP_K10_SUMMARY}" \
    --duration-classes 6 10 15 \
    --duration-map 1:6 3:10 8:15 \
    --dmax 15 \
    --n-ctrl 10 \
    --degree 3 \
    --rot-scale-eta 0.5
}

run_checkpoint() {
  local label="$1"
  local policy_path="$2"
  local sidecar="$3"
  local summary="$4"

  echo "===== Launching ${label} at $(date) ====="
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
  echo "===== Finished ${label} at $(date) ====="
}

ensure_hp_k10_sidecar

run_checkpoint \
  "smolvla_hiva_coeff_hp_k10_bigflow_job1_duration_noisy_weights_full_w1p0_sigma0p25_b160_s4_20260505_201244_10eps_bs4" \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_hp_k10_bigflow_job1_duration_noisy_weights_full_w1p0_sigma0p25_b160_s4_20260505_201244/checkpoints/last/pretrained_model" \
  "${HP_K10_SIDECAR}" \
  "${HP_K10_SUMMARY}"

run_checkpoint \
  "smolvla_hiva_coeff_cleaner_suffix_bigflow_job1_lambda0p5_b160_s4_20260505_170037_10eps_bs4" \
  "/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/HiVA_train/smolvla_hiva_coeff_cleaner_suffix_bigflow_job1_lambda0p5_b160_s4_20260505_143646_20260505_170037_143549051_pid466408/checkpoints/last/pretrained_model" \
  "${K6_SIDECAR}" \
  "${K6_SUMMARY}"

echo "All requested partial evaluations finished at $(date)."
