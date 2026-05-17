#!/usr/bin/env bash
set -euo pipefail

# Partial LIBERO eval for the HP k10 coefficient HiVA job2 checkpoint:
#   S=4, hiva_duration_loss=ce_mean, hiva_suffix_attention=full.
#
# Runs all 4 LIBERO suites in parallel on GPU_IDS=4,5,6,7 by default.
# Each suite evaluates task_ids [0..9] with N_EPISODES=10 and eval.batch_size=4.

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

BIGCORNEA_ROOT="${BIGCORNEA_ROOT:-/nfs/bigcornea/add_disk2/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${BIGCORNEA_ROOT}/libero_lerobot_v3_lerobotkeys}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${BIGCORNEA_ROOT}/libero_hiva_coeff_sidecar_d6_10_15_k10_canonical_hp.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${BIGCORNEA_ROOT}/libero_hiva_coeff_sidecar_d6_10_15_k10_canonical_hp.summary.json}"

POLICY_PATH="${POLICY_PATH:-/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_hp_k10_bigflow_job2_ce_mean_full_w1p0_sigma0p25_b160_s4_20260505_201244/checkpoints/last/pretrained_model}"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-smolvla_hiva_coeff_hp_k10_bigflow_job2_ce_mean_full_w1p0_sigma0p25_b160_s4_20260505_201244_10eps_bs4}"

if [[ ! -x "${HELPER}" ]]; then
  echo "Missing executable helper: ${HELPER}" >&2
  exit 1
fi

echo "TIMESTAMP=${TIMESTAMP}"
echo "GPU_IDS=${GPU_IDS}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "N_EPISODES=${N_EPISODES}"
echo "POLICY_PATH=${POLICY_PATH}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "HIVA_COEFF_SIDECAR=${HIVA_COEFF_SIDECAR}"
echo "HIVA_COEFF_SUMMARY=${HIVA_COEFF_SUMMARY}"

TIMESTAMP="${TIMESTAMP}" \
GPU_IDS="${GPU_IDS}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
N_EPISODES="${N_EPISODES}" \
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED}" \
EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT}" \
EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT}" \
DATA_ROOT="${DATA_ROOT}" \
POLICY_PATH="${POLICY_PATH}" \
CHECKPOINT_LABEL="${CHECKPOINT_LABEL}" \
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR}" \
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY}" \
bash "${HELPER}"
