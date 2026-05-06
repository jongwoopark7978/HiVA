#!/usr/bin/env bash
set -euo pipefail

# Sequential partial LIBERO evaluation queue for normalized-duration-loss
# coefficient HiVA checkpoints with lambda_duration=1.0 on bigtoken.
# Each checkpoint runs the four suites in parallel on GPU_IDS.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
TASK_IDS_ALL="${TASK_IDS_ALL:-[0,1,2,3,4,5,6,7,8,9]}"
N_EPISODES="${N_EPISODES:-10}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-1}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-1}"
EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT:-400}"
EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT:-40}"

POLICY_PATHS=(
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_cleaner_suffix_bigcornea_sigma0p25_w1p0_b80_s4_durw_sweep_b80_s4_20260504_175101/checkpoints/last/pretrained_model"
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_cleaner_suffix_bigcornea_J1_sigma0p20_w1p0_b80_s4_j1j2_b80_w1_20260505_023954/checkpoints/last/pretrained_model"
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_cleaner_suffix_bigcornea_J1_sigma0p30_w1p0_b80_s4_j1j2_b80_w1_20260505_023954/checkpoints/last/pretrained_model"
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_cleaner_suffix_bigcornea_J2_sigma0p25_w1p0_b80_s2_j1j2_b80_w1_20260505_023954/checkpoints/last/pretrained_model"
)

CHECKPOINT_LABELS=(
  "smolvla_hiva_coeff_cleaner_suffix_bigcornea_sigma0p25_w1p0_b80_s4_20260504_175101_10eps_bs4"
  "smolvla_hiva_coeff_cleaner_suffix_bigcornea_J1_sigma0p20_w1p0_b80_s4_20260505_023954_10eps_bs4"
  "smolvla_hiva_coeff_cleaner_suffix_bigcornea_J1_sigma0p30_w1p0_b80_s4_20260505_023954_10eps_bs4"
  "smolvla_hiva_coeff_cleaner_suffix_bigcornea_J2_sigma0p25_w1p0_b80_s2_20260505_023954_10eps_bs4"
)

echo "===== normalized-duration w1 sweep started at $(date) ====="
echo "TIMESTAMP=${TIMESTAMP}"
echo "GPU_IDS=${GPU_IDS}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "TASK_IDS_ALL=${TASK_IDS_ALL}"
echo "N_EPISODES=${N_EPISODES}"
echo "MAX_EPISODES_RENDERED=${MAX_EPISODES_RENDERED}"

for idx in "${!POLICY_PATHS[@]}"; do
  policy_path="${POLICY_PATHS[$idx]}"
  checkpoint_label="${CHECKPOINT_LABELS[$idx]}"

  echo "===== Starting ${checkpoint_label} at $(date) ====="
  echo "POLICY_PATH=${policy_path}"
  if [[ ! -d "${policy_path}" ]]; then
    echo "Missing policy directory: ${policy_path}" >&2
    exit 1
  fi

  env \
    TIMESTAMP="${TIMESTAMP}" \
    GPU_IDS="${GPU_IDS}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    TASK_IDS_ALL="${TASK_IDS_ALL}" \
    N_EPISODES="${N_EPISODES}" \
    MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS}" \
    MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED}" \
    EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT}" \
    EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT}" \
    POLICY_PATH="${policy_path}" \
    CHECKPOINT_LABEL="${checkpoint_label}" \
    bash "${SCRIPT_DIR}/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"

  echo "===== Finished ${checkpoint_label} at $(date) ====="
done

echo "===== normalized-duration w1 sweep finished at $(date) ====="
