#!/usr/bin/env bash
set -euo pipefail

# CE-LP-MT HiVA: Constant Execution Long Preview Maximum Target HiVA.
#
# This reproduces the K=8/F=15/S=0.25 BigCornea stage-0 setup from:
#   BestS0p25_81.35_smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k8_f15_bigcornea_b64_s0p25_20260524_185903
# but removes duration prediction entirely. The model trains only the HiVA coefficient
# flow-matching heads and uses constant execution horizon hiva_dmax=10 at inference.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-smolvla_ce_lpmt_hiva_coeffonly_stage0_v5_d4_6_10_k8_f15_bigcornea_b64_s0p25_${RUN_STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/nfs/bigcornea/add_disk3/jongwoopark/HiVA_train/finetuning_stage0}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"

REFERENCE_DIR="${REFERENCE_DIR:-/nfs/bigcornea/add_disk3/jongwoopark/HiVA_train/Best_Models/BestS0p25_81.35_smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k8_f15_bigcornea_b64_s0p25_20260524_185903}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea/add_disk2/jongwoopark}"
SIDECAR="${SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k8_f15_canonical_lp_mt.parquet}"
SIDECAR_SUMMARY="${SIDECAR_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k8_f15_canonical_lp_mt.summary.json}"

for required_path in "${REFERENCE_DIR}" "${SIDECAR}" "${SIDECAR_SUMMARY}" "/home/jongwoopark/lerobot/smolvla_libero"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "ERROR: required path does not exist: ${required_path}" >&2
    exit 2
  fi
done

echo "CE-LP-MT HiVA coefficient-only finetune"
echo "RUN_STAMP=${RUN_STAMP}"
echo "RUN_NAME=${RUN_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "REFERENCE_DIR=${REFERENCE_DIR}"
echo "SIDECAR=${SIDECAR}"
echo "SIDECAR_SUMMARY=${SIDECAR_SUMMARY}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}" \
NUM_GPUS="${NUM_GPUS:-8}" \
NUM_PROCESSES="${NUM_PROCESSES:-8}" \
BATCH_PER_GPU="${BATCH_PER_GPU:-64}" \
BATCH_SIZE="${BATCH_SIZE:-64}" \
S="${S:-0.25}" \
BASE_STEPS="${BASE_STEPS:-20000}" \
STEPS="${STEPS:-10000}" \
SAVE_FREQ="${SAVE_FREQ:-0}" \
SAVE_STEPS="${SAVE_STEPS:-[6000,6250,7000,7500,8750,10000]}" \
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-300}" \
SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-10000}" \
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-2.5e-6}" \
WANDB_ENABLE="${WANDB_ENABLE:-true}" \
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}" \
WANDB_NOTES="${WANDB_NOTES:-CE-LP-MT HiVA coefficient-only ablation of BestS0p25 K8/F15/S0.25; no duration head or duration loss; constant execution horizon 10.}" \
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}" \
INIT_SMOLVLA="${INIT_SMOLVLA:-/home/jongwoopark/lerobot/smolvla_libero}" \
SIDECAR="${SIDECAR}" \
SIDECAR_SUMMARY="${SIDECAR_SUMMARY}" \
HIVA_DURATION_CLASSES="[4,6,10]" \
HIVA_DMAX=10 \
HIVA_FIT_HORIZON=15 \
HIVA_K=8 \
HIVA_DEGREE=3 \
HIVA_BASIS_MODE=canonical_lp_mt \
HIVA_DURATION_PREDICTION_TYPE=none \
HIVA_DURATION_READOUT=none \
HIVA_DURATION_LOSS=none \
HIVA_DURATION_NOISY_LOSS_WEIGHT=0.0 \
HIVA_DURATION_CLEAN_LOSS_WEIGHT=0.0 \
HIVA_DURATION_FM_LOSS_WEIGHT=0.0 \
HIVA_DURATION_HEAD_TYPE=none \
HIVA_SUFFIX_ATTENTION=full \
HIVA_TR_LOSS_WEIGHT=1.0 \
HIVA_ROT_LOSS_WEIGHT=1.0 \
HIVA_GRIP_LOSS_WEIGHT=1.0 \
HIVA_DECODED_ACTION_LOSS_WEIGHT=0.0 \
HIVA_RESIDUAL_ENABLED=false \
POLICY_CHUNK_SIZE=15 \
POLICY_N_ACTION_STEPS=10 \
OUTPUT_DIR="${OUTPUT_DIR}" \
RUN_NAME="${RUN_NAME}" \
bash "${SCRIPT_DIR}/finetune_bigcornea_ckpt_20k_hiva_coeff_common.sh"
