#!/usr/bin/env bash
set -euo pipefail

# Stage-0 LP-MT HiVA coeff-pool finetune matching the BestS2 v5 d4/6/10 run,
# with S=0.125. With 8 GPUs x batch 64 this gives 20k total steps.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-64}"
S="${S:-0.125}"
BASE_STEPS="${BASE_STEPS:-20000}"
BASE_NUM_GPUS="${BASE_NUM_GPUS:-1}"
BASE_BATCH_PER_GPU="${BASE_BATCH_PER_GPU:-64}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"

GLOBAL_BATCH_SIZE=$((NUM_GPUS * BATCH_PER_GPU))
BASE_GLOBAL_BATCH_SIZE=$((BASE_NUM_GPUS * BASE_BATCH_PER_GPU))
STEPS="${STEPS:-$("${CONDA_ENV_BIN}/python" - <<PY
import math
steps = math.ceil(int("${BASE_STEPS}") * int("${BASE_GLOBAL_BATCH_SIZE}") / int("${GLOBAL_BATCH_SIZE}") / float("${S}"))
print(max(1, steps))
PY
)}"
SAVE_STEPS="${SAVE_STEPS:-[10000,12500,13000,13500,14000,14500,15000,16000,17500,20000]}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-$("${CONDA_ENV_BIN}/python" - <<PY
import math
print(max(1, math.ceil(int("${STEPS}") * float("${WARMUP_RATIO}"))))
PY
)}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
S_TAG="${S//./p}"
RUN_NAME="${RUN_NAME:-smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigcornea_b${BATCH_PER_GPU}_s${S_TAG}_${RUN_STAMP}}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea/add_disk2/jongwoopark}"
SIDECAR="${SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
SIDECAR_SUMMARY="${SIDECAR_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
INIT_SMOLVLA="${INIT_SMOLVLA:-/home/jongwoopark/lerobot/smolvla_libero}"

for required_path in "${SIDECAR}" "${SIDECAR_SUMMARY}" "${DATA_ROOT}" "${INIT_SMOLVLA}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "ERROR: required path does not exist: ${required_path}" >&2
    exit 2
  fi
done

echo "Stage-0 LP-MT HiVA v5 S=0.125 milestone finetune"
echo "RUN_NAME=${RUN_NAME}"
echo "GPU_IDS=${GPU_IDS} NUM_GPUS=${NUM_GPUS} BATCH_PER_GPU=${BATCH_PER_GPU} S=${S}"
echo "BASE_STEPS=${BASE_STEPS} BASE_GLOBAL_BATCH_SIZE=${BASE_GLOBAL_BATCH_SIZE} GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "STEPS=${STEPS} SAVE_STEPS=${SAVE_STEPS} SCHEDULER_WARMUP_STEPS=${SCHEDULER_WARMUP_STEPS}"
echo "SIDECAR=${SIDECAR}"
echo "SIDECAR_SUMMARY=${SIDECAR_SUMMARY}"
echo "INIT_SMOLVLA=${INIT_SMOLVLA}"

GPU_IDS="${GPU_IDS}" \
NUM_GPUS="${NUM_GPUS}" \
NUM_PROCESSES="${NUM_PROCESSES}" \
BATCH_PER_GPU="${BATCH_PER_GPU}" \
S="${S}" \
BASE_STEPS="${BASE_STEPS}" \
BASE_NUM_GPUS="${BASE_NUM_GPUS}" \
BASE_BATCH_PER_GPU="${BASE_BATCH_PER_GPU}" \
STEPS="${STEPS}" \
SAVE_FREQ=0 \
SAVE_STEPS="${SAVE_STEPS}" \
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS}" \
SCHEDULER_DECAY_STEPS="${STEPS}" \
WANDB_ENABLE="${WANDB_ENABLE:-true}" \
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}" \
DATA_ROOT="${DATA_ROOT}" \
INIT_SMOLVLA="${INIT_SMOLVLA}" \
SIDECAR="${SIDECAR}" \
SIDECAR_SUMMARY="${SIDECAR_SUMMARY}" \
HIVA_DURATION_CLASSES="[4,6,10]" \
HIVA_DMAX=10 \
HIVA_FIT_HORIZON=15 \
HIVA_K=10 \
HIVA_DEGREE=3 \
HIVA_BASIS_MODE=canonical_lp_mt \
HIVA_DURATION_LOSS=ce_mean \
HIVA_SUFFIX_ATTENTION=full \
HIVA_DURATION_READOUT=coeff_modality_pool \
HIVA_DURATION_HEAD_TYPE=residual_ffn \
HIVA_DURATION_PREDICTION_TYPE=categorical \
HIVA_TR_LOSS_WEIGHT=1.0 \
HIVA_ROT_LOSS_WEIGHT=1.0 \
HIVA_GRIP_LOSS_WEIGHT=1.0 \
HIVA_DURATION_NOISY_LOSS_WEIGHT=1.0 \
HIVA_DURATION_CLEAN_LOSS_WEIGHT=0.0 \
HIVA_DURATION_FM_LOSS_WEIGHT=1.0 \
HIVA_DURATION_NOISY_SIGMA=0.25 \
HIVA_DECODED_ACTION_LOSS_WEIGHT=0.0 \
HIVA_RESIDUAL_ENABLED=false \
POLICY_CHUNK_SIZE=15 \
POLICY_N_ACTION_STEPS=10 \
RUN_NAME="${RUN_NAME}" \
bash "${SCRIPT_DIR}/finetune_bigcornea_ckpt_20k_hiva_coeff_common.sh"
