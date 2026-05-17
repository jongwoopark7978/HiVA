#!/usr/bin/env bash
set -euo pipefail

# Stage-0 LP-MT HiVA coeff-pool finetune for v5 d4/6/10 with alternate
# coefficient-token counts. Reuses the v5 duration sidecar, rebuilds only the
# coefficient sidecars for K=12 and K=6, then finetunes K=12 followed by K=6.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_BIN}/python}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-64}"
S="${S:-0.5}"
BASE_STEPS="${BASE_STEPS:-20000}"
BASE_NUM_GPUS="${BASE_NUM_GPUS:-1}"
BASE_BATCH_PER_GPU="${BASE_BATCH_PER_GPU:-64}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"

K_VALUES="${K_VALUES:-12 6}"
F="${F:-15}"
DMAX="${DMAX:-10}"
DEGREE="${DEGREE:-3}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
S_TAG="${S//./p}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea/add_disk2/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
DURATION_SIDECAR="${DURATION_SIDECAR:-${SIDECAR_ROOT}/libero_duration_sidecar_v5_d4_6_10_wide_commit6_all_episodes.parquet}"
INIT_SMOLVLA="${INIT_SMOLVLA:-/home/jongwoopark/lerobot/smolvla_libero}"
COEFF_BUILDER="${COEFF_BUILDER:-${REPO_ROOT}/src/lerobot/scripts/build_hiva_coeff_sidecar_lp_mt.py}"
OVERWRITE_SIDECARS="${OVERWRITE_SIDECARS:-false}"

GLOBAL_BATCH_SIZE=$((NUM_GPUS * BATCH_PER_GPU))
BASE_GLOBAL_BATCH_SIZE=$((BASE_NUM_GPUS * BASE_BATCH_PER_GPU))
STEPS="${STEPS:-$("${PYTHON_BIN}" - <<PY
import math
steps = math.ceil(int("${BASE_STEPS}") * int("${BASE_GLOBAL_BATCH_SIZE}") / int("${GLOBAL_BATCH_SIZE}") / float("${S}"))
print(max(1, steps))
PY
)}"
SAVE_STEPS="${SAVE_STEPS:-$("${PYTHON_BIN}" - <<PY
import math
steps = int("${STEPS}")
fractions = (0.6, 0.625, 0.7, 0.75, 0.875, 1.0)
milestones = sorted({min(steps, max(1, math.ceil(steps * f))) for f in fractions})
print("[" + ",".join(str(s) for s in milestones) + "]")
PY
)}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-$("${PYTHON_BIN}" - <<PY
import math
print(max(1, math.ceil(int("${STEPS}") * float("${WARMUP_RATIO}"))))
PY
)}"

for required_path in "${DATA_ROOT}" "${DURATION_SIDECAR}" "${INIT_SMOLVLA}" "${COEFF_BUILDER}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "ERROR: required path does not exist: ${required_path}" >&2
    exit 2
  fi
done

coeff_sidecar_path() {
  local k="$1"
  echo "${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k${k}_f${F}_canonical_lp_mt.parquet"
}

coeff_summary_path() {
  local k="$1"
  echo "${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k${k}_f${F}_canonical_lp_mt.summary.json"
}

build_coeff_sidecar() {
  local k="$1"
  local output
  local summary
  output="$(coeff_sidecar_path "${k}")"
  summary="$(coeff_summary_path "${k}")"

  echo
  echo "======================================================================"
  echo "Building/checking LP-MT v5 coefficient sidecar for K=${k}"
  echo "DURATION_SIDECAR=${DURATION_SIDECAR}"
  echo "OUTPUT=${output}"
  echo "SUMMARY=${summary}"
  echo "K=${k}, F=${F}, DMAX=${DMAX}, DEGREE=${DEGREE}"
  echo "======================================================================"

  if [[ "${OVERWRITE_SIDECARS}" == "true" ]]; then
    rm -f "${output}" "${summary}"
  fi

  if [[ -s "${output}" && -s "${summary}" ]]; then
    echo "Sidecar already exists; skipping build. Set OVERWRITE_SIDECARS=true to rebuild."
    return
  fi

  "${PYTHON_BIN}" "${COEFF_BUILDER}" \
    --data-root "${DATA_ROOT}" \
    --duration-sidecar "${DURATION_SIDECAR}" \
    --output "${output}" \
    --summary-json "${summary}" \
    --duration-classes 4 6 10 \
    --dmax "${DMAX}" \
    --fit-horizon "${F}" \
    --n-ctrl "${k}" \
    --degree "${DEGREE}" \
    --rot-scale-eta 0.5 \
    --preview-tail-weight 1.0 \
    --smooth 0.0
}

echo "Stage-0 LP-MT HiVA v5 K sweep"
echo "K_VALUES=${K_VALUES}"
echo "GPU_IDS=${GPU_IDS} NUM_GPUS=${NUM_GPUS} BATCH_PER_GPU=${BATCH_PER_GPU} S=${S}"
echo "BASE_STEPS=${BASE_STEPS} BASE_GLOBAL_BATCH_SIZE=${BASE_GLOBAL_BATCH_SIZE} GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "STEPS=${STEPS} SAVE_STEPS=${SAVE_STEPS} SCHEDULER_WARMUP_STEPS=${SCHEDULER_WARMUP_STEPS}"
echo "INIT_SMOLVLA=${INIT_SMOLVLA}"

for k in ${K_VALUES}; do
  build_coeff_sidecar "${k}"
done

for k in ${K_VALUES}; do
  SIDECAR="$(coeff_sidecar_path "${k}")"
  SIDECAR_SUMMARY="$(coeff_summary_path "${k}")"
  RUN_NAME="smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k${k}_f${F}_bigcornea_b${BATCH_PER_GPU}_s${S_TAG}_${RUN_STAMP}"

  for required_path in "${SIDECAR}" "${SIDECAR_SUMMARY}"; do
    if [[ ! -s "${required_path}" ]]; then
      echo "ERROR: required sidecar path does not exist or is empty: ${required_path}" >&2
      exit 3
    fi
  done

  echo
  echo "======================================================================"
  echo "Finetuning K=${k}: ${RUN_NAME}"
  echo "SIDECAR=${SIDECAR}"
  echo "SAVE_STEPS=${SAVE_STEPS}"
  echo "======================================================================"

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
  HIVA_DMAX="${DMAX}" \
  HIVA_FIT_HORIZON="${F}" \
  HIVA_K="${k}" \
  HIVA_DEGREE="${DEGREE}" \
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
done
