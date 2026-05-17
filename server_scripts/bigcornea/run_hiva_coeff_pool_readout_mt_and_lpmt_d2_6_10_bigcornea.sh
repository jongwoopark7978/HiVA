#!/usr/bin/env bash
set -euo pipefail

# Bigcornea follow-up sweep:
#   job1: MT d={2,15}, K=10, full attention, coeff_modality_pool duration readout.
#   job2: LP-MT K=10, preview/fit horizon=15, duration classes {2,6} then {2,10}.
#
# The LP-MT coefficient sidecars use freshly recomputed horizon-safe duration sidecars for
# d={2,6} and d={2,10}. Do not post-hoc map 15 -> 6/10; that would not respect each frame's
# switch/contact horizon.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/train_logs}"
mkdir -p "${LOG_DIR}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-64}"
S="${S:-2}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"

DATA_ROOT="${DATA_ROOT:-/nfs/bigcornea/add_disk2/jongwoopark/libero_lerobot_v3_lerobotkeys}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea/add_disk2/jongwoopark}"
DURATION_SIDECAR="${DURATION_SIDECAR:-${SIDECAR_ROOT}/libero_duration_sidecar_d2_15_w1_10_w3_0_all_episodes.parquet}"
CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_BIN}/python}"

BASE_INIT="${BASE_INIT:-/home/jongwoopark/lerobot/smolvla_libero}"
MT_INIT="${MT_INIT:-${BASE_INIT}}"
LPMT_INIT="${LPMT_INIT:-${BASE_INIT}}"

COMMON_ENV=(
  GPU_IDS="${GPU_IDS}"
  NUM_GPUS="${NUM_GPUS}"
  NUM_PROCESSES="${NUM_PROCESSES}"
  BATCH_PER_GPU="${BATCH_PER_GPU}"
  S="${S}"
  WANDB_ENABLE="${WANDB_ENABLE}"
  WANDB_PROJECT="${WANDB_PROJECT}"
  DATA_ROOT="${DATA_ROOT}"
  EVAL_FREQ="${EVAL_FREQ:-0}"
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
  EVAL_N_EPISODES="${EVAL_N_EPISODES:-1}"
  EVAL_TASK_IDS="${EVAL_TASK_IDS:-}"
  EVAL_MAX_PARALLEL_TASKS="${EVAL_MAX_PARALLEL_TASKS:-1}"
  HIVA_K=10
  HIVA_DEGREE=3
  HIVA_DURATION_LOSS=ce_mean
  HIVA_DURATION_HEAD_TYPE=residual_ffn
  HIVA_DURATION_NOISY_LOSS_WEIGHT="${HIVA_DURATION_NOISY_LOSS_WEIGHT:-1.0}"
  HIVA_DURATION_CLEAN_LOSS_WEIGHT="${HIVA_DURATION_CLEAN_LOSS_WEIGHT:-0.0}"
  HIVA_DURATION_NOISY_SIGMA="${HIVA_DURATION_NOISY_SIGMA:-0.25}"
)

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing required file: ${path}" >&2
    exit 2
  fi
}

build_lpmt_sidecar() {
  local dmax="$1"
  local duration_sidecar="${SIDECAR_ROOT}/libero_duration_sidecar_d2_${dmax}_w1_10_w3_0_all_episodes.parquet"
  local duration_summary="${SIDECAR_ROOT}/libero_duration_sidecar_d2_${dmax}_w1_10_w3_0_all_episodes.summary.json"
  local out="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_${dmax}_w1_10_w3_0_k10_f15_canonical_lp_mt.parquet"
  local summary="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_${dmax}_w1_10_w3_0_k10_f15_canonical_lp_mt.summary.json"
  local duration_log="${LOG_DIR}/build_hiva_duration_sidecar_d2_${dmax}_w1_10_w3_0_${RUN_STAMP}.log"
  local log="${LOG_DIR}/build_hiva_coeff_sidecar_d2_${dmax}_k10_f15_${RUN_STAMP}.log"

  if [[ ! -f "${duration_sidecar}" || ! -f "${duration_summary}" ]]; then
    echo "Building horizon-safe duration sidecar d={2,${dmax}} -> ${duration_sidecar}"
    "${PYTHON_BIN}" -m lerobot.scripts.build_hiva_duration_sidecar \
      --dataset.repo-id local/libero_lerobot_v3_lerobotkeys \
      --dataset.root "${DATA_ROOT}" \
      --output "${duration_sidecar}" \
      --summary-json "${duration_summary}" \
      --w1 10 \
      --w3 0 \
      --durations "2,${dmax}" \
      --labeler-version "hiva_duration_d2_${dmax}_w1_10_w3_0" \
      2>&1 | tee "${duration_log}"
  else
    echo "Using existing horizon-safe duration sidecar d={2,${dmax}}: ${duration_sidecar}"
  fi

  if [[ -f "${out}" && -f "${summary}" ]]; then
    echo "Sidecar already exists for d={2,${dmax}}: ${out}"
    return 0
  fi

  echo "Building LP-MT sidecar d={2,${dmax}}, K=10, f15 -> ${out}"
  "${PYTHON_BIN}" src/lerobot/scripts/build_hiva_coeff_sidecar_lp_mt.py \
    --data-root "${DATA_ROOT}" \
    --duration-sidecar "${duration_sidecar}" \
    --output "${out}" \
    --summary-json "${summary}" \
    --duration-classes 2 "${dmax}" \
    --dmax "${dmax}" \
    --fit-horizon 15 \
    --n-ctrl 10 \
    --degree 3 \
    --target-mode max_target \
    2>&1 | tee "${log}"
}

build_sidecars() {
  build_lpmt_sidecar 6
  build_lpmt_sidecar 10
}

run_mt_pool_readout() {
  echo "===== Starting job1 MT coeff_modality_pool duration readout at $(date) ====="
  env "${COMMON_ENV[@]}" \
    INIT_SMOLVLA="${MT_INIT}" \
    HIVA_DURATION_CLASSES="[2,15]" \
    HIVA_DMAX=15 \
    HIVA_FIT_HORIZON=15 \
    HIVA_BASIS_MODE=canonical_mt \
    HIVA_SUFFIX_ATTENTION=full \
    HIVA_DURATION_READOUT=coeff_modality_pool \
    POLICY_CHUNK_SIZE=15 \
    POLICY_N_ACTION_STEPS=15 \
    SIDECAR="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_canonical_mt.parquet" \
    SIDECAR_SUMMARY="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_canonical_mt.summary.json" \
    RUN_NAME="smolvla_hiva_coeff_mt_d2_15_w1_10_w3_0_coeffpool_full_ce_mean_k10_bigcornea_b${BATCH_PER_GPU}_s${S}_${RUN_STAMP}" \
    bash "${SCRIPT_DIR}/finetune_bigcornea_ckpt_20k_hiva_coeff_mt.sh"
  echo "===== Finished job1 MT coeff_modality_pool duration readout at $(date) ====="
}

run_lpmt_duration_pair() {
  local dmax="$1"
  echo "===== Starting job2 LP-MT d={2,${dmax}} K=10 f15 at $(date) ====="
  env "${COMMON_ENV[@]}" \
    INIT_SMOLVLA="${LPMT_INIT}" \
    HIVA_DURATION_CLASSES="[2,${dmax}]" \
    HIVA_DMAX="${dmax}" \
    HIVA_FIT_HORIZON=15 \
    HIVA_BASIS_MODE=canonical_lp_mt \
    HIVA_SUFFIX_ATTENTION=duration_reads_coeffs \
    HIVA_DURATION_READOUT=token \
    POLICY_CHUNK_SIZE=15 \
    POLICY_N_ACTION_STEPS="${dmax}" \
    SIDECAR="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_${dmax}_w1_10_w3_0_k10_f15_canonical_lp_mt.parquet" \
    SIDECAR_SUMMARY="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_${dmax}_w1_10_w3_0_k10_f15_canonical_lp_mt.summary.json" \
    RUN_NAME="smolvla_hiva_coeff_lpmt_d2_${dmax}_w1_10_w3_0_residual_ffn_duration_reads_coeffs_ce_mean_k10_f15_bigcornea_b${BATCH_PER_GPU}_s${S}_${RUN_STAMP}" \
    bash "${SCRIPT_DIR}/finetune_bigcornea_ckpt_20k_hiva_coeff_lp_mt.sh"
  echo "===== Finished job2 LP-MT d={2,${dmax}} K=10 f15 at $(date) ====="
}

echo "HiVA coeff pool-readout / LP-MT d2_6 d2_10 sweep started at $(date)"
echo "Host: $(hostname)"
echo "RUN_STAMP=${RUN_STAMP}"
echo "GPU_IDS=${GPU_IDS}"
echo "BATCH_PER_GPU=${BATCH_PER_GPU}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "SIDECAR_ROOT=${SIDECAR_ROOT}"
require_file "${MT_INIT}/model.safetensors"
require_file "${LPMT_INIT}/model.safetensors"
echo "MT_INIT=${MT_INIT}"
echo "LPMT_INIT=${LPMT_INIT}"
require_file "${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_canonical_mt.parquet"
require_file "${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_d2_15_w1_10_w3_0_k10_canonical_mt.summary.json"

build_sidecars &
SIDECAR_BUILD_PID=$!

run_mt_pool_readout

wait "${SIDECAR_BUILD_PID}"

run_lpmt_duration_pair 6
run_lpmt_duration_pair 10

echo "HiVA coeff pool-readout / LP-MT d2_6 d2_10 sweep finished at $(date)"
