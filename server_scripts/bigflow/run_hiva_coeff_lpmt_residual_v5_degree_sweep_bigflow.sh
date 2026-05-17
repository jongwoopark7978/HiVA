#!/usr/bin/env bash
set -euo pipefail

# Bigflow LP-MT HiVA residual-action + decoded-action-loss sweep over B-spline degree.
#
# Uses v5 d4/6/10 wide-commit coefficient sidecars:
#   libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p${P}_f15_canonical_lp_mt.parquet
#
# Defaults run one decoded-loss setting (daw=1.0) for P=4,5,6,7. Override
# HIVA_DECODED_ACTION_LOSS_WEIGHTS to sweep decoded-loss weights too.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DEGREES="${DEGREES:-4 5 6 7}"
HIVA_DECODED_ACTION_LOSS_WEIGHTS="${HIVA_DECODED_ACTION_LOSS_WEIGHTS:-1.0}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigflow/add_disk0/jongwoopark}"

GPU_IDS="${GPU_IDS:-4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-160}"
S="${S:-2}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"

HIVA_DURATION_CLASSES="${HIVA_DURATION_CLASSES:-[4,6,10]}"
HIVA_DMAX="${HIVA_DMAX:-10}"
HIVA_FIT_HORIZON="${HIVA_FIT_HORIZON:-15}"
HIVA_K="${HIVA_K:-10}"
HIVA_BASIS_MODE="${HIVA_BASIS_MODE:-canonical_lp_mt}"
HIVA_DURATION_LOSS="${HIVA_DURATION_LOSS:-ce_mean}"
HIVA_SUFFIX_ATTENTION="${HIVA_SUFFIX_ATTENTION:-full}"
HIVA_DURATION_READOUT="${HIVA_DURATION_READOUT:-coeff_modality_pool}"
HIVA_DURATION_HEAD_TYPE="${HIVA_DURATION_HEAD_TYPE:-residual_ffn}"

HIVA_DECODED_TR_LOSS_WEIGHT="${HIVA_DECODED_TR_LOSS_WEIGHT:-1.0}"
HIVA_DECODED_ROT_LOSS_WEIGHT="${HIVA_DECODED_ROT_LOSS_WEIGHT:-10.0}"
HIVA_DECODED_GRIP_LOSS_WEIGHT="${HIVA_DECODED_GRIP_LOSS_WEIGHT:-1.0}"
HIVA_DECODED_PREFIX_WEIGHT="${HIVA_DECODED_PREFIX_WEIGHT:-1.0}"
HIVA_DECODED_POST_DURATION_EXEC_WEIGHT="${HIVA_DECODED_POST_DURATION_EXEC_WEIGHT:-0.5}"
HIVA_DECODED_PREVIEW_WEIGHT="${HIVA_DECODED_PREVIEW_WEIGHT:-0.1}"
HIVA_DECODED_TERMINAL_WEIGHT="${HIVA_DECODED_TERMINAL_WEIGHT:-0.0}"
HIVA_DECODED_LOSS_BETA="${HIVA_DECODED_LOSS_BETA:-0.1}"

HIVA_RESIDUAL_ENABLED="${HIVA_RESIDUAL_ENABLED:-true}"
HIVA_RESIDUAL_HORIZON="${HIVA_RESIDUAL_HORIZON:-${HIVA_FIT_HORIZON}}"
HIVA_RESIDUAL_FFN_HIDDEN_MULT="${HIVA_RESIDUAL_FFN_HIDDEN_MULT:-4.0}"
HIVA_RESIDUAL_TOKEN_TIME_HIDDEN_MULT="${HIVA_RESIDUAL_TOKEN_TIME_HIDDEN_MULT:-2.0}"
HIVA_RESIDUAL_ALPHA_INIT="${HIVA_RESIDUAL_ALPHA_INIT:-0.1}"
HIVA_RESIDUAL_ZERO_INIT="${HIVA_RESIDUAL_ZERO_INIT:-true}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

read -r -a DEGREE_ARRAY <<< "${DEGREES}"
read -r -a DAW_ARRAY <<< "${HIVA_DECODED_ACTION_LOSS_WEIGHTS}"

echo "LP-MT residual v5 degree sweep started at $(date)"
echo "DEGREES=${DEGREE_ARRAY[*]}"
echo "HIVA_DECODED_ACTION_LOSS_WEIGHTS=${DAW_ARRAY[*]}"
echo "GPU_IDS=${GPU_IDS}"
echo "BATCH_PER_GPU=${BATCH_PER_GPU}"
echo "S=${S}"

for degree in "${DEGREE_ARRAY[@]}"; do
  sidecar="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k${HIVA_K}_p${degree}_f${HIVA_FIT_HORIZON}_canonical_lp_mt.parquet"
  summary="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k${HIVA_K}_p${degree}_f${HIVA_FIT_HORIZON}_canonical_lp_mt.summary.json"
  if [[ ! -f "${sidecar}" || ! -f "${summary}" ]]; then
    echo "Missing degree-${degree} sidecar or summary:" >&2
    echo "  ${sidecar}" >&2
    echo "  ${summary}" >&2
    exit 2
  fi

  for decoded_weight in "${DAW_ARRAY[@]}"; do
    daw_tag="${decoded_weight//./p}"
    run_name="smolvla_hiva_coeff_lpmt_residual_v5_d4_6_10_coeffpool_full_ce_mean_k${HIVA_K}_p${degree}_f${HIVA_FIT_HORIZON}_bigflow_b${BATCH_PER_GPU}_s${S}_daw${daw_tag}_rot10_${RUN_STAMP}"
    echo
    echo "======================================================================"
    echo "Launching ${run_name}"
    echo "SIDECAR=${sidecar}"
    echo "SUMMARY=${summary}"
    echo "HIVA_DEGREE=${degree}"
    echo "HIVA_DECODED_ACTION_LOSS_WEIGHT=${decoded_weight}"
    echo "======================================================================"

    GPU_IDS="${GPU_IDS}" \
    NUM_GPUS="${NUM_GPUS}" \
    NUM_PROCESSES="${NUM_PROCESSES}" \
    BATCH_PER_GPU="${BATCH_PER_GPU}" \
    S="${S}" \
    WANDB_ENABLE="${WANDB_ENABLE}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
    HIVA_DURATION_CLASSES="${HIVA_DURATION_CLASSES}" \
    HIVA_DMAX="${HIVA_DMAX}" \
    HIVA_FIT_HORIZON="${HIVA_FIT_HORIZON}" \
    HIVA_K="${HIVA_K}" \
    HIVA_DEGREE="${degree}" \
    HIVA_BASIS_MODE="${HIVA_BASIS_MODE}" \
    HIVA_DURATION_LOSS="${HIVA_DURATION_LOSS}" \
    HIVA_SUFFIX_ATTENTION="${HIVA_SUFFIX_ATTENTION}" \
    HIVA_DURATION_READOUT="${HIVA_DURATION_READOUT}" \
    HIVA_DURATION_HEAD_TYPE="${HIVA_DURATION_HEAD_TYPE}" \
    HIVA_DECODED_ACTION_LOSS_WEIGHT="${decoded_weight}" \
    HIVA_DECODED_TR_LOSS_WEIGHT="${HIVA_DECODED_TR_LOSS_WEIGHT}" \
    HIVA_DECODED_ROT_LOSS_WEIGHT="${HIVA_DECODED_ROT_LOSS_WEIGHT}" \
    HIVA_DECODED_GRIP_LOSS_WEIGHT="${HIVA_DECODED_GRIP_LOSS_WEIGHT}" \
    HIVA_DECODED_PREFIX_WEIGHT="${HIVA_DECODED_PREFIX_WEIGHT}" \
    HIVA_DECODED_POST_DURATION_EXEC_WEIGHT="${HIVA_DECODED_POST_DURATION_EXEC_WEIGHT}" \
    HIVA_DECODED_PREVIEW_WEIGHT="${HIVA_DECODED_PREVIEW_WEIGHT}" \
    HIVA_DECODED_TERMINAL_WEIGHT="${HIVA_DECODED_TERMINAL_WEIGHT}" \
    HIVA_DECODED_LOSS_BETA="${HIVA_DECODED_LOSS_BETA}" \
    HIVA_RESIDUAL_ENABLED="${HIVA_RESIDUAL_ENABLED}" \
    HIVA_RESIDUAL_HORIZON="${HIVA_RESIDUAL_HORIZON}" \
    HIVA_RESIDUAL_FFN_HIDDEN_MULT="${HIVA_RESIDUAL_FFN_HIDDEN_MULT}" \
    HIVA_RESIDUAL_TOKEN_TIME_HIDDEN_MULT="${HIVA_RESIDUAL_TOKEN_TIME_HIDDEN_MULT}" \
    HIVA_RESIDUAL_ALPHA_INIT="${HIVA_RESIDUAL_ALPHA_INIT}" \
    HIVA_RESIDUAL_ZERO_INIT="${HIVA_RESIDUAL_ZERO_INIT}" \
    POLICY_CHUNK_SIZE="${HIVA_FIT_HORIZON}" \
    POLICY_N_ACTION_STEPS="${HIVA_DMAX}" \
    SIDECAR="${sidecar}" \
    SIDECAR_SUMMARY="${summary}" \
    RUN_NAME="${run_name}" \
    bash "${SCRIPT_DIR}/finetune_bigflow_ckpt_20k_hiva_coeff_lp_mt.sh"
  done
done

echo "LP-MT residual v5 degree sweep finished at $(date)"
