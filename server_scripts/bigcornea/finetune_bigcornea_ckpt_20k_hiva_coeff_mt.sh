#!/usr/bin/env bash
set -euo pipefail

# Canonical max-target (MT) HiVA coefficient SmolVLA finetuning on bigcornea.
#
# MT decodes with one canonical Dmax x K B-spline basis. Duration predicts the
# executable prefix length, while the coefficient target spans the Dmax horizon.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HIVA_DURATION_CLASSES="${HIVA_DURATION_CLASSES:-[6,10,15]}"
HIVA_DMAX="${HIVA_DMAX:-15}"
HIVA_FIT_HORIZON="${HIVA_FIT_HORIZON:-${HIVA_DMAX}}"
HIVA_K="${HIVA_K:-10}"
HIVA_DEGREE="${HIVA_DEGREE:-3}"
HIVA_BASIS_MODE="${HIVA_BASIS_MODE:-canonical_mt}"
HIVA_DURATION_LOSS="${HIVA_DURATION_LOSS:-ce_mean}"
HIVA_SUFFIX_ATTENTION="${HIVA_SUFFIX_ATTENTION:-duration_reads_coeffs}"
HIVA_DURATION_HEAD_TYPE="${HIVA_DURATION_HEAD_TYPE:-residual_ffn}"
HIVA_RESIDUAL_ENABLED="${HIVA_RESIDUAL_ENABLED:-false}"
HIVA_RESIDUAL_HORIZON="${HIVA_RESIDUAL_HORIZON:-${HIVA_FIT_HORIZON}}"
HIVA_RESIDUAL_FFN_HIDDEN_MULT="${HIVA_RESIDUAL_FFN_HIDDEN_MULT:-4.0}"
HIVA_RESIDUAL_TOKEN_TIME_HIDDEN_MULT="${HIVA_RESIDUAL_TOKEN_TIME_HIDDEN_MULT:-2.0}"
HIVA_RESIDUAL_ALPHA_INIT="${HIVA_RESIDUAL_ALPHA_INIT:-0.1}"
HIVA_RESIDUAL_ZERO_INIT="${HIVA_RESIDUAL_ZERO_INIT:-true}"
HIVA_RESIDUAL_SCALE_TR="${HIVA_RESIDUAL_SCALE_TR:-}"
HIVA_RESIDUAL_SCALE_ROT="${HIVA_RESIDUAL_SCALE_ROT:-}"
HIVA_RESIDUAL_SCALE_GRIP="${HIVA_RESIDUAL_SCALE_GRIP:-}"
DATA_ROOT="${DATA_ROOT:-/nfs/bigcornea/add_disk2/jongwoopark/libero_lerobot_v3_lerobotkeys}"
SIDECAR="${SIDECAR:-/nfs/bigcornea/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d6_10_15_k${HIVA_K}_canonical_mt.parquet}"
SIDECAR_SUMMARY="${SIDECAR_SUMMARY:-/nfs/bigcornea/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d6_10_15_k${HIVA_K}_canonical_mt.summary.json}"

if [[ -z "${RUN_NAME:-}" ]]; then
  RUN_NAME="smolvla_hiva_coeff_mt_${HIVA_DURATION_HEAD_TYPE}_${HIVA_SUFFIX_ATTENTION}_${HIVA_DURATION_LOSS}_k${HIVA_K}_bigcornea_b${BATCH_PER_GPU:-64}_s${S:-2}_$(date +%Y%m%d_%H%M%S)"
fi

export HIVA_DURATION_CLASSES
export HIVA_DMAX
export HIVA_FIT_HORIZON
export HIVA_K
export HIVA_DEGREE
export HIVA_BASIS_MODE
export HIVA_DURATION_LOSS
export HIVA_SUFFIX_ATTENTION
export HIVA_DURATION_HEAD_TYPE
export HIVA_RESIDUAL_ENABLED
export HIVA_RESIDUAL_HORIZON
export HIVA_RESIDUAL_FFN_HIDDEN_MULT
export HIVA_RESIDUAL_TOKEN_TIME_HIDDEN_MULT
export HIVA_RESIDUAL_ALPHA_INIT
export HIVA_RESIDUAL_ZERO_INIT
export HIVA_RESIDUAL_SCALE_TR
export HIVA_RESIDUAL_SCALE_ROT
export HIVA_RESIDUAL_SCALE_GRIP
export DATA_ROOT
export SIDECAR
export SIDECAR_SUMMARY
export RUN_NAME

exec bash "${SCRIPT_DIR}/finetune_bigcornea_ckpt_20k_hiva_coeff_common.sh"
