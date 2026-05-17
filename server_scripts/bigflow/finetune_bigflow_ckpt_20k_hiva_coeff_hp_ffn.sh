#!/usr/bin/env bash
set -euo pipefail

# HP-basis cleaner-suffix HiVA coefficient finetuning with a stronger duration head.
#
# This launcher reuses the canonical hold-pad HP sidecar from the base HP script, but opts into
# hiva_duration_head_type=residual_ffn instead of the legacy single linear duration classifier.
#
# Example:
#   GPU_IDS=4,5,6,7 NUM_GPUS=4 BATCH_PER_GPU=160 S=2 \
#   WANDB_ENABLE=true WANDB_PROJECT=lerobot \
#   bash server_scripts/bigflow/finetune_bigflow_ckpt_20k_hiva_coeff_hp_ffn.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HIVA_BASIS_MODE="${HIVA_BASIS_MODE:-canonical_hp}"
export HIVA_K="${HIVA_K:-10}"
export HIVA_DURATION_LOSS="${HIVA_DURATION_LOSS:-ce_mean}"
export HIVA_SUFFIX_ATTENTION="${HIVA_SUFFIX_ATTENTION:-duration_prefix}"
export HIVA_DURATION_HEAD_TYPE="${HIVA_DURATION_HEAD_TYPE:-residual_ffn}"
export HIVA_DURATION_FFN_HIDDEN_MULT="${HIVA_DURATION_FFN_HIDDEN_MULT:-4.0}"
export HIVA_DURATION_FFN_ALPHA_INIT="${HIVA_DURATION_FFN_ALPHA_INIT:-0.1}"

exec bash "${SCRIPT_DIR}/finetune_bigflow_ckpt_20k_hiva_coeff_hp.sh" "$@"
