#!/usr/bin/env bash
set -euo pipefail

# LP-MT HiVA coefficient-pooling duration readout run on bigflow.
#
# User-facing shorthand:
#   hiva_duration_head_type=coeffpool
#
# Code-facing setting:
#   HIVA_DURATION_READOUT=coeff_modality_pool
#
# The coeff_modality_pool readout removes the separate duration suffix token and predicts
# categorical duration from pooled coefficient-token features with the residual FFN head.
#
# Example:
#   GPU_IDS=0,2,3 NUM_GPUS=3 BATCH_PER_GPU=160 WANDB_ENABLE=true \
#     bash server_scripts/bigflow/run_hiva_coeff_lpmt_d2_10_coeffpool_bigflow.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GPU_IDS="${GPU_IDS:-0,2,3}"
export NUM_GPUS="${NUM_GPUS:-3}"
export NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
export BATCH_PER_GPU="${BATCH_PER_GPU:-160}"
export S="${S:-2}"
export WANDB_ENABLE="${WANDB_ENABLE:-true}"
export WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"

export HIVA_DURATION_CLASSES="${HIVA_DURATION_CLASSES:-[2,10]}"
export HIVA_DMAX="${HIVA_DMAX:-10}"
export HIVA_FIT_HORIZON="${HIVA_FIT_HORIZON:-15}"
export HIVA_K="${HIVA_K:-10}"
export HIVA_BASIS_MODE="${HIVA_BASIS_MODE:-canonical_lp_mt}"
export HIVA_DURATION_LOSS="${HIVA_DURATION_LOSS:-ce_mean}"
export HIVA_SUFFIX_ATTENTION="${HIVA_SUFFIX_ATTENTION:-full}"
export HIVA_DURATION_READOUT="${HIVA_DURATION_READOUT:-coeff_modality_pool}"
export HIVA_DURATION_HEAD_TYPE="${HIVA_DURATION_HEAD_TYPE:-residual_ffn}"
export POLICY_CHUNK_SIZE="${POLICY_CHUNK_SIZE:-${HIVA_FIT_HORIZON}}"
export POLICY_N_ACTION_STEPS="${POLICY_N_ACTION_STEPS:-${HIVA_DMAX}}"

export SIDECAR="${SIDECAR:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d2_10_w1_10_w3_0_k10_f15_canonical_lp_mt.parquet}"
export SIDECAR_SUMMARY="${SIDECAR_SUMMARY:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d2_10_w1_10_w3_0_k10_f15_canonical_lp_mt.summary.json}"

export RUN_NAME="${RUN_NAME:-smolvla_hiva_coeff_lpmt_d2_10_coeffpool_full_ce_mean_k10_f15_bigflow_b${BATCH_PER_GPU}_s${S}_$(date +%Y%m%d_%H%M%S)}"

exec bash "${SCRIPT_DIR}/finetune_bigflow_ckpt_20k_hiva_coeff_lp_mt.sh"
