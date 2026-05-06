#!/usr/bin/env bash
set -euo pipefail

# Clean-plus-noisy duration-query ablation for HiVA-lite / duration-SmolVLA.
#
# Objective:
#   L_total = L_FM-action
#           + lambda_dur-clean * L_dur-clean
#           + lambda_dur-noisy * L_dur-noisy
#
# Settings from docs/notes/HiVA_v1_pilot.tex, B.1 Experiment 1:
#   sigma = 0.25
#   lambda_dur-clean = 0.05
#   lambda_dur-noisy = 0.01
#
# The base bigflow script keeps the single-prefix VLM cache logic enabled. It
# runs one noisy action-expert pass for the flow-matching loss and noisy duration
# loss, plus one clean action-expert pass for the teacher-forced duration loss.
#
# Example:
#   GPU_IDS=0,1,2,3 NUM_GPUS=4 BATCH_PER_GPU=96 S=2 \
#   WANDB_ENABLE=true WANDB_PROJECT=lerobot \
#   bash server_scripts/bigflow/finetune_bigflow_ckpt_20k_d_clean_noisy.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/server_scripts/common_wandb.sh"
build_run_id

export DURATION_LOSS_WEIGHT="${DURATION_LOSS_WEIGHT:-0.05}"
export DURATION_NOISY_LOSS_WEIGHT="${DURATION_NOISY_LOSS_WEIGHT:-0.01}"
export DURATION_NOISY_SIGMA="${DURATION_NOISY_SIGMA:-0.25}"
export RUN_NAME="${RUN_NAME:-smolvla_hiva_duration_clean_noisy_bigflow_s${S:-2}_${RUN_ID}}"

exec bash "${SCRIPT_DIR}/finetune_bigflow_ckpt_20k_d.sh"

# bs, mem
# 96, 41
