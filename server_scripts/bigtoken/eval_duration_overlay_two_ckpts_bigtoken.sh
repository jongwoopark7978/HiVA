#!/usr/bin/env bash
set -euo pipefail

# Run duration-overlay video generation for the two requested checkpoints:
#   1. bigcornea S=8 / step-reduced checkpoint
#   2. bigflow full-step checkpoint

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export TIMESTAMP

STEP8_POLICY_PATH="${STEP8_POLICY_PATH:-/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_duration_token_bigcornea_full_s8_20260430_152419/checkpoints/last/pretrained_model}"
FULLSTEP_POLICY_PATH="${FULLSTEP_POLICY_PATH:-/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_duration_token_bigflow_full_b32_20260429_rerun/checkpoints/last/pretrained_model}"

echo "Starting bigtoken duration-overlay video sweep at ${TIMESTAMP}"

POLICY_PATH="${STEP8_POLICY_PATH}" \
CHECKPOINT_LABEL="${STEP8_LABEL:-bigcornea_s8_step_reduced}" \
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR_STEP8:-/home/jongwoopark/lerobot/outputs/eval/duration_overlay_bigtoken_bigcornea_s8_step_reduced_${TIMESTAMP}}" \
bash "${SCRIPT_DIR}/eval_duration_overlay_bigtoken.sh"

POLICY_PATH="${FULLSTEP_POLICY_PATH}" \
CHECKPOINT_LABEL="${FULLSTEP_LABEL:-bigflow_fullstep_b32_rerun}" \
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR_FULLSTEP:-/home/jongwoopark/lerobot/outputs/eval/duration_overlay_bigtoken_bigflow_fullstep_b32_rerun_${TIMESTAMP}}" \
bash "${SCRIPT_DIR}/eval_duration_overlay_bigtoken.sh"

echo "Finished bigtoken duration-overlay video sweep at $(date +%Y%m%d_%H%M%S)"
