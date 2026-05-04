#!/usr/bin/env bash
set -euo pipefail

# Generate LIBERO overlay videos for the vanilla SmolVLA checkpoint without a
# duration token. The shared bigtoken overlay script still displays D, INF, MD,
# and final SC/FA; for vanilla SmolVLA, D is the fixed execution horizon.
#
# Example:
#   GPU_IDS=0,1,2,3 \
#   N_EPISODES=2 \
#   bash server_scripts/bigtoken/eval_smolvla_vanilla_overlay_bigtoken.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

export POLICY_PATH="${POLICY_PATH:-/home/jongwoopark/lerobot/outputs/train/smolvla_libero_from_official_best_avg77.5_20260425_004040/checkpoints/last/pretrained_model}"
export CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-smolvla_official_best_avg77_5_vanilla}"
export USE_DURATION_HEAD="${USE_DURATION_HEAD:-false}"
export N_ACTION_STEPS="${N_ACTION_STEPS:-1}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export N_EPISODES="${N_EPISODES:-2}"
export BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/home/jongwoopark/lerobot/outputs/eval/vanilla_overlay_bigtoken_${CHECKPOINT_LABEL}_${TIMESTAMP}}"

exec bash "${SCRIPT_DIR}/eval_duration_overlay_bigtoken.sh"
