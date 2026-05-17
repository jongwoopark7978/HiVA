#!/usr/bin/env bash
set -euo pipefail

# Wait for the residual-flow decoded-weight tmux queue, then launch stage-0 LP-MT
# HiVA v5 with S=0.5 and milestone checkpoints.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

WAIT_FOR_SESSION="${WAIT_FOR_SESSION:-hiva_residual_flow_daw_sweep_g0_7}"
WAIT_INTERVAL_S="${WAIT_INTERVAL_S:-300}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS}}"
BATCH_PER_GPU="${BATCH_PER_GPU:-64}"
S="${S:-0.5}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"

echo "Stage-0 LP-MT HiVA after-current queue"
echo "WAIT_FOR_SESSION=${WAIT_FOR_SESSION}"
echo "WAIT_INTERVAL_S=${WAIT_INTERVAL_S}"
echo "GPU_IDS=${GPU_IDS} NUM_GPUS=${NUM_GPUS} BATCH_PER_GPU=${BATCH_PER_GPU} S=${S}"

while tmux has-session -t "${WAIT_FOR_SESSION}" 2>/dev/null; do
  echo "$(date): waiting for tmux session ${WAIT_FOR_SESSION} to finish..."
  sleep "${WAIT_INTERVAL_S}"
done

while pgrep -af 'residual_flow_stage1_bigcornea|residual_flow_stage1_decoded|lerobot-train' | rg -q 'residual_flow_stage1'; do
  echo "$(date): residual-flow worker processes still present; waiting..."
  sleep 60
done

echo "$(date): dependency queue is finished; launching stage0 LP-MT HiVA."
GPU_IDS="${GPU_IDS}" \
NUM_GPUS="${NUM_GPUS}" \
NUM_PROCESSES="${NUM_PROCESSES}" \
BATCH_PER_GPU="${BATCH_PER_GPU}" \
S="${S}" \
WANDB_ENABLE="${WANDB_ENABLE}" \
WANDB_PROJECT="${WANDB_PROJECT}" \
bash "${SCRIPT_DIR}/run_hiva_coeff_lpmt_stage0_v5_s0p5_milestone_bigcornea.sh"
