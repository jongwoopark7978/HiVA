#!/usr/bin/env bash
set -euo pipefail

# Queue the P7 stage-0 LP-MT run after the current GPU4-7 residual-flow tmux
# queue finishes. This script itself is intended to run inside tmux and does
# not reserve GPUs while waiting.

REPO_ROOT="${REPO_ROOT:-/home/jongwoopark/lerobot}"
cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/queue_hiva_coeff_lpmt_stage0_p7_s0p5_after_residual_flow_bigflow_${RUN_STAMP}.outer.log}"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

WAIT_TMUX_SESSION="${WAIT_TMUX_SESSION:-hiva_rf_scale_sweep_20260514_040613}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
GPU_MEM_FREE_THRESHOLD_MB="${GPU_MEM_FREE_THRESHOLD_MB:-2000}"
POLL_SECONDS="${POLL_SECONDS:-300}"
RUN_SCRIPT="${RUN_SCRIPT:-${REPO_ROOT}/server_scripts/bigflow/run_hiva_coeff_lpmt_stage0_p7_s0p5_5000_bigflow.sh}"

echo "Queued P7 stage-0 LP-MT Bigflow run"
echo "QUEUE_LOG=${QUEUE_LOG}"
echo "WAIT_TMUX_SESSION=${WAIT_TMUX_SESSION}"
echo "GPU_IDS=${GPU_IDS}"
echo "GPU_MEM_FREE_THRESHOLD_MB=${GPU_MEM_FREE_THRESHOLD_MB}"
echo "POLL_SECONDS=${POLL_SECONDS}"
echo "RUN_SCRIPT=${RUN_SCRIPT}"
echo "Queued at $(date)"

while tmux has-session -t "${WAIT_TMUX_SESSION}" 2>/dev/null; do
  echo "[$(date)] Waiting for tmux session ${WAIT_TMUX_SESSION} to finish..."
  sleep "${POLL_SECONDS}"
done

while true; do
  busy="$(GPU_IDS="${GPU_IDS}" GPU_MEM_FREE_THRESHOLD_MB="${GPU_MEM_FREE_THRESHOLD_MB}" python - <<'PY'
import os, subprocess
ids = [int(x) for x in os.environ["GPU_IDS"].split(",") if x.strip()]
threshold = int(os.environ["GPU_MEM_FREE_THRESHOLD_MB"])
out = subprocess.check_output([
    "nvidia-smi",
    "--query-gpu=index,memory.used",
    "--format=csv,noheader,nounits",
], text=True)
busy = []
for line in out.strip().splitlines():
    idx_s, mem_s = [p.strip() for p in line.split(",")[:2]]
    idx, mem = int(idx_s), int(mem_s)
    if idx in ids and mem > threshold:
        busy.append(f"{idx}:{mem}MiB")
print(" ".join(busy))
PY
)"
  if [[ -z "${busy}" ]]; then
    break
  fi
  echo "[$(date)] Waiting for GPU memory to clear on ${GPU_IDS}; busy=${busy}"
  sleep "${POLL_SECONDS}"
done

echo "[$(date)] Dependencies cleared. Launching P7 stage-0 run."
exec env \
  GPU_IDS="${GPU_IDS}" \
  NUM_GPUS=4 \
  NUM_PROCESSES=4 \
  BATCH_PER_GPU=128 \
  BATCH_SIZE=128 \
  S=0.5 \
  "${RUN_SCRIPT}"
