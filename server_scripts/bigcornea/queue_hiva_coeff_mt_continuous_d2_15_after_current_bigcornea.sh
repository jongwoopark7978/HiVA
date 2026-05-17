#!/usr/bin/env bash
set -euo pipefail

# Queue the d2/15 continuous-duration MT HiVA sweep until another tmux session
# exits. This keeps the current GPU job untouched and starts the next sweep only
# after that session finishes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

WAIT_TMUX_SESSION="${WAIT_TMUX_SESSION:-hiva_mt_job2_clean_20260506_233112}"
SWEEP_TS="${SWEEP_TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"
OUTER_LOG="${OUTER_LOG:-${LOG_DIR}/hiva_coeff_mt_continuous_d2_15_bigcornea_${SWEEP_TS}.outer.log}"

echo "Queue created at $(date)"
echo "Waiting for tmux session: ${WAIT_TMUX_SESSION}"
echo "Queued sweep timestamp: ${SWEEP_TS}"
echo "Queued sweep outer log: ${OUTER_LOG}"

while tmux has-session -t "${WAIT_TMUX_SESSION}" 2>/dev/null; do
  echo "$(date): ${WAIT_TMUX_SESSION} still running; sleeping 300s"
  sleep 300
done

echo "$(date): ${WAIT_TMUX_SESSION} finished; starting continuous d2/15 sweep"
SWEEP_TS="${SWEEP_TS}" \
OUTER_LOG="${OUTER_LOG}" \
bash "${SCRIPT_DIR}/run_hiva_coeff_mt_continuous_d2_15_sweep_bigcornea.sh"
