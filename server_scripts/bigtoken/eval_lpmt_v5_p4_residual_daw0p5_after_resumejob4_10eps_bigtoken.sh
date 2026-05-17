#!/usr/bin/env bash
set -euo pipefail

# Queue v5 p4 residual DAW0.5 LP-MT eval after the current resumejob4 eval.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

WAIT_PATTERN="${WAIT_PATTERN:-eval_lpmt_v4_d2_4_10_resumejob4|job1_lpmt_v4_d2_4_10_coeffpool_resumejob4}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

echo "===== queued v5 p4 residual DAW0.5 eval at $(date) ====="
echo "WAIT_PATTERN=${WAIT_PATTERN}"
echo "TIMESTAMP=${TIMESTAMP}"

while pgrep -f "${WAIT_PATTERN}" >/dev/null; do
  echo "Waiting for resumejob4 eval to finish at $(date)"
  pgrep -af "${WAIT_PATTERN}" || true
  sleep 300
done

echo "resumejob4 eval is clear at $(date); starting v5 p4 residual DAW0.5 eval"

TIMESTAMP="${TIMESTAMP}_after_resumejob4" \
GPU_IDS="${GPU_IDS:-4,5,6,7}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}" \
bash "${SCRIPT_DIR}/eval_lpmt_v5_p4_residual_daw0p5_10eps_bigtoken.sh"

echo "===== queued v5 p4 residual DAW0.5 eval finished at $(date) ====="
