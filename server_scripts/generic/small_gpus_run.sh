#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPUS="${1:-0,1,2,3}"
SECS="${2:-60}"
LOG="${3:-small_gpus_$(date +%Y%m%d_%H%M%S).log}"

CUDA_VISIBLE_DEVICES="$GPUS" python small_gpus.py "$GPUS" "$SECS" 2>&1 | tee "$LOG"