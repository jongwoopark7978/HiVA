#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible alias for notes/scripts that used the older b64 filename.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/finetune_bigcornea_ckpt_20k_hiva_coeff_cleaner_suffix.sh" "$@"
