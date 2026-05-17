#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper for canonical MT/LP-MT coefficient sidecar generation on bigflow.
# By default this delegates to the LP-MT builder with {2,15}, executable Dmax=15,
# fit horizon=20, and K=10. Override HIVA_FIT_HORIZON=15 to build canonical MT.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/build_hiva_coeff_lp_mt_sidecar_bigflow.sh"
