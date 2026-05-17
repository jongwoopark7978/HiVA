#!/usr/bin/env python
from __future__ import annotations

"""Convenience wrapper for canonical max-target HiVA coefficient sidecar generation.

Delegates to build_hiva_coeff_sidecar_hp.py with --target-mode max_target. If --fit-horizon is
larger than --dmax, the builder writes a canonical_lp_mt sidecar; otherwise it writes canonical_mt.
"""

import sys

from build_hiva_coeff_sidecar_hp import main


if __name__ == "__main__":
    if "--target-mode" not in sys.argv:
        sys.argv.extend(["--target-mode", "max_target"])
    main()
