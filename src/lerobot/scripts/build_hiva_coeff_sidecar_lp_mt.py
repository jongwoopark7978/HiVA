#!/usr/bin/env python
from __future__ import annotations

"""Convenience wrapper for LP-MT coefficient sidecar generation.

This selects max_target mode and leaves --fit-horizon explicit in the launch scripts.  LP-MT is
canonical_mt with a longer fitting/preview horizon than the executable dmax.
"""

import sys

from build_hiva_coeff_sidecar_hp import main


if __name__ == "__main__":
    if "--target-mode" not in sys.argv:
        sys.argv.extend(["--target-mode", "max_target"])
    main()
