#!/usr/bin/env python3
"""Source-checkout wrapper for the installed asset provisioning command."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from luxai.s2s_magpie.provision import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
