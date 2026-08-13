#!/usr/bin/env python3
"""Tutorial part 4 — rolling ball, stove knob, mallet, multi-stage force key."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _run import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(part=4))
