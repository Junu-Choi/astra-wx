#!/usr/bin/env python3
"""Recompute the paper's 0.25 deg score-based numbers from data/ and print paper value vs reproduced value.

  python scripts/reproduce_paper_025.py [--data DIR] [--json OUT.json]
Exit status 0 when every number matches.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astra_eval025 import reproduce  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(reproduce.main())
