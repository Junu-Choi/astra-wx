#!/usr/bin/env python3
"""Scorecard of a per-initialisation score file against the WB2 result files of GenCast and IFS ENS (2020, 0.25 deg).

  python scripts/make_scorecard.py SCORES.npz --wb2-dir DIR --out OUT [--ens-states FILE_OR_DIR]
         [--boot 10000] [--block 14] [--block-alt N] [--seed 20201231] [--allow-partial] [--label NAME]

--wb2-dir     gencast_vs_era5_1440x721_2020.nc and ens_vs_era5_1440x721_2020.nc (WB2 benchmark_results)
--ens-states  per-initialisation IFS ENS 24 h scores (npz, or a directory of WeatherBench-X state_<init>.nc);
              without it every test is an unwidened fixed-reference test (no kappa; significant =
              significant_unwidened)
--block-alt   also the intervals for blocks of N initialisations (ci_alt, p_alt of the tested cells)
--allow-partial  accept a score file whose init_times are not exactly the 732 initialisations of 2020
--label       model name written into the outputs (default Astra)
Writes OUT/scorecard.json (every comparison with its interval and flags) and OUT/astra_vs_era5_1440x721_2020.nc.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astra_eval025 import scorecard as SC  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scores")
    ap.add_argument("--wb2-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ens-states", default="")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--block", type=int, default=14)
    ap.add_argument("--block-alt", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20201231)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--label", default="Astra")
    a = ap.parse_args(argv)
    SC.run(a.scores, a.out, refs_dir=a.wb2_dir, ens_states=a.ens_states or None, boot=a.boot, block=a.block,
           block_alt=a.block_alt, seed=a.seed, allow_partial=a.allow_partial, label=a.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
