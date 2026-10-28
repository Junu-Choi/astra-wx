#!/usr/bin/env python3
"""Per-initialisation 0.25 deg scores of an ensemble forecast store against WB2's ERA5 -> score file.

  python scripts/score_forecasts.py --forecast FC.zarr --era5 ERA5.zarr --out SCORES.npz
         [--ens-dim number] [--inits 2020] [--leads HOURS] [--group 2] [--members N] [--no-tp24h]

--forecast  WB2-layout forecast store on the 0.25 deg 1440x721 grid (variables of astra_eval025.score_zarr.CORE and
            total_precipitation_6hr every 6 h; latitude in either order, lat / lon accepted)
--era5      e.g. https://storage.googleapis.com/weatherbench2/datasets/era5/
            1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr
            (public, no credentials; gs:// paths need Google Cloud credentials)
--inits     '2020' = the 732 initialisations of 2020 at 00 and 12 UTC, or a comma-separated list
--leads     lead times in hours, comma-separated, e.g. 24,120,240 (default: 12 to 360 every 12 h)
--group     members per float32 scoring block; must divide the number of members (default 2)
--members   score the first N members of the ensemble dimension
--no-tp24h  leave the tp24h row NaN (total_precipitation_6hr is not read)
Needs xarray, zarr (and gcsfs with aiohttp for remote stores) and jax.  Then: scripts/make_scorecard.py SCORES.npz ...
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astra_eval025 import score_zarr as Z  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecast", required=True)
    ap.add_argument("--era5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ens-dim", default="number")
    ap.add_argument("--inits", default="2020")
    ap.add_argument("--leads", default=",".join(str(h) for h in range(12, 361, 12)))
    ap.add_argument("--group", type=int, default=2)
    ap.add_argument("--members", type=int, default=None)
    ap.add_argument("--no-tp24h", action="store_true")
    a = ap.parse_args(argv)
    import xarray as xr
    fc = xr.open_zarr(a.forecast)
    era5 = xr.open_zarr(a.era5)
    if a.members is not None:
        if not 1 <= a.members <= fc.sizes[a.ens_dim]:
            raise SystemExit(f"--members {a.members}: the store holds {fc.sizes[a.ens_dim]} members")
        fc = fc.isel({a.ens_dim: slice(0, a.members)})
    if a.inits == "2020":
        inits = np.arange(np.datetime64("2020-01-01T00", "h"), np.datetime64("2021-01-01T00", "h"),
                          np.timedelta64(12, "h"))
    else:
        inits = np.asarray([np.datetime64(s, "h") for s in a.inits.split(",")])
    leads = [int(h) for h in a.leads.split(",")]
    out = Z.score_inits(fc, era5, inits, leads, a.out, ens_dim=a.ens_dim, group=a.group, tp24h=not a.no_tp24h)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
