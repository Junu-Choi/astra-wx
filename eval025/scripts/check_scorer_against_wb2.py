#!/usr/bin/env python3
"""Checks the 0.25 deg scorer against WeatherBench 2's metrics on the same forecast and truth arrays.

  python scripts/check_scorer_against_wb2.py [--init 2020-01-02T00] [--leads 24,120]
         [--vars z500,t850,t2m,msl,tp24h] [--members 56,50] [--group 2] [--tol 1e-5]

Forecast: GenCast's public 0.25 deg ensemble in WB2 (56 members),
  https://storage.googleapis.com/weatherbench2/datasets/gencast/2020-1440x721.zarr
Truth: WB2's 0.25 deg ERA5,
  https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr
Both are read anonymously over HTTPS.  Each (variable, lead) reads one GenCast chunk holding every member (about
0.4 GB compressed for a pressure-level variable, which holds 3 levels, and 0.1-0.2 GB for a single-level one); the
default sample reads about 2.5 GB.

For every (lead, variable, ensemble size) the members are scored twice, on identical float32 arrays:
  ours  astra_eval025.scorer.make_block_scorer, called as score_forecasts.py calls it (members as deviations from the
        verifying field, in blocks of --group members, cell-area weights of weights.area_weights): fair CRPS, MSE of
        the ensemble mean, ensemble variance (M - 1 in the denominator);
  WB2   weatherbench2.metrics CRPS, EnsembleMeanMSE and EnsembleVariance with ensemble_dim 'sample' and the global
        region, the metrics of WB2's probabilistic evaluation.
Both stores are put in latitude-ascending order and the forecast takes the ERA5 grid coordinates, as WB2's
evaluation does.  Prints one row per (lead, variable, M) and the largest relative difference per quantity; the exit
status is 1 if the CRPS or the ensemble-mean MSE differ by more than --tol (relative).

Needs weatherbench2 (pip install git+https://github.com/google-research/weatherbench2; only weatherbench2.metrics and
weatherbench2.regions are imported), xarray, zarr, fsspec with aiohttp, and jax.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astra_eval025 import scorer as S  # noqa: E402
from astra_eval025 import weights as W  # noqa: E402

GENCAST = "https://storage.googleapis.com/weatherbench2/datasets/gencast/2020-1440x721.zarr"
ERA5 = "https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"
VARS = {"z500": ("geopotential", 500), "z700": ("geopotential", 700), "z850": ("geopotential", 850),
        "t500": ("temperature", 500), "t700": ("temperature", 700), "t850": ("temperature", 850),
        "q500": ("specific_humidity", 500), "q700": ("specific_humidity", 700), "q850": ("specific_humidity", 850),
        "u500": ("u_component_of_wind", 500), "u700": ("u_component_of_wind", 700),
        "u850": ("u_component_of_wind", 850), "v500": ("v_component_of_wind", 500),
        "v700": ("v_component_of_wind", 700), "v850": ("v_component_of_wind", 850),
        "t2m": ("2m_temperature", None), "msl": ("mean_sea_level_pressure", None),
        "u10": ("10m_u_component_of_wind", None), "v10": ("10m_v_component_of_wind", None),
        "tp24h": ("total_precipitation_24hr", None)}


def _ascending(da):
    """lat / lon -> latitude / longitude, latitude increasing."""
    da = da.rename({a: b for a, b in (("lat", "latitude"), ("lon", "longitude")) if a in da.dims})
    return da.sortby("latitude")


def _read(fc, era5, t0, h, var, level):
    """(M, lat, lon) forecast members and the (lat, lon) verifying field, float32, latitude ascending, on the ERA5
    grid coordinates."""
    dt = np.timedelta64(int(h), "h")
    f = fc[var].sel(time=t0.astype("datetime64[ns]"), prediction_timedelta=dt.astype("timedelta64[ns]"))
    y = era5[var].sel(time=(t0 + dt).astype("datetime64[ns]"))
    if level is not None:
        f, y = f.sel(level=level), y.sel(level=level)
    f, y = _ascending(f).transpose("sample", "latitude", "longitude"), _ascending(y)
    for c in ("latitude", "longitude"):
        a, b = np.asarray(f[c].values, np.float64), np.asarray(y[c].values, np.float64)
        if a.shape != b.shape or not np.allclose(a, b, rtol=0.0, atol=1e-4):
            raise ValueError(f"{var}: forecast and ERA5 {c} differ")
    f = f.assign_coords(latitude=y["latitude"], longitude=y["longitude"])
    return f.astype(np.float32).load(), y.astype(np.float32).load()


def ours(x, y, group):
    """x (M, lat, lon), y (lat, lon) -> dict(crps, mse, var) from the block scorer, float64."""
    M, nlat, nlon = x.shape
    pts = lambda a: np.ascontiguousarray(np.swapaxes(a, -1, -2).reshape(a.shape[:-2] + (-1,)))  # noqa: E731
    xp, yp = pts(x)[..., None], pts(y)[:, None]                     # (M, P, 1), (P, 1): latitude varying fastest
    import jax.numpy as jnp
    cur = jnp.asarray(yp)
    views = [(lambda xb, e: xb - cur, np.zeros(1, np.float32), np.ones(1, np.float32))]
    score = S.make_block_scorer(M=M, P_points=nlat * nlon, weights=W.area_weights(nlat, nlon), views=views)
    nb = M // group
    blocks = [jnp.asarray(xp[k * group:(k + 1) * group]) for k in range(nb)]
    zero = [np.zeros((nlat * nlon, 1), np.float32)]
    mse, _, sp2, crps = (float(np.asarray(v, np.float64)[0]) for v in score(blocks.__getitem__, nb, zero, zero)[0])
    return dict(crps=crps, mse=mse, var=sp2)


def wb2(f, y, var):
    """The same quantities from weatherbench2.metrics on xarray Datasets."""
    from weatherbench2 import metrics
    from weatherbench2.regions import SliceRegion
    fc, obs = f.to_dataset(name=var), y.to_dataset(name=var)
    kw = dict(region=SliceRegion(), skipna=False)
    get = lambda m: float(np.asarray(m.compute_chunk(fc, obs, **kw)[var].values, np.float64))  # noqa: E731
    return dict(crps=get(metrics.CRPS(ensemble_dim="sample")), mse=get(metrics.EnsembleMeanMSE(ensemble_dim="sample")),
                var=get(metrics.EnsembleVariance(ensemble_dim="sample")))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="2020-01-02T00")
    ap.add_argument("--leads", default="24,120")
    ap.add_argument("--vars", default="z500,t850,t2m,msl,tp24h")
    ap.add_argument("--members", default="56,50", help="ensemble sizes to score: the first N members")
    ap.add_argument("--group", type=int, default=2)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--forecast", default=GENCAST)
    ap.add_argument("--era5", default=ERA5)
    a = ap.parse_args(argv)
    import xarray as xr
    fc = xr.open_zarr(a.forecast, decode_timedelta=True)
    era5 = xr.open_zarr(a.era5)
    t0 = np.datetime64(a.init, "h")
    sizes = [int(m) for m in a.members.split(",")]
    for m in sizes:
        if not 2 <= m <= fc.sizes["sample"] or m % a.group:
            raise SystemExit(f"--members {m}: needs 2 <= M <= {fc.sizes['sample']} and divisible by --group")
    rows = []
    print(f"{'lead h':>6} {'var':>6} {'M':>3} | {'CRPS ours':>13} {'CRPS WB2':>13} {'rel':>9} | "
          f"{'MSE ours':>13} {'MSE WB2':>13} {'rel':>9} | {'var ours':>13} {'var WB2':>13} {'rel':>9}", flush=True)
    for h in (int(s) for s in a.leads.split(",")):
        for name in a.vars.split(","):
            var, level = VARS[name]
            f, y = _read(fc, era5, t0, h, var, level)
            for m in sizes:
                fm = f.isel(sample=slice(0, m))
                o, w = ours(fm.values, y.values, a.group), wb2(fm, y, var)
                rel = {k: (o[k] - w[k]) / abs(w[k]) for k in o}
                rows.append(rel)
                print(f"{h:6d} {name:>6} {m:3d} | {o['crps']:13.7g} {w['crps']:13.7g} {rel['crps']:+9.1e} | "
                      f"{o['mse']:13.7g} {w['mse']:13.7g} {rel['mse']:+9.1e} | "
                      f"{o['var']:13.7g} {w['var']:13.7g} {rel['var']:+9.1e}", flush=True)
    worst = {k: max(abs(r[k]) for r in rows) for k in ("crps", "mse", "var")}
    print("largest relative difference: " + ", ".join(f"{k} {v:.2e}" for k, v in worst.items())
          + f" ({len(rows)} rows)")
    ok = worst["crps"] <= a.tol and worst["mse"] <= a.tol
    print("PASS" if ok else f"FAIL: CRPS or ensemble-mean MSE differ by more than {a.tol:g}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
