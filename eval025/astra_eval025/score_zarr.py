"""Per-initialisation scores of a 0.25 deg ensemble forecast against ERA5, from xarray datasets in the WeatherBench 2
layout, written as a score file for scorecard.run.

Forecast dataset: variables with WB2 names, dimensions time (initialisation), an ensemble dimension,
prediction_timedelta, [level,] latitude, longitude (any order; lat / lon are renamed to latitude / longitude);
6-hourly `total_precipitation_6hr` (the four 6 h accumulations ending at a lead time give tp24h).  ERA5 dataset: the
WB2 0.25 deg ERA5 store (6-hourly, with `total_precipitation_6hr`).  Grid: 721 x 1440 equiangular with poles; a
forecast latitude running in the opposite order to ERA5's is flipped to ERA5's order.

Members are scored by scorer.make_block_scorer as deviations from the verifying field (x - y, in physical units, or
standardised with the mean_std argument), in blocks of `group` members, so that its float32 sums stay accurate for
fields with large means such as geopotential and mean sea-level pressure.
"""
from __future__ import annotations

import os

import numpy as np

from . import scorer as S
from . import variables as WV
from . import weights as W

CORE = ([(w, lv, f"{p}{lv}") for p, w in (("z", "geopotential"), ("t", "temperature"), ("q", "specific_humidity"),
                                          ("u", "u_component_of_wind"), ("v", "v_component_of_wind"))
         for lv in (500, 700, 850)]
        + [("2m_temperature", None, "t2m"), ("mean_sea_level_pressure", None, "msl"),
           ("10m_u_component_of_wind", None, "u10"), ("10m_v_component_of_wind", None, "v10")])
NAMES = [c[2] for c in CORE] + list(S.DERIVED)  # the columns of score_init
FILE_NAMES = [v.ours for v in WV.VARS]  # the rows of the score file, in the order of the stored score files
assert sorted(FILE_NAMES) == sorted(NAMES)
TP6 = "total_precipitation_6hr"


def _points(da):
    """(..., latitude, longitude) DataArray -> (..., P) float32, P = lon * lat with latitude varying fastest."""
    other = [d for d in da.dims if d not in ("longitude", "latitude")]
    a = da.transpose(*other, "longitude", "latitude").values
    return np.ascontiguousarray(a.reshape(a.shape[:len(other)] + (-1,)), np.float32)


def _field(ds, var, level, **sel):
    da = ds[var].sel(**sel)
    if level is not None:
        da = da.sel(level=level)
    return _points(da)


def _align(fc, era5):
    """lat / lon -> latitude / longitude; the forecast latitude flipped to ERA5's order if it runs the other way."""
    fc, era5 = (ds.rename({a: b for a, b in (("lat", "latitude"), ("lon", "longitude")) if a in ds.dims})
                for ds in (fc, era5))
    a, b = np.asarray(fc["latitude"].values, np.float64), np.asarray(era5["latitude"].values, np.float64)
    if (a.shape == b.shape and not np.allclose(a, b, rtol=0.0, atol=1e-6)
            and np.allclose(a[::-1], b, rtol=0.0, atol=1e-6)):
        fc = fc.isel(latitude=slice(None, None, -1))
    return fc, era5


def _jitted(M, nlat, nlon, mean_std, compiler_options):
    """Scorer objects independent of the initialisation: mean, std, wind index, `cur` (the verifying fields of the
    current lead, read by the views) and the block scorer."""
    Q = len(CORE)
    mean, std = (np.zeros(Q, np.float32), np.ones(Q, np.float32)) if mean_std is None else mean_std
    wsi = S.wind_index([c[2] for c in CORE])
    md = S.make_members_derived(wsi, mean, std, compiler_options=compiler_options)
    cur = {}  # verifying fields of the current lead: y (standardised as the members), yd (derived, physical)
    views = [(lambda x, e: x - cur["y"], np.zeros(Q, np.float32), std),
             (lambda x, e: md(x, e) - cur["yd"], np.zeros(len(S.DERIVED), np.float32),
              np.ones(len(S.DERIVED), np.float32))]
    w = W.area_weights(nlat, nlon)
    score = S.make_block_scorer(M=M, P_points=nlat * nlon, weights=w, views=views, compiler_options=compiler_options)
    return mean, std, wsi, cur, score


def score_init(fc, era5, t0, leads_h, *, ens_dim="number", group=2, mean_std=None, compiler_options=None, tp24h=True,
               jitted=None):
    """One initialisation -> dict(mse, sp2, crps) of shape (L, len(NAMES)), float64; tp24h NaN before 24 h, and at
    every lead with tp24h=False (total_precipitation_6hr is then not read).  jitted: the objects of _jitted, shared
    by score_inits across initialisations."""
    import jax.numpy as jnp
    fc, era5 = _align(fc, era5)
    for c in ("latitude", "longitude"):
        a, b = np.asarray(fc[c].values, np.float64), np.asarray(era5[c].values, np.float64)
        if a.shape != b.shape or not np.allclose(a, b, rtol=0.0, atol=1e-6):
            raise ValueError(f"forecast {c} ({a.size} values) differs from the ERA5 {c} ({b.size} values)")
    lat = np.asarray(era5["latitude"].values, np.float64)
    grid = np.linspace(-90.0, 90.0, lat.size)
    if not (np.allclose(lat, grid, rtol=0.0, atol=1e-6) or np.allclose(lat, grid[::-1], rtol=0.0, atol=1e-6)):
        raise ValueError(f"ERA5 latitude ({lat.size} values) is not an equiangular grid with poles "
                         f"(linspace(-90, 90, {lat.size}) in either order)")
    t0 = np.datetime64(t0, "h")
    nlat, nlon = era5.sizes["latitude"], era5.sizes["longitude"]
    P = nlat * nlon
    Q = len(CORE)
    M = fc.sizes[ens_dim]
    if M % group:
        raise ValueError(f"{M} members are not divisible into groups of {group}")
    nb = M // group
    if jitted is None:
        jitted = _jitted(M, nlat, nlon, mean_std, compiler_options)
    mean, std, wsi, cur, score = jitted
    fsel = dict(time=t0.astype("datetime64[ns]"))
    out = {k: np.full((len(leads_h), len(NAMES)), np.nan) for k in ("mse", "sp2", "crps")}
    for li, h in enumerate(leads_h):
        dt = np.timedelta64(int(h), "h")
        vt = (t0 + dt).astype("datetime64[ns]")
        truth = np.stack([_field(era5, v, lv, time=vt) for v, lv, _ in CORE], -1)
        ok_tp = tp24h and h >= 24
        tp_t = (S.tp24h_truth([_field(era5, TP6, None, time=(t0 + dt - np.timedelta64(6 * k, "h"))
                                      .astype("datetime64[ns]")) for k in (3, 2, 1, 0)]) if ok_tp else None)
        td = np.asarray(S.truth_derived(truth, tp_t, wsi, P))
        cur["y"] = jnp.asarray((truth - mean) / std if mean_std is not None else truth)
        cur["yd"] = jnp.asarray(td)
        lead = fc.sel(**fsel, prediction_timedelta=dt.astype("timedelta64[ns]"))
        x = np.stack([_field(lead, v, lv) for v, lv, _ in CORE], -1)  # (M, P, Q), read once per lead
        x = (x - mean) / std if mean_std is not None else x
        blocks = [jnp.asarray(x[k * group:(k + 1) * group]) for k in range(nb)]
        del x
        ext = []
        for k in range(nb):
            sl = {ens_dim: slice(k * group, (k + 1) * group)}
            if ok_tp:
                r = [_field(fc.isel(**sl), TP6, None, **fsel,
                            prediction_timedelta=(dt - np.timedelta64(6 * j, "h")).astype("timedelta64[ns]"))
                     for j in (3, 2, 1, 0)]
                ext.append(jnp.asarray(S.tp24h_members(r)))
            else:
                ext.append(jnp.asarray(np.zeros((group, P), np.float32)))
        zero = [np.zeros((P, Q), np.float32), np.zeros((P, len(S.DERIVED)), np.float32)]
        res = score(blocks.__getitem__, nb, zero, zero, extra=ext)
        (m_, _, s_, c_), (dm, _, ds, dc) = [[np.asarray(v, np.float64) for v in r] for r in res]
        if not ok_tp:
            dm[-1] = ds[-1] = dc[-1] = np.nan
        out["mse"][li] = np.concatenate([m_, dm])
        out["sp2"][li] = np.concatenate([s_, ds])
        out["crps"][li] = np.concatenate([c_, dc])
    return out


def score_inits(fc, era5, init_times, leads_h, path, **kw):
    """Score every initialisation and write the score file (scorecard.load_scores format, rows in FILE_NAMES order);
    returns the path written (.npz appended as by np.savez)."""
    ens_dim = kw.get("ens_dim", "number")
    fc, era5 = _align(fc, era5)
    jitted = _jitted(fc.sizes[ens_dim], era5.sizes["latitude"], era5.sizes["longitude"], kw.get("mean_std"),
                     kw.get("compiler_options"))
    rows = [score_init(fc, era5, t, leads_h, jitted=jitted, **kw) for t in init_times]
    t = (np.asarray(init_times, "datetime64[h]") - np.datetime64("1970-01-01T00", "h")).astype(np.int64)
    o = np.argsort(t)
    path = os.fspath(path)
    if not path.endswith(".npz"):
        path += ".npz"
    ci = [NAMES.index(n) for n in FILE_NAMES]
    np.savez(path, version=np.asarray(1), names=np.asarray(FILE_NAMES), init_times=t[o],
             leads_hours=np.asarray(leads_h, np.int64), M=np.asarray(int(fc.sizes[ens_dim])),
             **{k: np.stack([r[k] for r in rows])[o][..., ci] for k in ("crps", "mse", "sp2")})
    return path
