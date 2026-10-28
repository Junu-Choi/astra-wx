"""Front end for WB2-layout datasets: grid flattening, lead alignment, derived variables, grid order and names, the
tp24h and members options (synthetic data)."""
import importlib.util

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from astra_eval025 import score_zarr as Z  # noqa: E402
from astra_eval025 import scorecard as SC  # noqa: E402
from astra_eval025 import scorer as S  # noqa: E402
from astra_eval025 import variables as WV  # noqa: E402
from astra_eval025 import weights as W  # noqa: E402
from conftest import ROOT  # noqa: E402


SCALE = {"geopotential": 50.0, "temperature": 2.0, "specific_humidity": 1e-3, "u_component_of_wind": 3.0,
         "v_component_of_wind": 3.0, "2m_temperature": 2.0, "mean_sea_level_pressure": 80.0,
         "10m_u_component_of_wind": 2.0, "10m_v_component_of_wind": 2.0}
OFFSET = {"geopotential": 54000.0, "temperature": 260.0, "2m_temperature": 288.0, "mean_sea_level_pressure": 101000.0}


def _datasets(M=4, nlat=9, nlon=12, seed=0):
    rng = np.random.default_rng(seed)
    lat, lon = np.linspace(-90, 90, nlat), np.arange(nlon) * (360 / nlon)
    lev = np.array([500, 700, 850])
    t0 = np.datetime64("2020-01-02T00", "ns")
    et = t0 + np.arange(-4, 5) * np.timedelta64(6, "h")
    pt = np.arange(1, 5) * np.timedelta64(6, "h").astype("timedelta64[ns]")
    e, f = {}, {}
    for v, s in SCALE.items():
        o = OFFSET.get(v, 0.0)
        if v in ("geopotential", "temperature", "specific_humidity", "u_component_of_wind", "v_component_of_wind"):
            e[v] = (("time", "level", "latitude", "longitude"), o + rng.normal(size=(len(et), 3, nlat, nlon)) * s)
            f[v] = (("time", "number", "prediction_timedelta", "level", "longitude", "latitude"),
                    o + rng.normal(size=(1, M, len(pt), 3, nlon, nlat)) * s)
        else:
            e[v] = (("time", "latitude", "longitude"), o + rng.normal(size=(len(et), nlat, nlon)) * s)
            f[v] = (("time", "number", "prediction_timedelta", "longitude", "latitude"),
                    o + rng.normal(size=(1, M, len(pt), nlon, nlat)) * s)
    e[Z.TP6] = (("time", "latitude", "longitude"), rng.random((len(et), nlat, nlon)) * 1e-3)
    f[Z.TP6] = (("time", "number", "prediction_timedelta", "longitude", "latitude"),
                rng.random((1, M, len(pt), nlon, nlat)) * 1e-3)
    era5 = xr.Dataset(e, coords=dict(time=et, level=lev, latitude=lat, longitude=lon))
    fc = xr.Dataset(f, coords=dict(time=[t0], number=np.arange(M), prediction_timedelta=pt, level=lev,
                                   latitude=lat, longitude=lon))
    return fc.astype(np.float32), era5.astype(np.float32), t0


@pytest.mark.parametrize("standardise", [False, True])
def test_score_zarr_matches_float64_reference(tmp_path, standardise):
    fc, era5, t0 = _datasets()
    kw = {}
    if standardise:
        kw["mean_std"] = (np.asarray([OFFSET.get(v, 0.0) for v, _, _ in Z.CORE], np.float32),
                          np.asarray([2.0 * SCALE[v] for v, _, _ in Z.CORE], np.float32))
    p = Z.score_inits(fc, era5, [t0], [12, 24], tmp_path / "s.npz", group=2, **kw)
    D = SC.load_scores(p)
    nlat, nlon = era5.sizes["latitude"], era5.sizes["longitude"]
    w = W.area_weights(nlat, nlon)
    names = list(D["names"])
    assert names == [v.ours for v in WV.VARS]
    core = [c[2] for c in Z.CORE]
    qi, wi = [names.index(n) for n in core], [names.index(d) for d in S.DERIVED[:4]]
    for li, h in enumerate([12, 24]):
        vt = t0 + np.timedelta64(h, "h")
        lead = fc.sel(time=t0, prediction_timedelta=np.timedelta64(h, "h"))
        mem = np.stack([Z._field(lead, v, lv) for v, lv, _ in Z.CORE], -1).astype(np.float64)
        tru = np.stack([Z._field(era5, v, lv, time=vt) for v, lv, _ in Z.CORE], -1).astype(np.float64)
        Q = mem.shape[-1]
        ref = S.numpy_lead_scores(mem, tru, np.zeros_like(tru), w, np.zeros(Q), np.ones(Q))
        np.testing.assert_allclose(D["mse"][0, li, qi], ref["mse"], rtol=2e-5)
        np.testing.assert_allclose(D["sp2"][0, li, qi], ref["spread2"], rtol=2e-5)
        np.testing.assert_allclose(D["crps"][0, li, qi], ref["crps"], rtol=2e-5)
        ws = [np.hypot(mem[..., core.index(a)], mem[..., core.index(b)]) for a, b in
              (S.WS_PAIRS[d] for d in S.DERIVED[:4])]
        wst = [np.hypot(tru[..., core.index(a)], tru[..., core.index(b)]) for a, b in
               (S.WS_PAIRS[d] for d in S.DERIVED[:4])]
        dref = S.numpy_lead_scores(np.stack(ws, -1), np.stack(wst, -1), np.zeros((tru.shape[0], 4)), w,
                                   np.zeros(4), np.ones(4))
        np.testing.assert_allclose(D["crps"][0, li, wi], dref["crps"], rtol=2e-5)
        tpi = names.index("tp24h")
        if h < 24:
            assert np.isnan(D["crps"][0, li, tpi]) and np.isnan(D["mse"][0, li, tpi])
        else:
            tpm = sum(Z._field(fc.sel(time=t0, prediction_timedelta=np.timedelta64(h - 6 * k, "h")), Z.TP6, None)
                      for k in range(4)).astype(np.float64)
            tpt = sum(Z._field(era5, Z.TP6, None, time=vt - np.timedelta64(6 * k, "h")) for k in range(4))
            r = S.numpy_lead_scores(tpm[..., None], tpt[:, None].astype(np.float64), np.zeros((tpt.size, 1)), w,
                                    np.zeros(1), np.ones(1))
            np.testing.assert_allclose(D["crps"][0, li, tpi], r["crps"][0], rtol=2e-5)


def test_score_zarr_rejects_other_grid():
    fc, era5, t0 = _datasets()
    with pytest.raises(ValueError, match="latitude"):
        Z.score_init(fc.assign_coords(latitude=fc.latitude.values + 1.0), era5, t0, [12])
    with pytest.raises(ValueError, match="longitude"):
        Z.score_init(fc.isel(longitude=slice(1, None)), era5, t0, [12])
    lat = np.linspace(-80, 80, fc.sizes["latitude"])
    with pytest.raises(ValueError, match="equiangular grid with poles"):
        Z.score_init(fc.assign_coords(latitude=lat), era5.assign_coords(latitude=lat), t0, [12])


def test_score_zarr_reversed_latitude_and_lat_lon_names(tmp_path):
    fc, era5, t0 = _datasets()
    ref = SC.load_scores(Z.score_inits(fc, era5, [t0], [12, 24], tmp_path / "ref.npz"))
    flipped = fc.isel(latitude=slice(None, None, -1))
    for i, f in enumerate((flipped, flipped.rename(latitude="lat", longitude="lon"))):
        D = SC.load_scores(Z.score_inits(f, era5, [t0], [12, 24], tmp_path / f"s{i}.npz"))
        for k in ("crps", "mse", "sp2"):
            assert np.array_equal(D[k], ref[k], equal_nan=True)


def test_score_zarr_without_tp24h(tmp_path):
    fc, era5, t0 = _datasets()
    ref = SC.load_scores(Z.score_inits(fc, era5, [t0], [12, 24], tmp_path / "ref.npz"))
    p = Z.score_inits(fc.drop_vars(Z.TP6), era5.drop_vars(Z.TP6), [t0], [12, 24], tmp_path / "s", tp24h=False)
    assert p == str(tmp_path / "s.npz")
    D = SC.load_scores(p)
    tpi = list(D["names"]).index("tp24h")
    for k in ("crps", "mse", "sp2"):
        assert np.isnan(D[k][..., tpi]).all()
        assert np.array_equal(np.delete(D[k], tpi, -1), np.delete(ref[k], tpi, -1))


def test_score_forecasts_members_and_no_tp24h(tmp_path, monkeypatch):
    fc, era5, t0 = _datasets()
    monkeypatch.setattr(xr, "open_zarr", {"FC": fc, "ERA5": era5}.__getitem__)
    spec = importlib.util.spec_from_file_location("score_forecasts", ROOT / "scripts" / "score_forecasts.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.main(["--forecast", "FC", "--era5", "ERA5", "--out", str(tmp_path / "s"), "--inits", "2020-01-02T00",
                     "--leads", "12,24", "--members", "2", "--no-tp24h"]) == 0
    D = SC.load_scores(tmp_path / "s.npz")
    ref = SC.load_scores(Z.score_inits(fc.isel(number=slice(0, 2)), era5, [t0], [12, 24], tmp_path / "ref.npz",
                                       tp24h=False))
    assert int(D["M"]) == 2
    for k in ("crps", "mse", "sp2"):
        assert np.array_equal(D[k], ref[k], equal_nan=True)
    with pytest.raises(SystemExit, match="members"):
        cli.main(["--forecast", "FC", "--era5", "ERA5", "--out", str(tmp_path / "t"), "--inits", "2020-01-02T00",
                  "--leads", "12", "--members", str(fc.sizes["number"] + 1)])
