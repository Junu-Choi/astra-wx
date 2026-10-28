"""Skill relative to IFS ENS averaged over variables and the daily lead times of 1-10 days (Fig. 1a,b).

a  CRPS skill 100 (1 - CRPS / CRPS_IFS-ENS), mean over z500, q700, t850, u850, v850 and leads 24, 48, ..., 240 h.
b  ensemble-mean RMSE skill, the same with the unbiased ensemble-mean RMSE, over z500, t850, u850, v850.
Reference: IFS ENS verified against its own analysis (WB2 ens_vs_hres_t0_1440x721_2020.nc) or against ERA5
(ens_vs_era5_1440x721_2020.nc).  GenCast: gencast_vs_era5_1440x721_2020.nc.
"""
from pathlib import Path

import numpy as np

V5 = ["z500", "q700", "t850", "u850", "v850"]
V4 = ["z500", "t850", "u850", "v850"]
L10 = list(range(24, 241, 24))
WB2NAME = {"z": "geopotential", "t": "temperature", "q": "specific_humidity", "u": "u_component_of_wind",
           "v": "v_component_of_wind"}
IFS_REF_FILES = {"analysis": "ens_vs_hres_t0_1440x721_2020.nc", "era5": "ens_vs_era5_1440x721_2020.nc"}
METRIC = {"CRPS": "crps", "Unbiased Mean RMSE": "unbiased_mean_rmse", "Mean RMSE": "mean_rmse"}


def summarise_scores(z):
    """Per-initialisation scores -> 2020 means per variable and lead time {key: {var: {lead_h: value}}}."""
    names, LH = [str(s) for s in z["names"]], [int(h) for h in z["leads_hours"]]
    out = dict(M=int(z["M"]), n_inits=int(z["crps"].shape[0]), crps={}, mse={}, sp2={})
    for v in V5:
        j = names.index(v)
        for key in ("crps", "mse", "sp2"):
            a = np.asarray(z[key][:, :, j], dtype=np.float64)
            if not np.isfinite(a).all():
                raise SystemExit(f"non-finite {key} for {v}")
            out[key][v] = {str(h): float(a[:, i].mean()) for i, h in enumerate(LH)}
    return out


def skill(model, ref, vars_):
    """100 x mean over vars_ x daily leads 1-10 d of (1 - model / ref); model, ref: {var: {lead_h: score}}."""
    return 100.0 * float(np.mean([1.0 - model[v][str(h)] / ref[v][str(h)] for v in vars_ for h in L10]))


def wb2_series(path, metric, vars_):
    """{var: {lead_h: value}} of one WB2 result file (global region)."""
    import xarray as xr
    ds = xr.open_dataset(path, engine="scipy").load().sel(region="global")
    lt = [int(x) for x in np.asarray(ds.lead_time.values)]
    out = {}
    for v in vars_:
        a = np.asarray(ds[f"{METRIC[metric]}.{WB2NAME[v[0]]}"].sel(level=int(v[1:])).values, np.float64)
        out[v] = {str(h): float(a[lt.index(h)]) for h in lt}
    return out


def fig1_skills(ours, wdir):
    """Skills of Astra and GenCast relative to IFS ENS (against its analysis and against ERA5)."""
    wdir = Path(wdir)
    M = ours["M"]
    o_crps = {v: {str(h): ours["crps"][v][str(h)] for h in L10} for v in V5}
    o_urmse = {v: {str(h): float(np.sqrt(ours["mse"][v][str(h)] - ours["sp2"][v][str(h)] / M)) for h in L10}
               for v in V5}
    g = wdir / "gencast_vs_era5_1440x721_2020.nc"
    g_crps = wb2_series(g, "CRPS", V5)
    g_urmse = wb2_series(g, "Unbiased Mean RMSE", V5)
    ys = {}
    for key, fname in IFS_REF_FILES.items():
        rc = wb2_series(wdir / fname, "CRPS", V5)
        ru = wb2_series(wdir / fname, "Unbiased Mean RMSE", V5)
        ys[key] = dict(reference=fname,
                       crps=dict(ours=skill(o_crps, rc, V5), gencast=skill(g_crps, rc, V5)),
                       urmse=dict(ours=skill(o_urmse, ru, V4), gencast=skill(g_urmse, ru, V4)))
    return ys
