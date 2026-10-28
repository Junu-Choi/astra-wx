"""Inputs of the 0.25 deg scorecard and lead-time-curve figures: the WB2 numbers, our year scores (7 significant
digits), the significance flags of scorecard.build, bootstrap bands and variable means.

Definitions (ours from the per-initialisation global crps, mse, sp2, M = 50):
  CRPS   mean_i crps_i
  uRMSE  sqrt(mean_i (mse_i - sp2_i / M))
  RMSE   sqrt(mean_i mse_i)
  SSR    sqrt(mean_i sp2_i / mean_i (mse_i - sp2_i / M))
The WB2 files store unbiased_spread_skill = unbiased_mean_rmse / spread, so the reference SSR is its reciprocal.
Bands: circular block bootstrap (blocks of 14 initialisations, 10,000 resamples, seed 20201231), percentile 95 %.
Per variable (ours_refs, "bands"), percentile bands of the scores themselves (CRPS, uRMSE, SSR), unwidened; the skill
band of a per-variable curve is this band mapped through 100 (1 - S / S_IFS-ENS) with the IFS ENS value held fixed and
widened by kappa about the point.  For the variable means (variable_means), the skill band is formed here, IFS ENS held
fixed and the band widened by kappa about the point; for the spread-skill ratio, unwidened.
"""
import hashlib
import math
import warnings
from pathlib import Path

import numpy as np

from .scorecard import ENS_STATE_VARS, cbb_starts, cbb_sums

ROWS = ["z500", "z700", "z850", "t500", "t700", "t850", "q500", "q700", "q850", "u500", "u700", "u850",
        "v500", "v700", "v850", "t2m", "msl", "u10", "v10", "ws10", "tp24h"]
WB2 = {"z": "geopotential", "t": "temperature", "q": "specific_humidity", "u": "u_component_of_wind",
       "v": "v_component_of_wind", "t2m": "2m_temperature", "msl": "mean_sea_level_pressure",
       "u10": "10m_u_component_of_wind", "v10": "10m_v_component_of_wind", "ws10": "10m_wind_speed",
       "tp24h": "total_precipitation_24hr"}
DET_MODELS = {"hres": "IFS HRES", "graphcast": "GraphCast", "pangu": "Pangu-Weather", "fuxi": "FuXi",
              "excarta": "HEAL-ViT", "baguan": "Baguan"}
ENS_MODELS = {"gencast": "GenCast", "ens": "IFS ENS"}
CURVE_ROWS = ["z500", "t850", "q700", "msl", "t2m", "ws10", "tp24h"]
MEAN_VARS = ["z500", "t850", "q700", "u850", "v850", "t2m", "msl", "ws10"]
DET_VARS4 = ["z500", "t850", "t2m", "ws10"]
BOOT, BLOCK, SEED = 10000, 14, 20201231


def wb2_name(v):
    """row name -> (WB2 variable, level or None)"""
    if v in WB2:
        return WB2[v], None
    return WB2[v[0]], int(v[1:])


def sha16(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def r7(x):
    return None if x is None or not np.isfinite(x) else float(f"{float(x):.7g}")


# ------------------------------------------------------------------------------------------------- WB2 files
def open_wb2(path):
    import xarray as xr
    return xr.open_dataset(path, engine="scipy").load().sel(region="global")


def pick(ds, name, level, leads):
    """values at our lead times (None where absent)"""
    if name not in ds.data_vars:
        return [None] * len(leads)
    a = ds[name]
    if level is not None:
        if level not in [int(x) for x in a.level.values]:
            return [None] * len(leads)
        a = a.sel(level=level)
    lt = [int(x) for x in np.asarray(a.lead_time.values)]
    vals = np.asarray(a.values, np.float64)
    return [r7(vals[lt.index(h)]) if h in lt else None for h in leads]


def det_coverage(ds):
    """every rmse / vector_rmse variable of a deterministic file: first and last lead with a value, count"""
    lt = [int(x) for x in np.asarray(ds.lead_time.values)]
    out = dict(leads_hours=[lt[0], lt[-1]], step_hours=lt[1] - lt[0], n_leads=len(lt), variables={})
    for name in sorted(ds.data_vars):
        if not name.startswith(("rmse.", "vector_rmse.")):
            continue
        a = ds[name]
        for lev in ([None] if "level" not in a.dims else [int(x) for x in a.level.values]):
            x = np.asarray(a.values if lev is None else a.sel(level=lev).values, np.float64)
            ok = [h for h, q in zip(lt, np.isfinite(x)) if q]
            key = name.split(".", 1)[1] + ("" if lev is None else f"@{lev}")
            out["variables"][key] = (dict(first=ok[0], last=ok[-1], n=len(ok)) if ok else dict(n=0))
    return out


def wb2_refs(wdir, leads):
    out = dict(source="https://storage.googleapis.com/weatherbench2/benchmark_results/", region="global",
               leads_hours=leads, files={}, ensemble={}, deterministic={}, ssr_identity={})
    for key in ENS_MODELS:
        f = Path(wdir) / f"{key}_vs_era5_1440x721_2020.nc"
        ds = open_wb2(f)
        out["files"][f.name] = sha16(f)
        E = {m: {} for m in ("crps", "unbiased_mean_rmse", "mean_rmse", "unbiased_spread_skill")}
        implied = []
        for v in ROWS:
            w, lev = wb2_name(v)
            for m in E:
                E[m][v] = pick(ds, f"{m}.{w}", lev, leads)
            for mr, ur, us in zip(E["mean_rmse"][v], E["unbiased_mean_rmse"][v], E["unbiased_spread_skill"][v]):
                if None not in (mr, ur, us):
                    implied.append((ur / us) ** 2 / (mr ** 2 - ur ** 2))
        E["ssr"] = {v: [None if u is None else r7(1.0 / u) for u in E["unbiased_spread_skill"][v]] for v in ROWS}
        out["ensemble"][key] = dict(label=ENS_MODELS[key], file=f.name, **E)
        out["ssr_identity"][key] = dict(
            implied_ensemble_size_min=float(np.min(implied)), implied_ensemble_size_max=float(np.max(implied)),
            n_cells=len(implied),
            formula="(unbiased_mean_rmse / unbiased_spread_skill)^2 / (mean_rmse^2 - unbiased_mean_rmse^2)")
    for key, lab in DET_MODELS.items():
        f = Path(wdir) / f"{key}_vs_era5_1440x721_2020.nc"
        ds = open_wb2(f)
        out["files"][f.name] = sha16(f)
        R = {}
        for v in ROWS:
            w, lev = wb2_name(v)
            vals = pick(ds, f"rmse.{w}", lev, leads)
            if any(x is not None for x in vals):
                R[v] = vals
        out["deterministic"][key] = dict(label=lab, file=f.name, rmse=R, coverage=det_coverage(ds))
    return out


def ssr_identity_from_states(ens_states, init_times, wdir, leads):
    """IFS ENS at 24 h: sqrt(mean var / mean ue) of the per-initialisation scores against the published
    unbiased_spread_skill at 7 significant digits (reproduce.headline compares at full precision)."""
    pos = {int(t): i for i, t in enumerate(ens_states["times"])}
    idx = np.asarray([pos[int(x)] for x in init_times])
    li = leads.index(24)
    ens = open_wb2(Path(wdir) / "ens_vs_era5_1440x721_2020.nc")
    rep = {}
    for v in ENS_STATE_VARS:
        w, lv = wb2_name(v)
        var, ue = ens_states["vals"][v]["var"][idx], ens_states["vals"][v]["ue"][idx]
        pub = pick(ens, f"unbiased_spread_skill.{w}", lv, leads)[li]
        rep[v] = dict(states_spread_over_skill=float(math.sqrt(var.mean() / ue.mean())),
                      published_unbiased_spread_skill=pub)
    return rep


# ------------------------------------------------------------------------------------------------- ours
def ours_refs(z, sc):
    """z: per-initialisation scores (scorecard.load_scores); sc: the result of scorecard.build (or its JSON)."""
    names = [str(n) for n in z["names"]]
    leads = [int(h) for h in z["leads_hours"]]
    M = int(z["M"])
    ci = [names.index(v) for v in ROWS]
    crps, mse, sp2 = (np.asarray(z[k][:, :, ci], np.float64) for k in ("crps", "mse", "sp2"))
    ue = mse - sp2 / M
    valid = np.isfinite(crps).all(0) & np.isfinite(ue).all(0)
    with np.errstate(invalid="ignore"):
        m_crps, m_ue, m_sp2, m_mse = (np.where(valid, x.mean(0), np.nan) for x in (crps, ue, sp2, mse))
        ssr = np.sqrt(m_sp2 / m_ue)
    ours = dict(n_inits=int(crps.shape[0]), M=M, leads_hours=leads, crps={}, unbiased_mean_rmse={}, mean_rmse={},
                ssr={})
    for r, v in enumerate(ROWS):
        ours["crps"][v] = [r7(x) for x in m_crps[:, r]]
        ours["unbiased_mean_rmse"][v] = [r7(x) for x in np.sqrt(m_ue[:, r])]
        ours["mean_rmse"][v] = [r7(x) for x in np.sqrt(m_mse[:, r])]
        ours["ssr"][v] = [r7(x) for x in ssr[:, r]]

    cells = {(c["ref"], c["metric"], c["var"], c["lead_h"]): c for c in sc["cells"]}
    tests = {}
    for ref in ENS_MODELS:
        T = {}
        for key, metric in (("crps", "crps"), ("unbiased_mean_rmse", "urmse")):
            T[key] = {}
            for r, v in enumerate(ROWS):
                T[key][v] = [None] * len(leads)
                for li, h in enumerate(leads):
                    c = cells.get((ref, metric, v, h))
                    if c is None:
                        continue
                    mine = m_crps[li, r] if metric == "crps" else math.sqrt(m_ue[li, r])
                    if abs(c["ours"] / mine - 1) > 1e-9:
                        raise SystemExit(f"scorecard value mismatch {ref} {metric} {v} {h}")
                    T[key][v][li] = dict(sig=bool(c["significant"]), paired=bool(c["paired"]),
                                         ci=[r7(x) for x in c.get("ci_widened", c["ci"])])
        tests[ref] = T
    ours["tests"] = tests

    N, L = crps.shape[0], len(leads)
    ri = [ROWS.index(v) for v in CURVE_ROWS]
    X = np.concatenate([np.nan_to_num(q[:, :, ri], nan=0.0).reshape(N, -1) for q in (crps, ue, sp2)], axis=1)
    bm = cbb_sums(X, cbb_starts(N, BLOCK, BOOT, SEED), BLOCK) / N
    K = L * len(ri)
    b_crps, b_ue, b_sp2 = (bm[:, i * K:(i + 1) * K].reshape(BOOT, L, len(ri)) for i in range(3))
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = dict(crps=b_crps, unbiased_mean_rmse=np.sqrt(np.where(b_ue > 0, b_ue, np.nan)),
                    ssr=np.sqrt(b_sp2 / np.where(b_ue > 0, b_ue, np.nan)))
    bands = {}
    for key, rep in reps.items():
        with warnings.catch_warnings():  # cells undefined at every resample (tp24h at 12 h) give NaN
            warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
            lo, hi = np.nanpercentile(rep, [2.5, 97.5], axis=0)
        bands[key] = {v: dict(lo=[r7(x) if valid[li, ROWS.index(v)] else None for li, x in enumerate(lo[:, j])],
                              hi=[r7(x) if valid[li, ROWS.index(v)] else None for li, x in enumerate(hi[:, j])])
                      for j, v in enumerate(CURVE_ROWS)}
    ours["bands"] = bands
    ours["bands_method"] = dict(method="circular block bootstrap over time-ordered inits, percentile 95 % band",
                                block=BLOCK, B=BOOT, seed=SEED, rows=CURVE_ROWS)
    m = sc["meta"]
    ours["significance"] = dict(source="scorecard.build (significant)",
                                method="circular block bootstrap over time-ordered inits", block=m["block"],
                                B=m["boot"], seed=m["seed"], kappa=m["unpaired_kappa"], paired=m["ens_pairing"])
    return ours


def variable_means(z, ours, wb2):
    """Means over variables of the skill relative to IFS ENS vs ERA5 (CRPS: MEAN_VARS; RMSE: DET_VARS4) and of the
    spread-skill ratio (MEAN_VARS), for ours (with bootstrap bands), GenCast, IFS ENS and the deterministic models"""
    names = [str(n) for n in z["names"]]
    M = int(z["M"])
    allv = sorted(set(MEAN_VARS + DET_VARS4))
    ci = [names.index(v) for v in allv]
    crps, mse, sp2 = (np.asarray(z[k][:, :, ci], np.float64) for k in ("crps", "mse", "sp2"))
    ue = mse - sp2 / M
    N, L = crps.shape[:2]
    X = np.concatenate([q.reshape(N, -1) for q in (crps, ue, sp2)], axis=1)
    bm = cbb_sums(X, cbb_starts(N, BLOCK, BOOT, SEED), BLOCK) / N
    K = L * len(allv)
    rep = {k: bm[:, i * K:(i + 1) * K].reshape(BOOT, L, len(allv)) for i, k in enumerate(("crps", "ue", "sp2"))}
    full = {k: q.mean(0) for k, q in (("crps", crps), ("ue", ue), ("sp2", sp2))}
    kappa = float(ours["significance"]["kappa"])
    arr = lambda xs: np.array([np.nan if x is None else x for x in xs], float)  # noqa: E731

    def score(src, metric):
        if metric == "crps":
            return src["crps"]
        if metric == "unbiased_mean_rmse":
            return np.sqrt(src["ue"])
        return np.sqrt(src["sp2"] / src["ue"])

    def mean_skill(metric, variables):
        js = [allv.index(v) for v in variables]
        ref = np.stack([arr(wb2["ensemble"]["ens"][metric][v]) for v in variables], -1)
        pt = np.mean(100 * (1 - score(full, metric)[:, js] / ref), -1)
        reps = np.mean(100 * (1 - score(rep, metric)[:, :, js] / ref[None]), -1)
        lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
        lo, hi = pt - kappa * (pt - lo), pt + kappa * (hi - pt)
        gen = np.mean(100 * (1 - np.stack([arr(wb2["ensemble"]["gencast"][metric][v]) for v in variables], -1) / ref),
                      -1)
        out = dict(variables=variables, ours=[r7(x) for x in pt], ours_lo=[r7(x) for x in lo],
                   ours_hi=[r7(x) for x in hi], gencast=[r7(x) for x in gen])
        if metric == "unbiased_mean_rmse":
            for m in DET_MODELS:
                R = wb2["deterministic"][m]["rmse"]
                if all(v in R for v in variables):
                    d = np.mean(100 * (1 - np.stack([arr(R[v]) for v in variables], -1) / ref), -1)
                    out[m] = [r7(x) for x in d]
        return out

    js = [allv.index(v) for v in MEAN_VARS]
    ssr_pt = np.mean(score(full, "ssr")[:, js], -1)
    lo, hi = np.percentile(np.mean(score(rep, "ssr")[:, :, js], -1), [2.5, 97.5], axis=0)
    ssr = dict(variables=MEAN_VARS, ours=[r7(x) for x in ssr_pt], ours_lo=[r7(x) for x in lo],
               ours_hi=[r7(x) for x in hi],
               **{k: [r7(x) for x in np.mean(np.stack([arr(wb2["ensemble"][k]["ssr"][v]) for v in MEAN_VARS], -1), -1)]
                  for k in ("gencast", "ens")})
    return dict(crps_skill=mean_skill("crps", MEAN_VARS), rmse_skill=mean_skill("unbiased_mean_rmse", DET_VARS4),
                ssr=ssr, reference="IFS ENS vs ERA5 (WB2 ens_vs_era5_1440x721_2020.nc)",
                band=dict(block=BLOCK, B=BOOT, seed=SEED, kappa=kappa,
                          skill="IFS ENS held fixed, percentile band of the variable mean widened by kappa",
                          ssr="plain percentile band of the variable mean"))


# ------------------------------------------------------------------------------------------------- scorecard panels
def panel_data(wb2, ours, ref, metric, rows):
    """improvement 100 (S_ref - S_ours) / S_ref per row and lead, significance and pairing flags"""
    L = len(ours["leads_hours"])
    val = np.full((len(rows), L), np.nan)
    sig = np.zeros((len(rows), L), bool)
    par = np.zeros((len(rows), L), bool)
    for r, v in enumerate(rows):
        for li in range(L):
            o, w = ours[metric][v][li], wb2["ensemble"][ref][metric][v][li]
            if o is None or w is None:
                continue
            val[r, li] = 100.0 * (w - o) / w
            t = ours["tests"][ref][metric][v][li]
            if t is not None:
                sig[r, li], par[r, li] = t["sig"], t["paired"]
    return val, sig, par


def lower_counts(val, leads_hours, sig=None, rows=None, _sub=False):
    """cells where ours is better (improvement > 0) over all leads and from day 3 (lead >= 72 h); with sig, also those
    better and significant and those where the reference is better and significant, and the latest lead of the
    latter; with rows holding tp24h, the same counts without tp24h"""
    out = {}
    L = np.asarray(leads_hours)
    for key, m in (("all_leads", np.ones(len(L), bool)), ("from_day_3", L >= 72)):
        v = val[:, m]
        f = np.isfinite(v)
        out[key] = dict(better=int((v[f] > 0).sum()), cells=int(f.sum()),
                        percent=round(100.0 * float((v[f] > 0).sum()) / max(int(f.sum()), 1), 2))
        if sig is not None:
            s = np.asarray(sig, bool)[:, m]
            out[key]["better_and_significant"] = int(((v > 0) & s & f).sum())
            worse = (v < 0) & s & f
            out[key]["reference_better_and_significant"] = int(worse.sum())
            out[key]["latest_lead_hours_reference_better_and_significant"] = (
                int(L[m][np.where(worse.any(axis=0))[0].max()]) if worse.any() else None)
    if rows is not None and "tp24h" in rows and not _sub:
        keep = np.array([r != "tp24h" for r in rows])
        out["without_tp24h"] = lower_counts(val[keep], L, None if sig is None else np.asarray(sig, bool)[keep],
                                            _sub=True)
    return out
