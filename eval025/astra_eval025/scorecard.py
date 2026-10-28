"""Year scores, comparison with the WeatherBench 2 result files of GenCast and IFS ENS, and significance tests.

Input: a per-initialisation score file (npz) with
  names (C,), init_times (N,) int64 hours since 1970-01-01 (ascending), leads_hours (L,), M,
  crps / mse / sp2 (N, L, C): per-initialisation area-weighted global fair CRPS, squared error of the ensemble mean
  and ensemble variance (M - 1 in the denominator), physical units; NaN marks an undefined cell (tp24h at 12 h).

Metrics (WeatherBench-X definitions, functions of averages over initialisations):
  crps       mean_i crps_i
  urmse      sqrt(mean_i (mse_i - sp2_i / M))              unbiased_mean_rmse
  mean_rmse  sqrt(mean_i mse_i)                            RMSE of the M-member mean (depends on M)
  spread     sqrt(mean_i sp2_i)
  ssr        sqrt(mean sp2 / mean (mse - sp2 / M))         spread / unbiased_mean_rmse

Significance: circular block bootstrap over the time-ordered initialisations, blocks of `block` initialisations,
`boot` resamples, one seed, the same resampled index sets for every test; percentile 95 % interval of the score
ratio (ours / reference); significant if it excludes 1 (p_unwidened, significant_unwidened).
  paired           per-initialisation reference scores available (IFS ENS at 24 h): both series resampled together.
  fixed reference  otherwise: the published reference value is held fixed; both half-widths of the interval are
                   multiplied by kappa (ci_widened), kappa = median over the paired tests of the ratio of
                   paired to fixed-reference interval widths (at least 1).
significant: the paired test where paired, otherwise the widened test (the unwidened one when there is no kappa).
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np

from . import variables as WV

REF_FILES = {"gencast": "gencast_vs_era5_1440x721_2020.nc", "ens": "ens_vs_era5_1440x721_2020.nc"}
REF_M = {"gencast": 56, "ens": 50}             # members of the WB2 GenCast and IFS ENS forecasts
TESTED = ("crps", "urmse")
PAPER_ROWS = tuple(f"{p}{lv}" for p in "ztquv" for lv in WV.LEVELS) + ("t2m", "msl", "u10", "v10", "ws10", "tp24h")
FULL_N = 732
HOURS0 = np.datetime64("1970-01-01T00", "h")
FULL_TIMES = (np.arange(np.datetime64("2020-01-01T00", "h"), np.datetime64("2021-01-01T00", "h"),
                        np.timedelta64(12, "h")) - HOURS0).astype(np.int64)


# ------------------------------------------------------------------------------------------------------ input
def _s(z, k, default=""):
    return str(z[k]) if k in z else default


def load_scores(path) -> dict:
    """Read and validate a per-initialisation score file."""
    with np.load(path, allow_pickle=False) as z:
        D = {k: z[k] for k in z.files}
    if int(D.get("version", -1)) != 1:
        raise ValueError(f"{path}: version {D.get('version')} != 1")
    return validate(D, str(path))


def validate(D: dict, where: str = "scores") -> dict:
    D = dict(D)
    names = [str(n) for n in D["names"]]
    if len(set(names)) != len(names):
        raise ValueError(f"{where}: duplicate channel names")
    t = np.asarray(D["init_times"], np.int64)
    if t.ndim != 1 or (len(t) > 1 and not np.all(np.diff(t) > 0)):
        raise ValueError(f"{where}: init_times must be strictly ascending (time-ordered for the block bootstrap)")
    L = len(D["leads_hours"])
    for k in ("crps", "mse", "sp2"):
        if D[k].shape != (len(t), L, len(names)):
            raise ValueError(f"{where}: {k} shape {D[k].shape} != {(len(t), L, len(names))}")
    if int(D["M"]) < 2:
        raise ValueError(f"{where}: M must be >= 2")
    D = dict(D)
    D["names"] = np.asarray(names)
    D["init_times"] = t
    D["leads_hours"] = np.asarray(D["leads_hours"], np.int64)
    return D


def coverage(D) -> dict:
    """Initialisation coverage; complete = the 732 initialisations of 2020 at 00 and 12 UTC."""
    t = HOURS0 + np.asarray(D["init_times"], np.int64) * np.timedelta64(1, "h")
    mo = t.astype("datetime64[M]")
    hh = (t - t.astype("datetime64[D]")).astype(np.int64)
    per_month = {str(m): int((mo == m).sum()) for m in np.unique(mo)}
    complete = bool(np.array_equal(np.asarray(D["init_times"], np.int64), FULL_TIMES))
    return dict(n=int(len(t)), expect_n=FULL_N, per_month=per_month, n_00z=int((hh == 0).sum()),
                n_12z=int((hh == 12).sum()), complete=complete,
                why=None if complete else "init_times are not the 732 initialisations of 2020 at 00 and 12 UTC")


# ------------------------------------------------------------------------------------------------ our metrics
def _cell_valid(a, name):
    """(N, L, C) -> (L, C) bool: finite for every initialisation; mixed finite / NaN is an error."""
    fin = np.isfinite(a)
    allf, anyf = fin.all(0), fin.any(0)
    bad = anyf & ~allf
    if bad.any():
        li, ci = np.argwhere(bad)[0]
        raise ValueError(f"{name}: cell (lead index {li}, channel index {ci}) is NaN for some inits only")
    return allf


def ours_metrics(D) -> dict:
    """(L, C) arrays of every metric (NaN = undefined)."""
    M = float(D["M"])
    v = _cell_valid(D["crps"], "crps") & _cell_valid(D["mse"], "mse") & _cell_valid(D["sp2"], "sp2")
    nan = np.where(v, 1.0, np.nan)
    crps = D["crps"].mean(0) * nan
    mse, sp2 = D["mse"].mean(0) * nan, D["sp2"].mean(0) * nan
    ue = mse - sp2 / M
    return dict(crps=crps, urmse=np.sqrt(np.where(ue > 0, ue, np.nan)), mean_rmse=np.sqrt(mse), spread=np.sqrt(sp2),
                ssr=np.sqrt(sp2 / np.where(ue > 0, ue, np.nan)))


# ------------------------------------------------------------------------------------------------- references
def _open(path):
    import xarray as xr
    return xr.open_dataset(path, engine="scipy").load()


def load_refs(ref_dir, leads_hours, rows) -> dict:
    """{ref: {metric: {var: (L,) array}}} of the WB2 result files, global region, at our lead times (NaN if absent)."""
    ref_dir = Path(ref_dir)
    out = {}
    for ref, fprob in REF_FILES.items():
        P = _open(ref_dir / fprob).sel(region="global")
        lt = [int(x) for x in np.asarray(P.lead_time.values)]
        R = {m: {} for m in ("crps", "urmse", "mean_rmse", "ssr_pub")}

        def pick(ds, lts, name, var):
            if name not in ds.data_vars:
                return np.full(len(leads_hours), np.nan)
            a = ds[name]
            if var.level is not None:
                if var.level not in [int(x) for x in a.level.values]:
                    return np.full(len(leads_hours), np.nan)
                a = a.sel(level=var.level)
            vals = np.asarray(a.values, np.float64)
            return np.asarray([vals[lts.index(int(h))] if int(h) in lts else np.nan for h in leads_hours])

        for var in rows:
            R["crps"][var.ours] = pick(P, lt, f"crps.{var.wb2}", var)
            R["urmse"][var.ours] = pick(P, lt, f"unbiased_mean_rmse.{var.wb2}", var)
            R["mean_rmse"][var.ours] = pick(P, lt, f"mean_rmse.{var.wb2}", var)
            R["ssr_pub"][var.ours] = pick(P, lt, f"unbiased_spread_skill.{var.wb2}", var)
        out[ref] = R
    return out


# ------------------------------------------------------------------------------ IFS ENS per-initialisation scores
ENS_STATE_VARS = ("z500", "z700", "z850", "t500", "t700", "t850", "t2m", "msl")
_STAT = {"skill": "CRPSSkill_number", "spread": "CRPSSpread_number_fair_predictions",
         "var": "EnsembleVariance_number_skipna_ensemble_False",
         "ue": "UnbiasedEnsembleMeanSquaredError_number_skipna_ensemble_False",
         "se": "SquaredError_predictions_ensemble_mean_self._ensemble_dim='number'_self._skipna=False"}


def _state_time(path) -> int:
    s = Path(path).stem.replace("state_", "")
    return int((np.datetime64(s, "h") - HOURS0).astype(np.int64))


def _read_state(path):
    """One WeatherBench-X aggregation state file (state_<init>.nc) -> {(stat, var): global mean}, lead hours."""
    from scipy.io import netcdf_file
    f = netcdf_file(str(path), "r", mmap=False)
    try:
        reg = [b"".join(r).decode("utf-8").strip("\x00").strip() for r in f.variables["region"].data]
        gi = reg.index("global")
        lev = [int(x) for x in f.variables["level"].data]
        lead = [int(x) for x in f.variables["lead_time"].data]
        units = f.variables["lead_time"]._attributes.get("units", b"hours")
        if (units.decode() if isinstance(units, bytes) else str(units)) != "hours":
            raise ValueError(f"{path}: lead_time units {units!r}")
        out = {}
        for v in ENS_STATE_VARS:
            var = WV.BY_OURS[v]
            for stat, pre in _STAT.items():
                a = f.variables[f"{pre}#{var.wb2}#sum_weighted_statistics"]
                w = f.variables[f"{pre}#{var.wb2}#sum_weights"]
                if var.level is not None:
                    li = lev.index(var.level)
                    sa, sw = float(a.data[0, li, gi]), float(w.data[0, li, gi])
                else:
                    sa, sw = float(a.data[0, gi]), float(w.data[0, gi])
                out[(stat, v)] = sa / sw
        return out, lead
    finally:
        f.close()


def read_ens_state_files(state_dir):
    """Per-initialisation IFS ENS 24 h scores from WeatherBench-X state files -> (times, vals, lead_ok)."""
    files = sorted(glob.glob(str(Path(state_dir) / "state_*.nc")))
    if not files:
        return None
    times = np.asarray([_state_time(f) for f in files], np.int64)
    vals = {v: {k: np.empty(len(files)) for k in ("crps", "ue", "se", "var")} for v in ENS_STATE_VARS}
    lead_ok = True
    for i, f in enumerate(files):
        st, lead = _read_state(f)
        lead_ok &= (lead == [24])
        for v in ENS_STATE_VARS:
            vals[v]["crps"][i] = st[("skill", v)] - 0.5 * st[("spread", v)]
            vals[v]["ue"][i] = st[("ue", v)]
            vals[v]["se"][i] = st[("se", v)]
            vals[v]["var"][i] = st[("var", v)]
    return times, vals, lead_ok


def save_ens_states_npz(path, times, vals, lead_ok):
    arrs = {f"{v}.{k}": vals[v][k] for v in vals for k in vals[v]}
    np.savez(path, times=np.asarray(times, np.int64), lead_ok=np.asarray(bool(lead_ok)), **arrs)


def read_ens_states_npz(path):
    with np.load(path, allow_pickle=False) as z:
        if "times" not in z.files:
            return None
        times = np.asarray(z["times"], np.int64)
        vals = {v: {k: np.asarray(z[f"{v}.{k}"]) for k in ("crps", "ue", "se", "var")} for v in ENS_STATE_VARS}
        lead_ok = bool(z["lead_ok"])
    return times, vals, lead_ok


def load_ens_states(source, refs_dir, *, tol=1e-4, expect_n=732) -> dict | None:
    """Per-initialisation IFS ENS 24 h scores {var: dict(crps, ue, se, var) (n,)} + times + validation.
    source: a directory of WeatherBench-X state files or an npz written by save_ens_states_npz.  The means over the
    initialisations must reproduce the published 24 h global crps / unbiased_mean_rmse / mean_rmse of
    ens_vs_era5_1440x721_2020.nc to < tol (ok=False otherwise: every IFS ENS test is then a fixed-reference test)."""
    r = read_ens_states_npz(source) if str(source).endswith(".npz") else read_ens_state_files(source)
    if r is None:
        return None
    times, vals, lead_ok = r
    n = len(times)
    o = np.argsort(times)
    times = times[o]
    for v in vals:
        for k in vals[v]:
            vals[v][k] = vals[v][k][o]
    pub = _open(Path(refs_dir) / REF_FILES["ens"]).sel(region="global", lead_time=24)
    worst, rep = 0.0, {}
    for v in ENS_STATE_VARS:
        var = WV.BY_OURS[v]
        sel = (lambda a: a.sel(level=var.level)) if var.level is not None else (lambda a: a)
        mine = dict(crps=vals[v]["crps"].mean(), unbiased_mean_rmse=math.sqrt(vals[v]["ue"].mean()),
                    mean_rmse=math.sqrt(vals[v]["se"].mean()))
        for m, x in mine.items():
            p = float(sel(pub[f"{m}.{var.wb2}"]).values)
            r_ = abs(x - p) / abs(p)
            rep[f"{m}.{v}"] = dict(states=x, published=p, rel=r_)
            worst = max(worst, r_)
    ok = bool(lead_ok and n == expect_n and worst < tol)
    return dict(times=times, vals=vals, n=n, worst_rel=worst, ok=ok, report=rep, lead_ok=lead_ok)


# ------------------------------------------------------------------------------------ circular block bootstrap
def cbb_starts(N: int, block: int, B: int, seed: int) -> np.ndarray:
    """(B, ceil(N / block)) block start indices; the same sets serve every test."""
    k = -(-N // block)
    return np.random.default_rng(seed).integers(0, N, size=(B, k), dtype=np.int64)


def cbb_sums(X: np.ndarray, starts: np.ndarray, block: int, chunk: int = 64) -> np.ndarray:
    """X (N, K) -> (B, K): sums of each circular-block resample of length N (last block truncated)."""
    N, K = X.shape
    B, k = starts.shape
    lens = np.full(k, block, np.int64)
    lens[-1] = N - (k - 1) * block
    if lens[-1] <= 0:
        raise ValueError("bad block layout")
    cs = np.zeros((2 * N + 1, K), np.float64)
    np.cumsum(np.concatenate([X, X], 0), axis=0, out=cs[1:])
    out = np.empty((B, K), np.float64)
    for b0 in range(0, B, chunk):
        s = starts[b0:b0 + chunk]
        e = s + lens[None, :]
        out[b0:b0 + chunk] = (cs[e] - cs[s]).sum(1)
    return out


def _ci_p(rep: np.ndarray):
    """Percentile 95 % interval and two-sided bootstrap p-value for ratio == 1 (rep: (B, ...))."""
    lo, hi = np.nanpercentile(rep, [2.5, 97.5], axis=0)
    B = np.sum(np.isfinite(rep), axis=0)
    le, ge = np.sum(rep <= 1.0, axis=0) / np.maximum(B, 1), np.sum(rep >= 1.0, axis=0) / np.maximum(B, 1)
    p = np.clip(2.0 * np.minimum(le, ge), 1.0 / np.maximum(B, 1), 1.0)
    return lo, hi, p


# ------------------------------------------------------------------------------------------------- scorecard
def build(D, refs, ens_states, *, boot=10000, block=14, block_alt=None, seed=20201231) -> dict:
    names = [str(n) for n in D["names"]]
    rows = WV.present(names)
    q = {v.ours: names.index(v.ours) for v in rows}
    N, L = len(D["init_times"]), len(D["leads_hours"])
    M = float(D["M"])
    om = ours_metrics(D)
    leads = [int(h) for h in D["leads_hours"]]

    qs = [q[v.ours] for v in rows]
    S = {"crps": D["crps"][:, :, qs], "ue": D["mse"][:, :, qs] - D["sp2"][:, :, qs] / M}
    keys = list(S)
    R = len(rows)

    def boot_sums(bk):
        X = np.concatenate([np.nan_to_num(S[k], nan=0.0).reshape(N, -1) for k in keys], axis=1)
        sums = cbb_sums(X, cbb_starts(N, bk, boot, seed), bk)
        return {k: sums[:, i * L * R:(i + 1) * L * R].reshape(boot, L, R) / N for i, k in enumerate(keys)}

    blocks = (block,) if block_alt is None else (block, block_alt)
    BS = {bk: boot_sums(bk) for bk in blocks}

    def rep_metric(bs, metric):
        with np.errstate(invalid="ignore", divide="ignore"):
            if metric == "crps":
                return bs["crps"]
            if metric == "urmse":
                return np.sqrt(np.where(bs["ue"] > 0, bs["ue"], np.nan))
            raise KeyError(metric)

    paired = {}
    ens_note = "no IFS ENS per-initialisation scores"
    if ens_states is not None:
        if not ens_states["ok"]:
            ens_note = (f"IFS ENS per-initialisation scores rejected (n {ens_states['n']}, worst rel "
                        f"{ens_states['worst_rel']:.2e}, lead_ok {ens_states['lead_ok']}): all IFS ENS tests unpaired")
        elif 24 not in leads:
            ens_note = "IFS ENS per-initialisation scores valid but 24 h is not one of the lead times"
        else:
            pos = {int(t): i for i, t in enumerate(ens_states["times"])}
            idx = [pos.get(int(t), -1) for t in D["init_times"]]
            if min(idx) < 0:
                ens_note = (f"{sum(i < 0 for i in idx)} initialisations have no IFS ENS score: IFS ENS unpaired")
            else:
                idx = np.asarray(idx)
                for v in rows:
                    if v.ours in ens_states["vals"]:
                        paired[v.ours] = {k: ens_states["vals"][v.ours][k][idx] for k in ("crps", "ue")}
                ens_note = (f"IFS ENS 24 h paired on {len(idx)} inits for {sorted(paired)} (per-initialisation "
                            f"scores reproduce the published file to {ens_states['worst_rel']:.1e})")
    li24 = leads.index(24) if 24 in leads else None
    starts_p = {bk: cbb_starts(N, bk, boot, seed) for bk in blocks}

    cells = []
    for ref in ("gencast", "ens"):
        for metric in ("crps", "urmse"):
            rep = {bk: rep_metric(BS[bk], metric) for bk in BS}
            for ri, var in enumerate(rows):
                for li, h in enumerate(leads):
                    ours = float(om[metric][li, q[var.ours]])
                    pub = float(refs[ref][metric][var.ours][li]) if var.ours in refs[ref][metric] else float("nan")
                    if not (np.isfinite(ours) and np.isfinite(pub)):
                        continue
                    c = dict(ref=ref, metric=metric, var=var.ours, wb2_var=var.wb2, level=var.level, lead_h=h,
                             ours=ours, ref_published=pub, n_inits=N, block=block)
                    is_paired = (ref == "ens" and metric in TESTED and li == li24 and var.ours in paired)
                    if is_paired:
                        pr = paired[var.ours]
                        own_i = S["crps"][:, li, ri] if metric == "crps" else S["ue"][:, li, ri]
                        ref_i = pr["crps"] if metric == "crps" else pr["ue"]
                        refv = float(ref_i.mean()) if metric == "crps" else math.sqrt(float(ref_i.mean()))
                        ratio = ours / refv
                        out_ci = {}
                        for bk in blocks:
                            rs = cbb_sums(np.stack([own_i, ref_i], 1), starts_p[bk], bk) / N
                            rr = rs[:, 0] / rs[:, 1] if metric == "crps" else np.sqrt(rs[:, 0] / rs[:, 1])
                            out_ci[bk] = _ci_p(rr)
                        lo, hi, p = out_ci[block]
                        ulo, uhi, _ = _ci_p(rep[block][:, li, ri] / refv)
                        rho = float(np.corrcoef(own_i, ref_i)[0, 1])
                        c.update(ref=ref, ref_value=refv, ref_on="our inits (per-initialisation scores)", paired=True,
                                 rho=rho, ci_unpaired=[float(ulo), float(uhi)],
                                 ci_width_ratio_unpaired_over_paired=float((uhi - ulo) / max(hi - lo, 1e-300)),
                                 ratio_vs_published=ours / pub)
                        if block_alt is not None:
                            lo_alt, hi_alt, p_alt = out_ci[block_alt]
                    else:
                        refv = pub
                        ratio = ours / refv
                        lo, hi, p = _ci_p(rep[block][:, li, ri] / refv)
                        if block_alt is not None:
                            lo_alt, hi_alt, p_alt = _ci_p(rep[block_alt][:, li, ri] / refv)
                        c.update(ref_value=refv, ref_on="published (WB2 result file)", paired=False)
                    sig = bool(not (lo <= 1.0 <= hi))
                    c.update(ratio=ratio, pct=100.0 * (ratio - 1.0), ci=[float(lo), float(hi)], p_unwidened=float(p),
                             significant_unwidened=sig, better=bool(ratio < 1.0), tested=metric in TESTED)
                    if block_alt is not None:
                        c.update(ci_alt=[float(lo_alt), float(hi_alt)], p_alt=float(p_alt), block_alt=block_alt)
                    cells.append(c)
        for ri, var in enumerate(rows):
            for li, h in enumerate(leads):
                ssr_ref = refs[ref]["ssr_pub"].get(var.ours, np.full(L, np.nan))[li]
                o = float(om["ssr"][li, q[var.ours]])
                if np.isfinite(o):
                    cells.append(dict(ref=ref, metric="ssr", var=var.ours, wb2_var=var.wb2, level=var.level, lead_h=h,
                                      ours=o, ref_published=float(ssr_ref), tested=False, paired=False, n_inits=N,
                                      ours_definition="sqrt(mean var / mean (mse - var / M))",
                                      ref_definition="WB2 unbiased_spread_skill (its reciprocal is the spread-skill "
                                                     "ratio)"))
                mr_ref = refs[ref]["mean_rmse"].get(var.ours, np.full(L, np.nan))[li]
                o = float(om["mean_rmse"][li, q[var.ours]])
                if np.isfinite(o) and np.isfinite(mr_ref):
                    cells.append(dict(ref=ref, metric="mean_rmse", var=var.ours, wb2_var=var.wb2, level=var.level,
                                      lead_h=h, ours=o, ref_published=float(mr_ref), ratio=o / float(mr_ref),
                                      tested=False, paired=False, n_inits=N,
                                      note=f"depends on the ensemble size: ours M={int(M)}, reference M={REF_M[ref]}"))

    pw = [(c["ci"][1] - c["ci"][0]) / (c["ci_unpaired"][1] - c["ci_unpaired"][0])
          for c in cells if c.get("paired") and c["ci_unpaired"][1] > c["ci_unpaired"][0]]
    kappa = float(max(1.0, np.median(pw))) if pw else None
    for c in cells:
        if "ci" not in c:
            continue
        if c["paired"] or kappa is None:
            c["significant"] = c["significant_unwidened"]
            continue
        (lo, hi), r = c["ci"], c["ratio"]
        c["ci_widened"] = [r - kappa * (r - lo), r + kappa * (hi - r)]
        c["significant"] = bool(not (c["ci_widened"][0] <= 1.0 <= c["ci_widened"][1]))

    def summarise(targets):
        targets = set(targets)
        summary = {}
        for ref in ("gencast", "ens"):
            for metric in ("crps", "urmse"):
                per = {}
                for h in leads:
                    cc = [c for c in cells if c["ref"] == ref and c["metric"] == metric and c["lead_h"] == h
                          and c["var"] in targets]
                    if not cc:
                        continue
                    r = np.asarray([c["ratio"] for c in cc])
                    per[h] = dict(n=len(cc), mean_ratio=float(r.mean()),
                                  geomean_ratio=float(np.exp(np.log(r).mean())),
                                  wins=sum(c["better"] for c in cc), losses=sum(not c["better"] for c in cc),
                                  sig_wins_unwidened=sum(c["better"] and c["significant_unwidened"] for c in cc),
                                  sig_losses_unwidened=sum((not c["better"]) and c["significant_unwidened"]
                                                           for c in cc),
                                  sig_wins=sum(c["better"] and c["significant"] for c in cc),
                                  sig_losses=sum((not c["better"]) and c["significant"] for c in cc),
                                  paired=sum(c["paired"] for c in cc))
                summary[f"{ref}.{metric}"] = per
        return summary

    summary = summarise([v.ours for v in rows])
    summary_paper_rows = summarise(PAPER_ROWS)  # the paper's 20 targets and tp24h
    meta = dict(n_inits=N, M=int(M), leads_hours=leads, rows=[v.ours for v in rows], boot=boot, block=block,
                block_alt=block_alt, seed=seed, ens_pairing=ens_note, unpaired_kappa=kappa,
                kappa_from=(dict(cells=len(pw), median=float(np.median(pw)), min=float(np.min(pw)),
                                 max=float(np.max(pw))) if pw else None),
                first_init=str(HOURS0 + np.timedelta64(int(D["init_times"][0]), "h")),
                last_init=str(HOURS0 + np.timedelta64(int(D["init_times"][-1]), "h")),
                notes=_s(D, "notes"))
    if ens_states is not None:
        meta["ens_states_validation"] = dict(n=ens_states["n"], ok=ens_states["ok"],
                                             worst_rel=ens_states["worst_rel"], report=ens_states["report"])
    return dict(meta=meta, summary=summary, summary_paper_rows=summary_paper_rows, cells=cells, ours=om)


# ------------------------------------------------------------------------------------------ WB2-layout netCDF
def write_wb2_nc(D, om, path, *, label=""):
    """Our year scores in the layout of the WB2 result files (region 'global', lead_time [h], level 500/700/850)."""
    import xarray as xr
    names = [str(n) for n in D["names"]]
    rows = WV.present(names)
    leads = np.asarray(D["leads_hours"], np.int32)
    lev = np.asarray(WV.LEVELS, np.int32)
    data = {}
    om = dict(om, unbiased_spread_skill=om["urmse"] / om["spread"])  # the WB2 quantity, 1 / ssr
    for metric, key in (("crps", "crps"), ("urmse", "unbiased_mean_rmse"), ("mean_rmse", "mean_rmse"),
                        ("unbiased_spread_skill", "unbiased_spread_skill"), ("spread", "spread")):
        if metric not in om:
            continue
        for wb2 in dict.fromkeys(v.wb2 for v in rows):
            vs = [v for v in rows if v.wb2 == wb2]
            if vs[0].level is None:
                a = om[metric][:, names.index(vs[0].ours)][None, :]
                data[f"{key}.{wb2}"] = (("region", "lead_time"), a.astype(np.float64))
            else:
                a = np.full((1, len(leads), len(lev)), np.nan)
                for v in vs:
                    a[0, :, list(lev).index(v.level)] = om[metric][:, names.index(v.ours)]
                data[f"{key}.{wb2}"] = (("region", "lead_time", "level"), a)
    ds = xr.Dataset(data, coords=dict(region=np.asarray(["global"]), lead_time=leads, level=lev))
    ds["lead_time"].attrs["units"] = "hours"
    ds.attrs.update(dict(
        label=label, ensemble_members=int(D["M"]), n_inits=int(len(D["init_times"])),
        definitions="WeatherBench-X: crps fair; unbiased_mean_rmse sqrt(mean(SE_ensmean - var/M)); "
                    "unbiased_spread_skill unbiased_mean_rmse / spread, sqrt(mean UEMSE / mean var), as in the WB2 "
                    "files (the spread-skill ratio is its reciprocal); spread sqrt(mean var)",
        mean_rmse_note=f"mean_rmse is the RMSE of the {int(D['M'])}-member ensemble mean (depends on M)"))
    ds.to_netcdf(path, engine="scipy")
    return path


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def run(scores, out, *, refs_dir, ens_states=None, boot=10000, block=14, block_alt=None, seed=20201231,
        allow_partial=False, label="Astra", log=print):
    """Score file -> out/scorecard.json and out/astra_vs_era5_1440x721_2020.nc."""
    D = scores if isinstance(scores, dict) else load_scores(scores)
    cov = coverage(D)
    if not cov["complete"] and not allow_partial:
        raise SystemExit(f"refusing an incomplete initialisation set: {cov['why']} "
                         "(--allow-partial / allow_partial=True for a partial set)")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = WV.present(D["names"])
    if not rows:
        raise ValueError("no WB2 variable in the score file")
    refs = load_refs(refs_dir, D["leads_hours"], rows)
    ens = None
    if ens_states:
        ens = load_ens_states(ens_states, refs_dir)
        if ens is None:
            raise SystemExit(f"{ens_states}: no IFS ENS per-initialisation scores (expected WeatherBench-X "
                             "state_*.nc files or an npz written by save_ens_states_npz)")
        log(f"IFS ENS per-initialisation scores: n {ens['n']}, reproduce the published file to "
            f"{ens['worst_rel']:.2e} -> {'OK' if ens['ok'] else 'REJECTED'}")
    res = build(D, refs, ens, boot=boot, block=block, block_alt=block_alt, seed=seed)
    res["meta"]["label"] = label
    res["meta"]["coverage"] = cov
    write_wb2_nc(D, res["ours"], out / "astra_vs_era5_1440x721_2020.nc", label=label)
    js = {k: v for k, v in res.items() if k != "ours"}
    (out / "scorecard.json").write_text(json.dumps(js, indent=1, default=_json_default) + "\n")
    log(f"wrote {out / 'scorecard.json'}")
    return res
