"""The 0.25 deg score-based numbers of the paper, recomputed from the stored per-initialisation scores of Astra, the
WB2 result files and the per-initialisation IFS ENS scores at 24 h."""
import json
from pathlib import Path

import numpy as np

from . import figure_inputs as FI
from . import scorecard as SC
from . import skill as SK
from . import variables as WV

DATA = Path(__file__).resolve().parents[1] / "data"
MAIN_ROWS = ["z500", "t850", "q700", "u850", "v850", "t2m", "msl", "ws10"]
FULL_ROWS = [f"{v}{lv}" for v in "ztquv" for lv in (500, 700, 850)] + ["t2m", "msl", "u10", "v10", "ws10", "tp24h"]
TARGETS20 = FULL_ROWS[:-1]
PANELS = {"a": ("gencast", "crps"), "b": ("gencast", "unbiased_mean_rmse"), "c": ("ens", "crps"),
          "d": ("ens", "unbiased_mean_rmse")}


def compute(data_dir=DATA, log=print):
    """Every intermediate product: scorecard result, figure inputs, panel values and counts, Fig. 1 skills."""
    data_dir = Path(data_dir)
    wdir = data_dir / "wb2"
    D = SC.load_scores(data_dir / "astra_2020_025_per_init_scores.npz")
    rows = WV.present(D["names"])
    refs = SC.load_refs(wdir, D["leads_hours"], rows)
    ens = SC.load_ens_states(data_dir / "ifs_ens_2020_24h_per_init_scores.npz", wdir)
    log(f"IFS ENS per-initialisation scores reproduce the published file to {ens['worst_rel']:.2e}")
    res = SC.build(D, refs, ens)
    sc = json.loads(json.dumps({k: v for k, v in res.items() if k != "ours"}, default=SC._json_default))
    leads = [int(h) for h in D["leads_hours"]]
    wb2 = FI.wb2_refs(wdir, leads)
    wb2["ssr_identity"]["ens_states_24h"] = FI.ssr_identity_from_states(ens, D["init_times"], wdir, leads)
    ours = FI.ours_refs(D, sc)
    ours["means"] = FI.variable_means(D, ours, wb2)
    panels = {}
    for variant, prow in (("main", MAIN_ROWS), ("full", FULL_ROWS)):
        for k, (ref, metric) in PANELS.items():
            val, sig, par = FI.panel_data(wb2, ours, ref, metric, prow)
            panels[(variant, k)] = dict(rows=prow, values=val, significant=sig, paired=par,
                                        counts=FI.lower_counts(val, leads, sig, prow))
    fig1 = SK.fig1_skills(SK.summarise_scores(D), wdir)
    return dict(D=D, res=res, sc=sc, wb2=wb2, ours=ours, panels=panels, fig1=fig1, ens=ens, leads=leads)


def _analysis_counts(res, wdir, leads):
    """Astra (vs ERA5) against IFS ENS verified against its own analysis: CRPS lower on how many of the scored
    combinations of the 20 targets, and from which lead time on every one."""
    ds = FI.open_wb2(Path(wdir) / "ens_vs_hres_t0_1440x721_2020.nc")
    om, names = res["ours"], list(res["meta"]["rows"])
    better = np.zeros((len(TARGETS20), len(leads)), bool)
    scored = np.zeros_like(better)
    for r, v in enumerate(TARGETS20):
        w, lev = FI.wb2_name(v)
        ref = FI.pick(ds, f"crps.{w}", lev, leads)
        for li, x in enumerate(ref):
            o = om["crps"][li, names.index(v)]
            if x is None or not np.isfinite(o):
                continue
            scored[r, li] = True
            better[r, li] = o < x
    allb = [h for li, h in enumerate(leads) if (better[:, li] | ~scored[:, li]).all() and scored[:, li].any()]
    first_all = next((h for i, h in enumerate(leads) if all(x in allb for x in leads[i:])), None)
    return int(better.sum()), int(scored.sum()), first_all


def headline(P, data_dir=DATA, out=None):
    """[(id, statement, paper value, reproduced value, match)], appended to ``out`` when given"""
    wdir = Path(data_dir) / "wb2"
    out = [] if out is None else out

    def add(i, text, paper, mine, ok=None):
        out.append(dict(id=i, statement=text, paper=paper, reproduced=mine,
                        match=bool(paper == mine) if ok is None else bool(ok)))

    leads = P["leads"]
    L = np.asarray(leads)
    c = {k: v["counts"] for k, v in P["panels"].items()}
    fa, fb, fc, fd = (c[("full", k)] for k in "abcd")
    add("gc_crps_600", "CRPS lower than GenCast, 20 targets x 30 leads", "441/600 (73.5%)",
        f"{fa['without_tp24h']['all_leads']['better']}/{fa['without_tp24h']['all_leads']['cells']} "
        f"({fa['without_tp24h']['all_leads']['percent']:.1f}%)")
    add("gc_crps_500", "CRPS lower than GenCast from day 3", "406/500 (81%)",
        f"{fa['without_tp24h']['from_day_3']['better']}/{fa['without_tp24h']['from_day_3']['cells']} "
        f"({fa['without_tp24h']['from_day_3']['percent']:.0f}%)")
    add("ens_crps_600", "CRPS lower than IFS ENS (vs ERA5) on all 600", "600/600",
        f"{fc['without_tp24h']['all_leads']['better']}/{fc['without_tp24h']['all_leads']['cells']}")
    add("ens_rmse_600", "Ensemble-mean RMSE lower than IFS ENS (vs ERA5) on all 600", "600/600",
        f"{fd['without_tp24h']['all_leads']['better']}/{fd['without_tp24h']['all_leads']['cells']}")
    add("ens_crps_sig", "CRPS significantly lower than IFS ENS", "594/600",
        f"{fc['without_tp24h']['all_leads']['better_and_significant']}/600")
    add("ens_rmse_sig", "RMSE significantly lower than IFS ENS", "489/600",
        f"{fd['without_tp24h']['all_leads']['better_and_significant']}/600")
    for k, paper in (("a", "441 / 38 / 50"), ("b", "320 / 39 / 38"), ("c", "600 / 594 / 0"), ("d", "600 / 489 / 0")):
        s = c[("full", k)]["without_tp24h"]["all_leads"]
        add(f"ed_{k}_600", f"ED Fig. scorecard {k}: better / significantly better / reference significantly better "
                           "(600, no tp24h)", paper,
            f"{s['better']} / {s['better_and_significant']} / {s['reference_better_and_significant']}")
    for k, paper in (("a", "460 / 38 / 56"), ("b", "322 / 39 / 44"), ("c", "629 / 623 / 0"), ("d", "629 / 518 / 0")):
        s = c[("full", k)]["all_leads"]
        add(f"ed_{k}_629", f"ED Fig. scorecard {k}: the same counts with tp24h (629 cells)", paper,
            f"{s['better']} / {s['better_and_significant']} / {s['reference_better_and_significant']}")
    m = c[("main", "a")]["from_day_3"]
    add("fig4a_200", "Fig. 4a: CRPS lower than GenCast from day 3", "153/200", f"{m['better']}/{m['cells']}")
    mc = c[("main", "c")]["all_leads"]
    add("fig4c_sig", "Fig. 4c: CRPS significantly lower than IFS ENS for every variable and lead time", "240/240",
        f"{mc['better_and_significant']}/{mc['cells']}")
    late = max((x for x in (fa["without_tp24h"]["all_leads"]["latest_lead_hours_reference_better_and_significant"],
                            fb["without_tp24h"]["all_leads"]["latest_lead_hours_reference_better_and_significant"])
                if x is not None), default=None)
    add("gc_sig_60h", "Latest lead with a significant difference in GenCast's favour (CRPS or RMSE, 20 targets)",
        "<= 60 h", f"{late} h", ok=late is not None and late <= 60)
    pa = P["panels"][("full", "a")]
    tpi = FULL_ROWS.index("tp24h")
    w_tp = np.any([(p["values"][tpi] < 0) & p["significant"][tpi] & np.isfinite(p["values"][tpi])
                   for p in (pa, P["panels"][("full", "b")])], axis=0)
    late_tp = int(L[np.where(w_tp)[0].max()]) if w_tp.any() else None
    add("gc_sig_tp", "tp24h: latest lead with a significant difference in GenCast's favour (CRPS or RMSE)",
        "<= 84 h (3.5 d)", f"{late_tp} h", ok=late_tp is not None and late_tp <= 84)
    worst = min(float(np.nanmin(P["panels"][("full", k)]["values"][:len(TARGETS20)][:, L >= 72])) for k in "ab")
    add("gc_08", "From day 3, largest excess of either score over GenCast's (20 targets)", "<= 0.8 %",
        f"{-worst:.3f} %", ok=-worst <= 0.8)
    pm = P["panels"][("main", "a")]
    for v, dd in (("z500", 9), ("msl", 11)):
        r = MAIN_ROWS.index(v)
        pos = pm["values"][r] > 0
        last_all = int(L[np.argmin(pos)] - 12) if not pos.all() else int(L[-1])
        add(f"gc_{v}_days", f"CRPS lower than GenCast at every lead from 12 h for {v} up to", f"{dd * 24} h",
            f"{last_all} h")
        sg = pm["significant"][r] & pos
        last_sig = int(L[np.argmin(sg)] - 12) if not sg.all() else int(L[-1])
        add(f"gc_{v}_sig", f"{v}: CRPS significantly lower than GenCast at every lead up to",
            {"z500": "84 h (3.5 d)", "msl": "96 h (4 d)"}[v], f"{last_sig} h",
            ok=last_sig == {"z500": 84, "msl": 96}[v])
    v12 = np.concatenate([pm["values"][MAIN_ROWS.index(v)][:3] for v in ("z500", "msl")])
    add("gc_12_36", "z500 and msl CRPS improvement over GenCast at 12-36 h", "3.5-4.7 %",
        f"{v12.min():.2f}-{v12.max():.2f} %", ok=round(v12.min(), 1) == 3.5 and round(v12.max(), 1) == 4.7)
    pc = P["panels"][("main", "c")]
    i5, i15 = leads.index(120), leads.index(360)
    v5, v15 = pc["values"][:, i5], pc["values"][:, i15]
    t2 = MAIN_ROWS.index("t2m")
    add("ens_5d_15d", "CRPS improvement over IFS ENS (Fig. 4 variables) at 5 d / 15 d; t2m",
        "6-20 % / 1-6 %; t2m 20 %, 6 %",
        f"{v5.min():.1f}-{v5.max():.1f} % / {v15.min():.1f}-{v15.max():.1f} %; t2m {v5[t2]:.1f} %, {v15[t2]:.1f} %",
        ok=(round(v5.min()) == 6 and round(v5.max()) == 20 and round(v15.min()) == 1 and round(v15.max()) == 6
            and int(np.argmax(v5)) == t2 and int(np.argmax(v15)) == t2))
    pcf = P["panels"][("full", "c")]
    tp_c = pcf["values"][tpi]
    fin = np.isfinite(tp_c)
    allsig = bool(((tp_c[fin] > 0) & pcf["significant"][tpi][fin]).all())
    add("tp_ens", "tp24h CRPS vs IFS ENS: significantly lower at every lead; improvement at 1 d / 15 d",
        "every lead; 21 % / 3 %", f"{'every lead' if allsig else 'not every lead'}; {tp_c[leads.index(24)]:.1f} % / "
        f"{tp_c[i15]:.1f} %", ok=allsig and round(tp_c[leads.index(24)]) == 21 and round(tp_c[i15]) == 3)
    tp_a = pa["values"][tpi]
    after = L >= 144
    add("tp_gc_6d", "tp24h CRPS lower than GenCast at every lead from day 6", "every lead from 144 h",
        f"{int((tp_a[after] > 0).sum())}/{int(after.sum())} leads", ok=bool((tp_a[after] > 0).all()))
    nb, ns, first = _analysis_counts(P["res"], wdir, leads)
    add("ana_570", "CRPS lower than IFS ENS verified against its own analysis", "560/570 (98.2%)",
        f"{nb}/{ns} ({100 * nb / ns:.1f}%)")
    add("ana_36h", "... on every scored combination from", "36 h", f"{first} h")
    ys = P["fig1"]
    labels = {"analysis": ("Fig. 1a CRPS skill vs IFS ENS verified against its own analysis: Astra / GenCast",
                           "Fig. 1b RMSE skill vs IFS ENS verified against its own analysis: Astra / GenCast"),
              "era5": ("CRPS skill as in Fig. 1a, IFS ENS verified against ERA5 (Results, Methods): Astra / GenCast",
                       None)}
    for key, paper in (("analysis", ("7.1", "6.1")), ("era5", ("8.3", None))):
        y = ys[key]
        add(f"fig1_crps_{key}", labels[key][0], f"{paper[0]} / {paper[0]}",
            f"{y['crps']['ours']:.4f} / {y['crps']['gencast']:.4f}",
            ok=f"{y['crps']['ours']:.1f}" == paper[0] and f"{y['crps']['gencast']:.1f}" == paper[0])
        if paper[1]:
            add(f"fig1_rmse_{key}", labels[key][1],
                f"{paper[1]} / {paper[1]}", f"{y['urmse']['ours']:.4f} / {y['urmse']['gencast']:.4f}",
                ok=f"{y['urmse']['ours']:.1f}" == paper[1] and f"{y['urmse']['gencast']:.1f}" == paper[1])
    mn = P["ours"]["means"]
    cs = mn["crps_skill"]
    d1, d5, d10 = (cs["ours"][leads.index(h)] for h in (24, 120, 240))
    add("curve_skill", "Mean CRPS skill (8 variables) vs IFS ENS (ERA5) at 1, 5, 10 d", "21 / 9.6 / 4.5 %",
        f"{d1:.2f} / {d5:.2f} / {d10:.2f} %", ok=round(d1) == 21 and round(d5, 1) == 9.6 and round(d10, 1) == 4.5)
    diff = np.asarray(cs["ours"]) - np.asarray(cs["gencast"])
    m36 = L >= 36
    dd = diff[m36]
    others = dd[dd <= 0]
    add("curve_vs_gc", "From 36 h: leads where Astra's mean CRPS skill exceeds GenCast's; max margin; others within",
        "26/28; 0.6 pp; 0.01 pp", f"{int((dd > 0).sum())}/{int(m36.sum())}; {dd.max():.2f} pp; "
        f"{np.abs(others).max() if len(others) else 0:.3f} pp",
        ok=int((dd > 0).sum()) == 26 and round(dd.max(), 1) == 0.6 and (np.abs(others) <= 0.01).all())
    rs = mn["rmse_skill"]
    det = [k for k in FI.DET_MODELS if k in rs]
    ours_r = np.asarray(rs["ours"], float)
    beats = all(np.all(ours_r[np.isfinite(np.asarray(rs[k], float))]
                       > np.asarray(rs[k], float)[np.isfinite(np.asarray(rs[k], float))]) for k in det)
    add("curve_rmse_all", "Mean RMSE skill above every deterministic model at every lead", "yes",
        "yes" if beats else "no")
    for h, paper_m, paper_d in ((24, "graphcast", 2), (72, "excarta", 4), (240, "baguan", 11)):
        li = leads.index(h)
        best = max(det, key=lambda k: -np.inf if rs[k][li] is None else rs[k][li])
        gap = rs["ours"][li] - rs[best][li]
        add(f"curve_rmse_{h}", f"Mean RMSE skill margin over the best deterministic model at {h} h",
            f"{paper_d} pp ({FI.DET_MODELS[paper_m]})", f"{gap:.2f} pp ({FI.DET_MODELS[best]})",
            ok=best == paper_m and round(gap) == paper_d)
    li = leads.index(360)
    gap = rs["ours"][li] - rs["fuxi"][li]
    add("curve_rmse_fuxi", "Mean RMSE skill margin over FuXi at 15 d", "17 pp", f"{gap:.2f} pp", ok=round(gap) == 17)
    ss = mn["ssr"]
    o, g, e = (np.asarray(ss[k], float) for k in ("ours", "gencast", "ens"))
    add("ssr_15d", "Mean spread-skill ratio (8 variables) at 15 d", "1.00", f"{o[-1]:.2f}")
    cg = np.abs(o - 1) < np.abs(g - 1)
    ce = np.abs(o - 1) < np.abs(e - 1)
    fg = next((int(L[i]) for i in range(len(L)) if cg[i:].all()), None)
    fe = next((int(L[i]) for i in range(len(L)) if ce[i:].all()), None)
    add("ssr_vs", "Spread-skill ratio closer to 1 than GenCast / IFS ENS at every lead from", "180 h / 216 h",
        f"{fg} h / {fe} h")
    s15 = [P["ours"]["ssr"][v][leads.index(360)] for v in FI.CURVE_ROWS]
    add("ssr_15d_per_var", "ED Fig. curves c: spread-skill ratio at 15 d per variable (z500, t850, q700, msl, t2m, "
                           "ws10, tp24h)", "0.97-1.03", f"{min(s15):.2f}-{max(s15):.2f}")
    meta = P["sc"]["meta"]
    add("kappa", "Widening factor kappa (median over 16 paired tests)", "1.09 (16 tests)",
        f"{meta['unpaired_kappa']:.2f} ({meta['kappa_from']['cells']} tests)")
    add("ens_states", "IFS ENS per-initialisation scores reproduce the published 24 h scores to", "8e-8",
        f"{P['ens']['worst_rel']:.1e}", ok=P["ens"]["worst_rel"] < 8.5e-8)
    ident = P["wb2"]["ssr_identity"]["ens_states_24h"]
    ds = FI.open_wb2(wdir / "ens_vs_era5_1440x721_2020.nc").sel(lead_time=24)
    dev = 0.0
    for v, x in ident.items():
        w, lev = FI.wb2_name(v)
        a = ds[f"unbiased_spread_skill.{w}"]
        pub = float((a.sel(level=lev) if lev is not None else a).values)
        dev = max(dev, abs(x["states_spread_over_skill"] * pub - 1))
    add("ssr_identity", "IFS ENS 24 h: per-initialisation spread/skill x published unbiased_spread_skill - 1 "
                        "(published value at full precision)", "<= 2e-8", f"{dev:.1e}", ok=dev <= 2e-8)
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, epilog="Exit status 0 when every number matches.")
    ap.add_argument("--data", default=str(DATA), help="directory of the stored score files (default: data/)")
    ap.add_argument("--json", default="", help="also write the rows to this JSON file")
    a = ap.parse_args(argv)
    P = compute(a.data)
    rows = []
    try:
        headline(P, a.data, out=rows)
    except Exception as e:                          # the rows computed so far, then the failure
        rows.append(dict(id="error", statement=f"stopped: {type(e).__name__}: {e}", paper="", reproduced="",
                         match=False))
    w = max(len(r["statement"]) for r in rows)
    for r in rows:
        print(f"{'OK  ' if r['match'] else 'DIFF'} {r['statement']:{w}s}  paper {r['paper']:>32s}  "
              f"reproduced {r['reproduced']}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1) + "\n")
    return 0 if all(r["match"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
