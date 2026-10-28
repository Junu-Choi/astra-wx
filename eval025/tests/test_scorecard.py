"""Year scores, comparison with the WB2 result files and the significance tests, on the stored scores."""
import numpy as np

from astra_eval025 import scorecard as SC
from conftest import DATA, TEXT_KEYS, as_json, diff, golden


def test_inputs_complete():
    D = SC.load_scores(DATA / "astra_2020_025_per_init_scores.npz")
    cov = SC.coverage(D)
    assert cov["complete"] and cov["n"] == 732 and cov["n_00z"] == cov["n_12z"] == 366
    assert int(D["M"]) == 50 and list(D["leads_hours"]) == list(range(12, 361, 12))
    tp = list(D["names"]).index("tp24h")
    assert np.isnan(D["crps"][:, 0, tp]).all() and np.isfinite(D["crps"][:, 1:, tp]).all()


def test_ifs_ens_per_init_scores_reproduce_published(P):
    ens = P["ens"]
    assert ens["ok"] and ens["n"] == 732 and ens["worst_rel"] < 1e-7


def test_build_matches_expected(P):
    g = golden("scorecard_build.json.gz")
    res = as_json({k: v for k, v in P["res"].items()})
    assert not diff(res["cells"], g["cells"], skip=TEXT_KEYS)
    assert not diff(res["summary"], g["summary"])
    assert not diff(res["summary_paper_rows"], g["summary_paper_rows"])
    assert not diff({k: v for k, v in res["meta"].items() if k not in TEXT_KEYS}, g["meta"], skip=TEXT_KEYS)
    assert not diff(res["ours"], g["ours"])


def test_kappa_and_paired_tests(P):
    meta = P["res"]["meta"]
    assert meta["kappa_from"]["cells"] == 16
    assert round(meta["unpaired_kappa"], 2) == 1.09
    paired = [c for c in P["res"]["cells"] if c.get("paired")]
    assert len(paired) == 16
    assert {c["var"] for c in paired} == set(SC.ENS_STATE_VARS)
    assert all(c["ref"] == "ens" and c["lead_h"] == 24 for c in paired)
    for c in P["res"]["cells"]:
        if "ci_widened" in c:
            r, (lo, hi), (clo, chi) = c["ratio"], c["ci"], c["ci_widened"]
            assert np.isclose(r - clo, meta["unpaired_kappa"] * (r - lo))
            assert np.isclose(chi - r, meta["unpaired_kappa"] * (hi - r))


def test_summary_paper_rows_equal_panel_counts(P):
    s = P["res"]["summary_paper_rows"]
    for k, key in (("a", "gencast.crps"), ("b", "gencast.urmse"), ("c", "ens.crps"), ("d", "ens.urmse")):
        tot = [sum(d[x] for d in s[key].values())
               for x in ("n", "wins", "sig_wins", "sig_losses")]
        c = P["panels"][("full", k)]["counts"]["all_leads"]
        assert tot == [c["cells"], c["better"], c["better_and_significant"], c["reference_better_and_significant"]]


def test_cbb_sums_equal_explicit_resampling():
    rng = np.random.default_rng(3)
    N, block, B = 50, 7, 20
    X = rng.normal(size=(N, 4))
    st = SC.cbb_starts(N, block, B, 11)
    got = SC.cbb_sums(X, st, block)
    k = st.shape[1]
    lens = [block] * (k - 1) + [N - (k - 1) * block]
    for b in range(B):
        idx = np.concatenate([(s + np.arange(n)) % N for s, n in zip(st[b], lens)])
        assert len(idx) == N
        np.testing.assert_allclose(got[b], X[idx].sum(0), rtol=1e-12, atol=1e-12)
