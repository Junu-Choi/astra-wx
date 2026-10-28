"""Figure inputs, scorecard panels and Fig. 1 skills: equal to the reference outputs in data/expected within the
tolerance of conftest.diff."""
from conftest import as_json, diff, golden


def _strip(o):
    o = as_json(o)
    for k in ("source", "paired"):
        o["significance"].pop(k, None)
    return o


def test_figure_inputs(P):
    g = golden("figure_inputs.json.gz")
    assert not diff(_strip(P["ours"]), _strip(g["ours"]))
    w = as_json(P["wb2"])
    ident = w["ssr_identity"].pop("ens_states_24h")
    assert not diff(w, g["wb2"])
    assert not diff(ident, g["ssr_identity_states"])


def test_panels(P):
    g = golden("panels.json.gz")
    for (variant, k), p in P["panels"].items():
        e = g[f"{variant}.{k}"]
        assert not diff(as_json([p["values"], p["significant"], p["paired"], p["counts"]]),
                        [e["values"], e["significant"], e["paired"], e["counts"]]), (variant, k)


def test_fig1_skills(P):
    g = golden("fig1_skills.json.gz")
    got = {k: {m: P["fig1"][k][m] for m in ("crps", "urmse")} for k in P["fig1"]}
    assert not diff(as_json(got), g)
