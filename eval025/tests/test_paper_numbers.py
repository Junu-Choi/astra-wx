"""The 0.25 deg score-based numbers of the paper that reproduce.headline recomputes from the stored scores."""
from astra_eval025 import reproduce
from conftest import DATA


def test_headline_numbers(P):
    rows = reproduce.headline(P, DATA)
    bad = [(r["id"], r["paper"], r["reproduced"]) for r in rows if not r["match"]]
    assert not bad, bad
    ids = {r["id"]: r for r in rows}
    assert ids["gc_crps_600"]["reproduced"] == "441/600 (73.5%)"
    assert ids["gc_crps_500"]["reproduced"] == "406/500 (81%)"
    assert ids["ens_crps_600"]["reproduced"] == "600/600"
    assert ids["ens_rmse_600"]["reproduced"] == "600/600"
    assert ids["ana_570"]["reproduced"] == "560/570 (98.2%)"
