"""Per-(initialisation, lead time) scorer on a stored synthetic fixture (JAX on CPU): equal to the reference outputs
in data/expected within float32 rounding, and to the float64 pairwise reference within float32 precision."""
import numpy as np

from astra_eval025 import scorer as S
from astra_eval025 import weights as W
from conftest import EXPECTED


def test_area_weights():
    w = np.asarray(W.latitude_weights(721), np.float64)
    phi = np.deg2rad(np.linspace(-90, 90, 721))
    b = np.concatenate([[-np.pi / 2], 0.5 * (phi[1:] + phi[:-1]), [np.pi / 2]])
    e = np.diff(np.sin(b))
    np.testing.assert_allclose(w, e / e.mean(), rtol=1e-6)
    assert W.area_weights(721, 1440).shape == (721 * 1440,)


def test_block_scorer_fixture():
    import jax.numpy as jnp
    z = np.load(EXPECTED / "scorer_fixture.npz")
    members, truth, mean, std = z["members"], z["truth"], z["mean"], z["std"]
    names, lat, lon, G = [str(n) for n in z["names"]], int(z["lat"]), int(z["lon"]), int(z["G"])
    rings = list(z["rings"])
    M, P = members.shape[:2]
    nb = M // G
    Qb = names.index("tp6h")
    w = W.area_weights(lat, lon)
    wsi = S.wind_index(names)
    md = S.make_members_derived(wsi, mean, std)
    ext = [jnp.asarray(S.tp24h_members(list(r))) for r in rings]
    blk = [jnp.asarray(members[k * G:(k + 1) * G]) for k in range(nb)]
    views = [(lambda x, e: x, mean, std), (lambda x, e: md(x, e), np.zeros(5, np.float32), np.ones(5, np.float32))]
    score = S.make_block_scorer(M=M, P_points=P, weights=w, views=views)
    td = np.asarray(S.truth_derived(truth, z["tp24_truth"], wsi, P))
    res = score(lambda k: blk[k], nb, [truth, td], [np.zeros((P, len(names)), np.float32),
                                                    np.zeros((P, 5), np.float32)], extra=ext)
    for v, vr in enumerate(res):
        for i, x in enumerate(vr):
            np.testing.assert_allclose(np.asarray(x), z[f"out{v}_{i}"], rtol=1e-6, atol=1e-6 if i == 1 else 0,
                                       err_msg=f"view {v}, output {i}")
    ref = S.numpy_lead_scores(members[:, :, :Qb], truth[:, :Qb], np.zeros((P, Qb)), w, mean[:Qb], std[:Qb])
    for k, i in (("mse", 0), ("spread2", 2), ("crps", 3)):
        np.testing.assert_allclose(ref[k], z[f"ref_{k}"], rtol=1e-12)
        np.testing.assert_allclose(np.asarray(res[0][i])[:Qb], ref[k], rtol=2e-5)
