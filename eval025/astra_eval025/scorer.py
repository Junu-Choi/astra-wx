"""Per-(initialisation, lead time) ensemble scores on the 0.25 deg grid.

For every channel the scorer returns area-weighted means over the grid points of
  mse    squared error of the ensemble mean,
  acc    anomaly correlation of the ensemble mean against `clims` (not used in the paper; pass zeros),
  sp2    ensemble variance (M - 1 in the denominator),
  crps   fair CRPS, (1/M) sum_i <|x_i - y|> - 1/(M(M-1)) sum_{i<j} <|x_i - x_j|>,
with <.> the area-weighted mean (weights.area_weights).  Members are visited in blocks: block a, then every block
b > a.  Each view is (transform, chan_mean, chan_std): transform maps a member block to the scored (Bm, P, Qv)
field (members are standardised, x = (phys - chan_mean) / chan_std; for physical members use mean 0 and std 1).
Grid layout: P = lon * lat with latitude varying fastest.
"""
from __future__ import annotations

import numpy as np


def make_block_scorer(*, M: int, P_points: int, weights, views, compiler_options=None):
    """Returns score(block, nb, truths, clims, extra=None) -> [(mse, acc, sp2, crps) per view], each (Qv,)."""
    import jax
    import jax.numpy as jnp

    w_f64 = np.asarray(weights, np.float64)
    denom = float(w_f64.sum())
    wc = jnp.asarray(w_f64, jnp.float32)[:, None]
    Mf = float(M)
    bc = Mf / max(Mf - 1.0, 1.0)
    jit = lambda f: jax.jit(f, compiler_options=compiler_options)  # noqa: E731
    stats = []
    for tf, cm, cs in views:
        mu = jnp.asarray(cm, jnp.float32)
        sd = jnp.asarray(cs, jnp.float32)

        def _self(x, s1, s2, at, pr, tn, mu=mu, sd=sd):
            s1 = s1 + jnp.sum(x, axis=0)
            s2 = s2 + jnp.sum(jnp.square(x), axis=0)
            at = at + jnp.sum(wc * jnp.sum(jnp.abs(x - tn[None]), axis=0), axis=0)
            n = x.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    pr = pr + jnp.sum(wc * jnp.abs(x[i] - x[j]), axis=0)
            return s1, s2, at, pr

        def _cross(x, y, pr):
            return pr + jnp.sum(wc * jnp.sum(jnp.abs(x[:, None] - y[None]), axis=(0, 1)), axis=0)

        def _final(s1, s2, at, pr, truth, clim, mu=mu, sd=sd):
            mean_n = s1 / Mf
            var_n = jnp.clip(bc * (s2 / Mf - jnp.square(mean_n)), 0.0, None)
            mean_p = mean_n * sd + mu
            mse = jnp.sum(wc * jnp.square(mean_p - truth), axis=0) / denom
            pa, ta = mean_p - clim, truth - clim
            acc = jnp.sum(wc * pa * ta, axis=0) / (jnp.sqrt(jnp.sum(wc * pa * pa, axis=0)
                                                            * jnp.sum(wc * ta * ta, axis=0)) + 1e-12)
            sp2 = jnp.sum(wc * var_n * sd * sd, axis=0) / denom
            crps = (at / Mf - pr / (Mf * (Mf - 1.0))) / denom * sd
            return mse, acc, sp2, crps
        stats.append((tf, jit(_self), jit(_cross), jit(_final), mu, sd))

    def score(block, nb, truths, clims, extra=None):
        """block(k) -> (Bm, P, Q0) member block k < nb; extra[k] -> per-block input of the transforms (or None);
        truths / clims: per view, (P, Qv) in physical units."""
        acc_ = []
        for v, (tf, _s, _c, _f, mu, sd) in enumerate(stats):
            qv = int(np.shape(truths[v])[-1])
            z = jnp.zeros((P_points, qv), jnp.float32)
            acc_.append([z, z, jnp.zeros((qv,), jnp.float32), jnp.zeros((qv,), jnp.float32),
                         (jnp.asarray(truths[v], jnp.float32) - mu) / sd])
        for a in range(nb):
            xa = block(a)
            ea = None if extra is None else extra[a]
            ta = [st[0](xa, ea) for st in stats]
            for v, st in enumerate(stats):
                s1, s2, at, pr, tn = acc_[v]
                acc_[v][:4] = st[1](ta[v], s1, s2, at, pr, tn)
            for b in range(a + 1, nb):
                xb = block(b)
                for v, st in enumerate(stats):
                    acc_[v][3] = st[2](ta[v], st[0](xb, None if extra is None else extra[b]), acc_[v][3])
                del xb
            jax.block_until_ready([a_[3] for a_ in acc_])
            del xa, ta
        return [st[3](*acc_[v][:4], jnp.asarray(truths[v], jnp.float32), jnp.asarray(clims[v], jnp.float32))
                for v, st in enumerate(stats)]

    return score


def numpy_lead_scores(members, truth_phys, clim, weights, chan_mean, chan_std):
    """Float64 reference for one initialisation and lead time (pairwise sums, no blocking).

    members (M, P, Q) standardised; truth_phys, clim (P, Q) physical; weights (P,).
    Returns dict(mse, acc, spread2, crps), each (Q,), in physical units."""
    x = np.asarray(members, np.float64)
    y = np.asarray(truth_phys, np.float64)
    c = np.asarray(clim, np.float64)
    w = np.asarray(weights, np.float64)
    mu = np.asarray(chan_mean, np.float64)
    sd = np.asarray(chan_std, np.float64)
    M = x.shape[0]
    den = w.sum()
    wc = w[:, None]

    mean_phys = x.mean(0) * sd + mu
    var = x.var(0, ddof=1) * sd * sd
    mse = (wc * (mean_phys - y) ** 2).sum(0) / den
    pa, ta = mean_phys - c, y - c
    acc = ((wc * pa * ta).sum(0)
           / (np.sqrt((wc * pa * pa).sum(0) * (wc * ta * ta).sum(0)) + 1e-12))
    spread2 = (wc * var).sum(0) / den

    ynorm = (y - mu) / sd
    attr = np.zeros(x.shape[-1])
    for i in range(M):
        attr += (wc * np.abs(x[i] - ynorm)).sum(0) / den
    attr /= M
    pair = np.zeros(x.shape[-1])
    for i in range(M):
        for j in range(i + 1, M):
            pair += (wc * np.abs(x[i] - x[j])).sum(0) / den
    pair /= M * (M - 1)
    return dict(mse=mse, acc=acc, spread2=spread2, crps=(attr - pair) * sd)


# ------------------------------------------------------------------------------------------- derived variables
WS_PAIRS = {"ws10": ("u10", "v10"), "ws500": ("u500", "v500"), "ws700": ("u700", "v700"), "ws850": ("u850", "v850")}
DERIVED = ("ws10", "ws500", "ws700", "ws850", "tp24h")


def wind_index(names):
    """(4, 2) int32 channel indices of the (u, v) pairs of ws10, ws500, ws700, ws850."""
    names = list(names)
    return np.asarray([[names.index(WS_PAIRS[d][0]), names.index(WS_PAIRS[d][1])] for d in DERIVED[:4]], np.int32)


def make_members_derived(ws_idx, chan_mean, chan_std, compiler_options=None):
    """jit (Mb, P, Q) standardised members, (Mb, P) 24 h precipitation [m] -> (Mb, P, 5) physical ws10, ws500,
    ws700, ws850 (each member's own sqrt(u^2 + v^2)) and tp24h."""
    import jax
    import jax.numpy as jnp
    mu_d = jnp.asarray(np.asarray(chan_mean, np.float32))
    sd_d = jnp.asarray(np.asarray(chan_std, np.float32))
    wsi = jnp.asarray(ws_idx)

    def _members_derived(flat, tp24):
        u = flat[:, :, wsi[:, 0]] * sd_d[wsi[:, 0]] + mu_d[wsi[:, 0]]
        v = flat[:, :, wsi[:, 1]] * sd_d[wsi[:, 1]] + mu_d[wsi[:, 1]]
        return jnp.concatenate([jnp.sqrt(u * u + v * v), tp24[:, :, None]], axis=-1)
    return jax.jit(_members_derived, compiler_options=compiler_options)


def truth_derived(truth_phys, tp24_truth, ws_idx, P_points):
    """(P, Q) physical verifying state, (P,) 24 h precipitation [m] or None -> (P, 5) derived truth."""
    import jax.numpy as jnp
    tr = jnp.asarray(truth_phys)
    u = tr[:, ws_idx[:, 0]]
    v = tr[:, ws_idx[:, 1]]
    return jnp.concatenate([jnp.sqrt(u * u + v * v),
                            (jnp.asarray(tp24_truth) if tp24_truth is not None
                             else jnp.zeros((P_points,), jnp.float32))[:, None]], -1)


def tp24h_members(r):
    """Sum of the four 6 h accumulations ending at the lead, r = [(Mb, P) float32 x 4], oldest first."""
    return r[0] + r[1] + r[2] + r[3]


def tp24h_truth(tp6h_rows):
    """Verifying 24 h precipitation: the four ERA5 6 h accumulations ending at the valid time, (P,) float32 each."""
    return np.sum(list(tp6h_rows), axis=0).astype(np.float32)
