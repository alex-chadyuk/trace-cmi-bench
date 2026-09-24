"""D-CB-6 and D-CB-15: the Bernoulli KL from log-probabilities is exact, the
published float32 clamp corrupts saturated cells in the counted directions,
and the four statistics relate as their definitions say."""
import math

import numpy as np
import pytest
import torch

from tracecmibench.statistics import (
    LOG_CLAMP, bernoulli_kl_compat, bernoulli_kl_from_log, cell_statistics, log1mexp, stats_for,
)


def _kl(a, b):
    return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))


def test_kl_analytic():
    rng = np.random.default_rng(0)
    a = rng.uniform(1e-6, 1 - 1e-6, 200)
    b = rng.uniform(1e-6, 1 - 1e-6, 200)
    got = bernoulli_kl_from_log(torch.log(torch.tensor(a)), torch.log(torch.tensor(b))).numpy()
    want = np.array([_kl(x, y) for x, y in zip(a, b)])
    assert np.allclose(got, want, rtol=1e-10, atol=1e-12)
    assert (got >= -1e-15).all()
    la = torch.log(torch.tensor(a))
    assert torch.allclose(bernoulli_kl_from_log(la, la), torch.zeros_like(la), atol=1e-15)
    x = torch.tensor([-1e-12, -0.1, -0.7, -5.0, -40.0], dtype=torch.float64)
    assert torch.allclose(log1mexp(x), torch.log(1 - torch.exp(x)) if False else torch.log1p(-torch.exp(x)), atol=1e-12)
    assert abs(float(log1mexp(torch.tensor(-1e-12, dtype=torch.float64))) - math.log(1e-12)) < 1e-6


def _logp(N, Lc, c, L, rng, saturate=None):
    lp = torch.log(torch.tensor(rng.uniform(0.05, 0.95, size=(N, Lc + 1, L))))
    if saturate:
        for (l, r, t) in saturate:
            lp[l, r, t] = 0.0                                  # probability exactly 1
    return lp


def test_log1mexp_vs_compat_counts():
    rng = np.random.default_rng(1)
    c, L, N = 2, 8, 4
    Lc = L - c
    # row 2 at position 5 (window q = 3) saturates in particle 0. Row 2 is the cause-OBSERVED row of cause j = 1
    # (cell (1, 3): q = 1) and the cause-NOISED row of cause j = 2 (cell (2, 3): p = 1).
    cells = cell_statistics(_logp(N, Lc, c, L, rng, saturate=[(0, 2, 5)]), c, max_lag=Lc, compat_eps=1e-9)
    safe = cells["safe"]
    for k, m in safe.items():
        band = cells["band"]
        assert torch.isfinite(m[band]).all(), k
    assert cells["n_clamped_cells"] == 2
    cc = cells["compat_counts"]
    # KL(a || b): b = 1 -> log(1/(1-b)) = +inf (fmax); a = 1 -> 0 * log 0 = NaN (zero)
    assert cc["kl_mean_bf"] == {"n_cell_to_zero": 1, "n_cell_to_fmax": 1}     # (2,3) p=1 -> NaN; (1,3) q=1 -> inf
    assert cc["kl_mean_df"] == {"n_cell_to_zero": 1, "n_cell_to_fmax": 1}     # (1,3) q=1 -> NaN; (2,3) p=1 -> inf
    # the mean over particles of clamped probabilities stays inside (0, 1): no corruption in mean-then-KL
    assert cc["mean_kl_bf"] == {"n_cell_to_zero": 0, "n_cell_to_fmax": 0}
    compat = cells["compat"]
    fmax = torch.finfo(torch.float32).max
    assert compat["kl_mean_bf"][1, 3] == fmax and compat["kl_mean_df"][1, 3] == 0.0
    assert compat["kl_mean_bf"][2, 3] == 0.0 and compat["kl_mean_df"][2, 3] == fmax
    # the same inputs, safe family: finite and positive at those cells
    assert 0 < safe["kl_mean_df"][1, 3] < 1e3 and 0 < safe["kl_mean_bf"][2, 3] < 1e3
    # the published clamp really is a no-op at the top in float32
    assert bernoulli_kl_compat(torch.tensor([1.0]).clamp(1e-9, 1 - 1e-9), torch.tensor([0.5])).isnan().all()


def test_four_stats_orderings():
    rng = np.random.default_rng(2)
    c, L = 2, 9
    Lc = L - c
    one = cell_statistics(_logp(1, Lc, c, L, rng), c, Lc, 1e-9)["safe"]
    band = torch.triu(torch.ones(Lc, Lc, dtype=torch.bool), 1)
    assert torch.allclose(one["kl_mean_bf"][band], one["mean_kl_bf"][band])         # N = 1: mean-then-KL = KL-then-mean
    assert torch.allclose(one["kl_mean_df"][band], one["mean_kl_df"][band])
    assert not torch.allclose(one["kl_mean_bf"][band], one["kl_mean_df"][band])      # argument order matters
    many = cell_statistics(_logp(32, Lc, c, L, rng), c, Lc, 1e-9)["safe"]
    assert (many["kl_mean_bf"][band] >= many["mean_kl_bf"][band] - 1e-12).all()      # Jensen: KL is jointly convex
    assert (many["kl_mean_df"][band] >= many["mean_kl_df"][band] - 1e-12).all()
    assert torch.isnan(many["kl_mean_bf"][~band]).all()
    fam = stats_for(cell_statistics(_logp(2, Lc, c, L, rng), c, Lc, 1e-9), "compat")
    assert set(fam) == {"kl_mean_bf", "kl_mean_df", "mean_kl_bf", "mean_kl_df"}
    with pytest.raises(ValueError):
        stats_for(many, "other")


def test_lag_bound_masks_cells():
    rng = np.random.default_rng(3)
    c, L = 1, 8
    Lc = L - c
    cells = cell_statistics(_logp(3, Lc, c, L, rng), c, max_lag=2, compat_eps=1e-9)
    j, q = np.meshgrid(np.arange(Lc), np.arange(Lc), indexing="ij")
    inside = (q > j) & (q - j <= 2)
    m = cells["safe"]["kl_mean_bf"].numpy()
    assert np.isfinite(m[inside]).all() and np.isnan(m[~inside]).all()
    assert cells["n_cells"] == inside.sum()
    assert LOG_CLAMP < 0
