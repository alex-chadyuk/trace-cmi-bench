"""D-CB-13: the exact truth is positional, within lag h, in the paper's KL
order, and consistent with the world: a lag the mechanism cannot carry gets no
edge, a strong direct interaction does."""
import numpy as np

from tracecmibench import generator as gen


def test_pairs_are_within_the_lag_window_and_kl_is_nonnegative():
    scm = gen.SCM(40, history=3, sparsity=0.5, w_scale=3.0, decay_rate=1.0, embed_dim=4, hidden=8, seed=0)
    x, _ = scm.sample(8, 10, np.random.default_rng(0))
    t = scm.truth(x, 0.05, 10, np.random.default_rng(1))
    lag = t["j"].astype(int) - t["i"].astype(int)
    assert lag.min() == 1 and lag.max() == 3 and (t["i"] >= 0).all() and (t["j"] <= 9).all()
    assert (t["kl"] >= -1e-12).all() and t["edge"].dtype == bool and (t["edge"] == (t["kl"] > 0.05)).all()
    # every (seq, i, j) with 1 <= j - i <= h appears exactly once
    expected = sum(min(3, j) for j in range(1, 10)) * 8
    assert len(t["kl"]) == expected and len({(s, i, j) for s, i, j in zip(t["seq"], t["i"], t["j"])}) == expected


def test_history_one_world_has_only_lag_one_edges():
    scm = gen.SCM(40, history=1, sparsity=0.5, w_scale=4.0, decay_rate=1.0, embed_dim=4, hidden=8, seed=2)
    x, _ = scm.sample(16, 8, np.random.default_rng(0))
    t = scm.truth(x, 0.05, 10, np.random.default_rng(1))
    assert ((t["j"] - t["i"]) == 1).all() and t["edge"].any()


def test_decay_makes_lag_one_dominant():
    scm = gen.SCM(60, history=4, sparsity=0.8, w_scale=4.0, decay_rate=1.0, embed_dim=4, hidden=8, seed=3)
    x, _ = scm.sample(32, 16, np.random.default_rng(0))
    t = scm.truth(x, 0.05, 10, np.random.default_rng(1))
    lag = (t["j"] - t["i"]).astype(int)
    rate = {k: t["edge"][lag == k].mean() for k in range(1, 5)}
    assert rate[1] > rate[2] >= rate[4]


def test_kl_is_paper_order():
    """KL(p || q) with p the factual conditional: hand-computed on one pair."""
    scm = gen.SCM(6, history=1, sparsity=0.0, w_scale=2.0, decay_rate=1.0, embed_dim=2, hidden=3, seed=4)
    x = np.array([[1, 2]])
    rng = np.random.default_rng(0)
    t = scm.truth(x, 0.0, 10, rng)
    # recompute: context of position 1 is [1]; replace it by u ~ U(6), ten draws with the same stream
    rng2 = np.random.default_rng(0)
    base = scm.log_softmax(scm.logits(np.array([[1]])))[0]
    u = rng2.integers(0, 6, size=(1, 10)).ravel()
    q = scm.log_softmax(scm.logits(u[:, None]))
    kl = (np.exp(base)[None, :] * (base[None, :] - q)).sum(-1).mean()
    assert abs(t["kl"][0] - kl) < 1e-12
