"""The four per-cell statistics (D7, D-CB-15) and the numerically safe Bernoulli KL (D-CB-6).

From the probe's `logp [N, Lc+1, L]` every cell `(j, q)` with `j < q` and
`q − j <= M` (window positions) has, per particle `l`, the log-probability of
the observed effect with the cause noised (`lp`, row `j`) and observed (`lq`,
row `j+1`). In float64:

    kl_mean_bf = mean_l KL(p_l || q_l)     per-particle KL, base (cause noised) first, then mean
    kl_mean_df = mean_l KL(q_l || p_l)     per-particle KL, cause observed first
    mean_kl_bf = KL(p_bar || q_bar)        mean over particles in probability space, then KL
    mean_kl_df = KL(q_bar || p_bar)

`KL(a || b)` is the Bernoulli divergence computed from log-probabilities with
`log(1 − p) = log1mexp(log p)`, `log p` clamped at `−1e-12` (clamped cells
counted). The published float32 `clamp(p, ε, 1 − ε)` — a no-op at the upper
end for ε = 1e-9, so a saturated particle turns the cell into 0 (NaN) or the
float maximum (+inf) — is reproduced on the same inputs in the same pass as
`compat`, and the cells it corrupts are counted by direction (PRD scenario
15). `--divergence` selects which family is emitted; the counts are always
recorded.
"""
from __future__ import annotations

import math

import torch

from .constants import STATISTICS

LOG_CLAMP = -1e-12                    # log p <= this, so 1 - p stays representable in float64
LN2 = math.log(2.0)


def log1mexp(x):
    """`log(1 − exp(x))` for `x <= 0`, accurate near both ends."""
    return torch.where(x > -LN2, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def bernoulli_kl_from_log(la, lb):
    """`KL(Bern(a) || Bern(b))` from `log a`, `log b` (float64, clamped by the caller)."""
    a = torch.exp(la)
    return a * (la - lb) + (1.0 - a) * (log1mexp(la) - log1mexp(lb))


def bernoulli_kl_compat(a, b):
    """The published float32 formula on clamped probabilities: NaN at a = 1, +inf at b = 1."""
    return a * torch.log(a / b) + (1.0 - a) * torch.log((1.0 - a) / (1.0 - b))


def cell_statistics(logp, c, max_lag, compat_eps):
    """`logp [N, Lc+1, L]` → dict of `[Lc, Lc]` float64 tensors (NaN outside the strict-upper
    lag-bounded band) for the four safe statistics, the four compat statistics, and counts."""
    N, R, L = logp.shape
    Lc = L - c
    lp = logp[:, :Lc, c:].double()            # [N, j, q]: row j  = cause j noised
    lq = logp[:, 1:Lc + 1, c:].double()       # [N, j, q]: row j+1 = cause j observed
    j = torch.arange(Lc, device=logp.device)
    band = (j[None, :] > j[:, None]) & ((j[None, :] - j[:, None]) <= max_lag)   # [j, q]
    # --- safe family ------------------------------------------------------------------------
    clamped = ((lp > LOG_CLAMP) | (lq > LOG_CLAMP)) & band[None]
    lp_c = torch.clamp(lp, max=LOG_CLAMP)
    lq_c = torch.clamp(lq, max=LOG_CLAMP)
    p, q = torch.exp(lp_c), torch.exp(lq_c)
    kl_bf = bernoulli_kl_from_log(lp_c, lq_c)                  # KL(p || q) per particle
    kl_df = bernoulli_kl_from_log(lq_c, lp_c)
    p_bar, q_bar = p.mean(0), q.mean(0)
    lpb, lqb = torch.clamp(torch.log(p_bar), max=LOG_CLAMP), torch.clamp(torch.log(q_bar), max=LOG_CLAMP)
    safe = {
        "kl_mean_bf": kl_bf.mean(0), "kl_mean_df": kl_df.mean(0),
        "mean_kl_bf": bernoulli_kl_from_log(lpb, lqb), "mean_kl_df": bernoulli_kl_from_log(lqb, lpb),
    }
    # --- published behaviour on the same inputs ---------------------------------------------------
    eps = float(compat_eps)
    p32 = torch.clamp(torch.exp(lp).float(), eps, 1.0 - eps)
    q32 = torch.clamp(torch.exp(lq).float(), eps, 1.0 - eps)
    compat_raw = {
        "kl_mean_bf": bernoulli_kl_compat(p32, q32).mean(0), "kl_mean_df": bernoulli_kl_compat(q32, p32).mean(0),
        "mean_kl_bf": bernoulli_kl_compat(p32.mean(0), q32.mean(0)), "mean_kl_df": bernoulli_kl_compat(q32.mean(0), p32.mean(0)),
    }
    counts = {}
    compat = {}
    fmax = torch.finfo(torch.float32).max
    for k, v in compat_raw.items():
        nan = torch.isnan(v) & band
        inf = torch.isinf(v) & band
        counts[k] = {"n_cell_to_zero": int(nan.sum()), "n_cell_to_fmax": int(inf.sum())}
        compat[k] = torch.nan_to_num(v, nan=0.0, posinf=fmax, neginf=0.0).double()
    nan_mask = torch.where(band, 0.0, float("nan")).double()
    out = {
        "safe": {k: v + nan_mask for k, v in safe.items()},
        "compat": {k: v + nan_mask for k, v in compat.items()},
        "band": band,
        "n_cells": int(band.sum()),
        "n_clamped_cells": int(clamped.any(0).sum()),
        "compat_counts": counts,
    }
    return out


def stats_for(cells, divergence):
    """The family `--divergence` selects, as `{stat: [Lc, Lc]}`."""
    if divergence not in ("log1mexp", "compat"):
        raise ValueError(f"divergence must be log1mexp or compat, got {divergence!r}")
    fam = cells["safe"] if divergence == "log1mexp" else cells["compat"]
    return {k: fam[k] for k in STATISTICS}


__all__ = ["log1mexp", "bernoulli_kl_from_log", "bernoulli_kl_compat", "cell_statistics", "stats_for", "LOG_CLAMP"]
