"""Edge selection (D9): an edge is present iff its aggregated score exceeds τ strictly.

`scores` is the table `project.PairAccumulator.result()` returns (or
`read_scores_npz`): one row per `(src, dst)` token-id pair with `max_<stat>`,
`mean_<stat>`, `count` and `lag_max_<stat>`. The threshold-free axes read the
full ranking, so every scored pair is also written with its score.
"""
from __future__ import annotations

import numpy as np

from .constants import AGGREGATIONS, STATISTICS


def column(scores, stat, agg):
    if stat not in STATISTICS:
        raise ValueError(f"unknown statistic {stat!r}")
    if agg not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {agg!r}")
    return np.asarray(scores[f"{agg}_{stat}"], dtype=np.float64)


def select_edges(scores, stat, agg, tau):
    """`(src, dst, score)` arrays of the pairs with `score > tau`."""
    s = column(scores, stat, agg)
    hit = s > tau
    return scores["src"][hit], scores["dst"][hit], s[hit]


def ranking(scores, stat, agg):
    """Every scored pair with its score (for AUROC / AP)."""
    s = column(scores, stat, agg)
    return scores["src"], scores["dst"], s


def per_lag_ranking(scores, stat, lag):
    """Pairs whose per-lag max at `lag` (1-based) exists, with that value."""
    lm = np.asarray(scores[f"lag_max_{stat}"], dtype=np.float64)
    col = lm[:, lag - 1]
    ok = np.isfinite(col)
    return scores["src"][ok], scores["dst"][ok], col[ok]


def f1_at(edges, truth):
    """Directed precision / recall / F1 of one edge set against a truth set (gate use only —
    benchmark corpora are scored by the benchmark's scorer, never here)."""
    tp = len(edges & truth)
    fp = len(edges - truth)
    fn = len(truth - edges)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": (2 * p * r / (p + r)) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


__all__ = ["column", "select_edges", "ranking", "per_lag_ranking", "f1_at"]
