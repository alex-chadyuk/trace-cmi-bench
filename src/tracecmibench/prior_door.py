"""The separately named door to the shipped structural prior (PRD non-negotiable 2,
scenario 19, External Surfaces).

trace-bench ships `topology/prior.json` — the deployment call topology in the
outcome-propagation direction, the second argument of the lab's structural-
prior contract — but not under a method-readable prefix, so the benchmark's
accessor refuses it. It is not ground truth (an operator possesses it before
any incident), and the prior-informed arm reads it here, through a door that
permits exactly that one path and nothing else, logs the read, and returns
the list of paths read for the run record (`results.json.prior_reads`).

Only `prior/*` arms may import this module; `tests/test_prior_door_only_path.py`
asserts no other module does.
"""
from __future__ import annotations

import json
from pathlib import Path

from .constants import PRIOR_RELATIVE_PATH
from .log import log


class NotPriorReadable(PermissionError):
    pass


def open_prior_path(corpus_dir, relative_path):
    """Open one corpus path through the prior door; refuses every path but the prior."""
    rel = str(relative_path).replace("\\", "/").lstrip("./")
    if rel != PRIOR_RELATIVE_PATH:
        raise NotPriorReadable(f"{rel!r} is not the shipped prior; this door opens only {PRIOR_RELATIVE_PATH!r}")
    corpus_dir = Path(corpus_dir).resolve()
    target = (corpus_dir / rel).resolve()
    if target.parent != corpus_dir / "topology" or target.name != "prior.json":
        raise NotPriorReadable(f"{relative_path!r} escapes the corpus directory")
    return open(target, "r", encoding="utf-8")


def open_prior(corpus_dir):
    """The prior JSON and the list of corpus paths read outside the method-readable set."""
    with open_prior_path(corpus_dir, PRIOR_RELATIVE_PATH) as f:
        prior = json.load(f)
    reads = [PRIOR_RELATIVE_PATH]
    log({"event": "prior_read", "path": PRIOR_RELATIVE_PATH, "n_edges": len(prior.get("edges", [])),
         "n_columns": len(prior.get("columns", []))})
    return prior, reads
