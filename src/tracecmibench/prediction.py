"""The prediction files the benchmark's scorer consumes (D9, D-CB-18, PRD scenario 8).

Two files per read, because the scorer treats every listed edge as present:

- `prediction-<grain>-<agg>.json` — the edges with score > τ, for the
  structural axes;
- `ranking-<grain>-<agg>.json` — every scored pair, for AUROC / AP; plus
  `ranking-<grain>-<agg>-lag<k>.json` per lag for per-lag recall.

Both carry `bidirected: []` and a top-level `structural_limitation` field
naming the assumption a faithful arm rests on; `annotate` fills in the count
of bidirected truth edges the arm could never have found. Unknown top-level
keys are ignored by the scorer.
"""
from __future__ import annotations

from .constants import STRUCTURAL_LIMITATION
from .record import write_json


def structural_limitation():
    return {"assumption": STRUCTURAL_LIMITATION, "truth_bidirected_edges": None,
            "note": "a faithful TRACE arm cannot emit a bidirected edge; its bidirected recall on a latent target is zero by construction"}


def edge_list(src, dst, score, vocab):
    return [{"src": vocab.token_string(int(u)), "dst": vocab.token_string(int(v)), "score": float(s)}
            for u, v, s in zip(src, dst, score)]


def prediction_document(src, dst, score, vocab, **meta):
    return {"directed": edge_list(src, dst, score, vocab), "bidirected": [],
            "structural_limitation": structural_limitation(), "meta": meta}


def write_prediction(path, src, dst, score, vocab, **meta):
    doc = prediction_document(src, dst, score, vocab, **meta)
    write_json(path, doc)
    return len(doc["directed"])


__all__ = ["structural_limitation", "edge_list", "prediction_document", "write_prediction"]
