"""Score every sweep cell at every τ with the benchmark's scorer (D13; score side).

    python -m tracecmibench.scoresweep --corpus <dir> --sweep-dir out/<sweep-run> \
        --grains request session --taus 1e-6 3e-6 1e-5 1.72e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 \
        --output-folder out/<run>

Runs only after the method process has exited and the score tier (`graphs/`)
has been pulled: this module is not on a method code path and reads the
target, the alphabet and the manifest directly. No metric arithmetic of its
own: the target and alphabet are loaded once per grain, the universe built
once, and `tracebench.score.score_at_floor` is called at the target's default
floor for every `(cell, aggregation, τ)`; the directed / skeleton /
orientation sub-trees are copied into `val-table.json` as returned. Per
`(cell, aggregation)` the full ranking is scored once more for the
threshold-free axes and for the coverage rule (plan §5): the recall of the
full ranking is the fraction of truth edges that co-occur in the probed
sample, because the scorer counts every listed edge as present.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tracebench.constants import ALPHABET_JSON, GRAPHS_DIR, SCORING_TARGET_JSON
from tracebench.score import SCORING_TARGET_SESSION_JSON, build_universe, score_at_floor

from .arms import arm_spec
from .constants import AGGREGATIONS, GRAINS, MANIFEST_JSON, VAL_TABLE_JSON
from .corpus import Corpus
from .log import log
from .prediction import prediction_document
from .project import read_scores_npz
from .record import RunRecord, read_json, write_json
from .select import ranking, select_edges
from .vocab import Vocab

VAL_TABLE_SCHEMA = "tracecmibench/val-table@1"


def load_target(corpus_dir, grain):
    gdir = Path(corpus_dir) / GRAPHS_DIR
    target = read_json(gdir / (SCORING_TARGET_JSON if grain == "request" else SCORING_TARGET_SESSION_JSON))
    alphabet = read_json(gdir / ALPHABET_JSON)
    ordered, unordered = build_universe(alphabet, target)
    return target, alphabet, ordered, unordered


def manifest_facts(corpus_dir):
    m = read_json(Path(corpus_dir) / MANIFEST_JSON)
    return {"tool_version": m.get("tool_version"), "config_hash": m.get("config_hash"), "instance": m.get("instance"),
            "variant": m.get("variant"), "seed": m.get("seed")}


def sweep_files(sweep_dir, grain):
    return sorted(Path(sweep_dir).glob(f"scores-*-{grain}.npz"))


def score_cell(scores, stat, agg, tau, vocab, target, alphabet, floor, ordered, unordered):
    src, dst, s = select_edges(scores, stat, agg, tau)
    doc = prediction_document(src, dst, s, vocab)
    r = score_at_floor(target, alphabet, doc, floor, ordered, unordered)
    return {"n_edges": int(len(src)), "directed": r["directed"], "skeleton": r["skeleton"], "orientation": r["orientation"],
            "shd_mixed": r["shd_mixed"], "predictions_outside_universe": r["universe"]["predictions_outside_universe"]}


def score_ranking(scores, stat, agg, vocab, target, alphabet, floor, ordered, unordered):
    src, dst, s = ranking(scores, stat, agg)
    doc = prediction_document(src, dst, s, vocab)
    r = score_at_floor(target, alphabet, doc, floor, ordered, unordered)
    inside = int(len(src)) - int(r["universe"]["predictions_outside_universe"])
    return {"n_pairs": int(len(src)), "auroc": r["auroc"], "average_precision": r["average_precision"],
            "coverage": {"universe_ordered_pairs": r["universe"]["ordered_pairs"],
                         "pairs_cooccurring": inside,
                         "universe_fraction_cooccurring": inside / r["universe"]["ordered_pairs"] if r["universe"]["ordered_pairs"] else None,
                         "truth_directed": r["universe"]["truth_directed"],
                         "reachable_recall_ceiling": r["directed"]["recall"]}}


def run_scoresweep(args, rec):
    facts = manifest_facts(args.corpus)
    rows, coverage = [], {}
    model_sha = None
    corpus_id = None
    for grain in args.grains:
        target, alphabet, ordered, unordered = load_target(args.corpus, grain)
        floor = float(target["default_floor"])
        files = sweep_files(args.sweep_dir, grain)
        if not files:
            raise FileNotFoundError(f"no scores-*-{grain}.npz under {args.sweep_dir}")
        for f in files:
            scores = read_scores_npz(f)
            arm = str(scores["arm"]); c = int(scores["c"]); N = int(scores["N"]); g = int(scores["g"])
            ordering = str(scores["ordering"])
            model_sha = model_sha or str(scores["model_sha256"])
            corpus_id = corpus_id or str(scores["corpus_id"])
            if str(scores["model_sha256"]) != model_sha:
                raise ValueError(f"{f.name}: mixed models in one sweep ({scores['model_sha256']} vs {model_sha})")
            vocab = Vocab.from_model_vocab(Corpus(args.corpus, ordering, grain).vocab_json())
            stat = arm_spec(arm)["statistic"]
            for agg in AGGREGATIONS:
                key = f"{arm}/{agg}/{grain}/c{c}/N{N}"
                coverage[key] = score_ranking(scores, stat, agg, vocab, target, alphabet, floor, ordered, unordered)
                for tau in args.taus:
                    rows.append({"arm": arm, "agg": agg, "grain": grain, "c": c, "N": N, "g": g, "tau": float(tau), "floor": floor,
                                 **score_cell(scores, stat, agg, float(tau), vocab, target, alphabet, floor, ordered, unordered)})
            log({"event": "scoresweep_file", "file": f.name, "grain": grain, "arm": arm, "c": c, "N": N})
    table = {"schema": VAL_TABLE_SCHEMA, "corpus_id": corpus_id, "model_sha256": model_sha, **facts,
             "sweep_dir_name": Path(args.sweep_dir).name, "grains": list(args.grains), "taus": [float(t) for t in args.taus],
             "n_rows": len(rows), "coverage": coverage, "cells": rows}
    write_json(rec.out_dir / VAL_TABLE_JSON, table)
    return {"val_table": VAL_TABLE_JSON, "n_rows": len(rows), "corpus_id": corpus_id, "model_sha256": model_sha, **facts}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True, help="a corpus with its score tier pulled (graphs/ present)")
    p.add_argument("--sweep-dir", required=True, help="the sweep run's output folder")
    p.add_argument("--grains", required=True, nargs="+", choices=GRAINS)
    p.add_argument("--taus", required=True, nargs="+", type=float, help="the τ grid (plan §3)")
    p.add_argument("--output-folder", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, "scoresweep", vars(args)) as rec:
        res = run_scoresweep(args, rec)
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
