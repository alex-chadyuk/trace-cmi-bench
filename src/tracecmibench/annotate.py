"""Score one test read with the benchmark's scorer and add provenance (D13; PRD scenarios 8, 10, 11, 15).

    python -m tracecmibench.annotate --corpus <dir> --run-dir out/<discover-run> \
        --arm faithful/library --agg max --grain request --pretrain-results out/<pretrain>/run/results.json \
        --per-lag-files 8 --output-folder results/xs/latent/seed=0/faithful/library/max/request

Score side, after the method process has exited and `graphs/` has been
pulled. Calls `tracebench.score.score_corpus` — the function behind
`python -m tracebench.score` — on the thresholded prediction (structural
axes; written verbatim as `score.json`), on the full ranking (threshold-free
axes; `score-ranking.json`) and on each per-lag thresholded prediction
(per-lag recall). `annotate.json` then adds, without touching any metric:
the structural-limitation field with the count of bidirected truth edges the
arm could never have found (scenario 8), the coverage of the probed sample
(plan §5), the alphabet tokens the views never mint (D-CB-16), the oracle's
in-regime flag, the benchmark tool version and config hash, the corpus
identity, the model hash and the cell's frozen values. `causal_validity`
passes through as the scorer wrote it (scenario 11).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tracebench.constants import ALPHABET_JSON, GRAPHS_DIR
from tracebench.score import score_corpus

from .arms import arm_spec, cell_key
from .constants import (
    AGGREGATIONS, GRAINS, MANIFEST_JSON, ORACLE_IN_REGIME, PREDICTION_JSON_FMT, PREDICTION_LAG_JSON_FMT,
    RANKING_JSON_FMT, RESULTS_JSON, RUN_DIR, SCORES_NPZ_FMT,
)
from .corpus import Corpus
from .log import log
from .prediction import structural_limitation
from .project import read_scores_npz
from .record import RunRecord, read_json, write_json
from .vocab import Vocab

SCORE_JSON = "score.json"
SCORE_RANKING_JSON = "score-ranking.json"
ANNOTATE_JSON = "annotate.json"
ANNOTATE_SCHEMA = "tracecmibench/annotate@1"


def unreachable_tokens(corpus_dir, ordering, grain):
    """Alphabet tokens the view's vocabulary cannot emit (D-CB-16)."""
    alphabet = read_json(Path(corpus_dir) / GRAPHS_DIR / ALPHABET_JSON)
    vocab = Vocab.from_model_vocab(Corpus(corpus_dir, ordering, grain).vocab_json())
    minted = {vocab.token_string(t) for t in vocab.real_ids}
    tokens = [t["token"] for t in alphabet["tokens"]]
    missing = [t for t in tokens if t not in minted]
    return {"alphabet_tokens": len(tokens), "unreachable": len(missing), "unreachable_tokens": missing}


def run_annotate(args, rec):
    run_dir = Path(args.run_dir)
    disc = read_json(run_dir / RUN_DIR / RESULTS_JSON)
    if disc.get("status") != "ok" or disc.get("arm") != args.arm or disc.get("grain") != args.grain:
        raise ValueError(f"{run_dir} is not a completed discover run of {args.arm} on the {args.grain} grain")
    if disc.get("split") != "test":
        log({"event": "annotate_note", "note": f"annotating a {disc.get('split')} read"})
    key = cell_key(args.arm, args.agg, args.grain)
    corpus_dir = Path(args.corpus)
    manifest = read_json(corpus_dir / MANIFEST_JSON)
    prediction = read_json(run_dir / PREDICTION_JSON_FMT.format(grain=args.grain, agg=args.agg))
    ranking = read_json(run_dir / RANKING_JSON_FMT.format(grain=args.grain, agg=args.agg))
    score = score_corpus(corpus_dir, prediction, grain=args.grain)
    score_rank = score_corpus(corpus_dir, ranking, grain=args.grain)
    write_json(rec.out_dir / SCORE_JSON, score)
    write_json(rec.out_dir / SCORE_RANKING_JSON, score_rank)
    per_lag = {}
    n_lags = int(disc["predictions"][args.agg]["per_lag_files"])
    for lag in range(1, min(int(args.per_lag_files), n_lags) + 1):
        p = run_dir / PREDICTION_LAG_JSON_FMT.format(grain=args.grain, agg=args.agg, lag=lag)
        if not p.exists():
            break
        per_lag[str(lag)] = score_corpus(corpus_dir, read_json(p), grain=args.grain)["directed"]["recall"]
    inside = len(ranking["directed"]) - score_rank["universe"]["predictions_outside_universe"]
    scores = read_scores_npz(run_dir / SCORES_NPZ_FMT.format(grain=args.grain))
    pre = read_json(args.pretrain_results)
    oracle = pre.get("oracle", {})
    if pre.get("model_sha256") != disc["model_sha256"]:
        raise ValueError(f"pretrain record binds model {pre.get('model_sha256')} but the read used {disc['model_sha256']}")
    limitation = structural_limitation()
    limitation["truth_bidirected_edges"] = int(score["universe"]["truth_bidirected"])
    annotate = {
        "schema": ANNOTATE_SCHEMA, "cell": key, "arm": args.arm, "agg": args.agg, "grain": args.grain, "split": disc["split"],
        "rung": manifest.get("instance"), "variant": score["variant"], "seed": manifest.get("seed"),
        "corpus_id": disc["corpus_id"], "tool_version": manifest.get("tool_version"), "config_hash": manifest.get("config_hash"),
        "model_sha256": disc["model_sha256"], "arm_class": arm_spec(args.arm)["class"],
        "frozen": {"tau": disc["predictions"][args.agg]["tau"], "c": disc["c"], "N": disc["N"], "g": disc["g"],
                   "freeze": disc.get("freeze")},
        "floor": score["floor"],
        "structural_limitation": limitation,
        "coverage": {"n_sequences_probed": disc["n_probed"], "n_skipped_short": disc["n_skipped_short"],
                     "pairs_scored": int(len(ranking["directed"])), "pairs_in_universe": int(inside),
                     "universe_ordered_pairs": score_rank["universe"]["ordered_pairs"],
                     "universe_fraction_cooccurring": (inside / score_rank["universe"]["ordered_pairs"]) if score_rank["universe"]["ordered_pairs"] else None,
                     "truth_directed": score_rank["universe"]["truth_directed"],
                     "reachable_recall_ceiling": score_rank["directed"]["recall"],
                     "n_pairs_in_table": int(len(scores["src"]))},
        "unreachable_tokens": unreachable_tokens(corpus_dir, str(scores["ordering"]), args.grain),
        "corrupted_cells": disc["counts"],
        "oracle": {"eps_hat": oracle.get("eps_hat"), "in_regime": (oracle.get("eps_hat") is not None and oracle["eps_hat"] < ORACLE_IN_REGIME),
                   "entropy_order": oracle.get("entropy_order"), "val_loss": oracle.get("val_loss")},
        "budget": pre.get("budget"),
        "score": score, "score_ranking": {"auroc": score_rank["auroc"], "average_precision": score_rank["average_precision"]},
        "per_lag_recall": per_lag,
        "causal_validity": score["causal_validity"],
        "files": {"score": SCORE_JSON, "score_ranking": SCORE_RANKING_JSON, "discover_run": run_dir.name},
    }
    write_json(rec.out_dir / ANNOTATE_JSON, annotate)
    log({"event": "annotate_done", "cell": key, "f1_directed": score["directed"]["f1"], "reachable_recall_ceiling": annotate["coverage"]["reachable_recall_ceiling"]})
    return {"annotate": ANNOTATE_JSON, "score": SCORE_JSON, "cell": key, "corpus_id": disc["corpus_id"], "model_sha256": disc["model_sha256"],
            "f1_directed": score["directed"]["f1"]}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True, help="a corpus with its score tier pulled")
    p.add_argument("--run-dir", required=True, help="the discover run's output folder")
    p.add_argument("--arm", required=True)
    p.add_argument("--agg", required=True, choices=AGGREGATIONS)
    p.add_argument("--grain", required=True, choices=GRAINS)
    p.add_argument("--pretrain-results", required=True, help="the pretrain run's results.json (oracle, budget, model hash)")
    p.add_argument("--per-lag-files", required=True, type=int, help="score per-lag predictions for lags 1..K")
    p.add_argument("--output-folder", required=True, help="the results cell directory")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, "annotate", vars(args)) as rec:
        res = run_annotate(args, rec)
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
