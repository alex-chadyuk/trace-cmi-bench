"""The blind validation sweep (D12): one probing pass per `(arm, c, N)`, both grains,
τ never applied here.

    python -m tracecmibench.sweep --corpus <dir> --ordering end --grains request session \
        --split val --model <model.pt> --arms faithful/paper faithful/library \
        --contexts 1 2 3 --particles-grid 2 8 32 128 --guidance 3 --max-len 64 --max-lag 63 \
        --num-sequences 0 --sequence-sample head --microbatch-rows 2048 --memory-cap-gb 20 \
        --divergence log1mexp --clamp-eps 1e-9 --probe-amp bf16 --gate <report> \
        --seed 0 --device cuda --output-folder out/<run>

Refuses `--split test` (the test split is read once, by `discover`, after a
committed freeze) and refuses without a passing gate report. The effective
guidance per cell is `g = min(--guidance, c)` (plan §3, D-CB-9) and rides
every score file. `--contexts` applies to every grain given; a rung's request
and session grids differ, so a replica passes one grain per invocation when
they do. `--num-sequences` takes one value per grain (or one for all).

Emits `scores-<arm>-c<c>-N<N>-<grain>.npz` per cell (the corpus-level table
with `max`, `mean`, `count` and per-lag max for all four statistics) and a
`results.json` with per-cell throughput, wall clock and corrupted-cell counts.
`scoresweep` reads these on the score side.
"""
from __future__ import annotations

import argparse

from .arms import arm_slug, arm_spec
from .constants import GRAINS, ORDERINGS, SPLITS, SWEEP_SCORES_NPZ_FMT
from .corpus import Corpus
from .data import SequenceStore
from .discover import add_probe_knobs, probe_store
from .hashes import require_gate
from .log import log
from .model import DecoderLM, sha256_of
from .project import write_scores_npz
from .record import RunRecord
from .vocab import Vocab


class SweepRefusal(RuntimeError):
    pass


def run_sweep(args, rec):
    if args.split == "test":
        raise SweepRefusal("sweep runs on validation only; the test split is read once by discover after a committed freeze")
    require_gate(args.gate)
    for arm in args.arms:
        arm_spec(arm)
    if len(args.num_sequences) not in (1, len(args.grains)):
        raise ValueError("--num-sequences takes one value, or one per grain in the order of --grains")
    model, _ = DecoderLM.load(args.model, args.device)
    model_sha = sha256_of(args.model)
    cells = []
    for gi, grain in enumerate(args.grains):
        n_seq = args.num_sequences[0] if len(args.num_sequences) == 1 else args.num_sequences[gi]
        corpus = Corpus(args.corpus, args.ordering, grain)
        vocab = Vocab.from_model_vocab(corpus.vocab_json())
        if vocab.size != model.config["vocab_size"]:
            raise ValueError(f"{grain}: the view's vocabulary has {vocab.size} ids but the model was trained on {model.config['vocab_size']}")
        store = SequenceStore.from_corpus(corpus, args.split, vocab, args.max_len, n=n_seq, mode=args.sequence_sample, seed=args.seed)
        for arm in args.arms:
            for c in args.contexts:
                g = min(args.guidance, c)
                for N in args.particles_grid:
                    scores, _, facts = probe_store(model, vocab, store, arm, c, g, N, args.max_lag, args.microbatch_rows,
                                                   args.memory_cap_gb, args.device, args.probe_amp, args.divergence,
                                                   args.clamp_eps, args.seed, args.split, keep_cells=False)
                    name = SWEEP_SCORES_NPZ_FMT.format(arm=arm_slug(arm), c=c, n=N, grain=grain)
                    write_scores_npz(rec.out_dir / name, scores, arm=arm, c=c, N=N, g=g, grain=grain, split=args.split,
                                     ordering=args.ordering, model_sha256=model_sha, corpus_id=corpus.corpus_id,
                                     n_sequences=facts["n_probed"], max_lag=args.max_lag, divergence=args.divergence)
                    cells.append({"arm": arm, "c": c, "N": N, "g": g, "grain": grain, "file": name, "n_pairs": int(len(scores["src"])),
                                  **{k: facts[k] for k in ("n_sequences", "n_probed", "n_skipped_short", "n_cells", "tok_pos",
                                                            "tok_pos_per_s", "wall_clock_s", "counts")}})
                    log({"event": "sweep_cell_done", "arm": arm, "c": c, "N": N, "grain": grain, "file": name})
    total_tok = sum(c["tok_pos"] for c in cells)
    total_wall = sum(c["wall_clock_s"] for c in cells)
    rec.note(tok_pos_per_s=total_tok / max(total_wall, 1e-9))
    return {"split": args.split, "model_sha256": model_sha, "arms": list(args.arms), "contexts": list(args.contexts),
            "particles_grid": list(args.particles_grid), "grains": list(args.grains), "n_cells": len(cells),
            "probe_wall_clock_s": round(total_wall, 3), "tok_pos_per_s": total_tok / max(total_wall, 1e-9),
            "gate": str(args.gate), "cells": cells}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True)
    p.add_argument("--ordering", required=True, choices=ORDERINGS)
    p.add_argument("--grains", required=True, nargs="+", choices=GRAINS)
    p.add_argument("--split", required=True, choices=SPLITS, help="val; test is refused")
    p.add_argument("--model", required=True)
    p.add_argument("--arms", required=True, nargs="+")
    p.add_argument("--contexts", required=True, nargs="+", type=int, help="c grid, BOS-counted")
    p.add_argument("--particles-grid", required=True, nargs="+", type=int, help="N grid")
    p.add_argument("--guidance", required=True, type=int, help="g; the cell uses min(g, c)")
    p.add_argument("--num-sequences", required=True, nargs="+", type=int, help="per grain (or one value); 0 = whole split")
    add_probe_knobs(p)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, "sweep", vars(args)) as rec:
        res = run_sweep(args, rec)
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
