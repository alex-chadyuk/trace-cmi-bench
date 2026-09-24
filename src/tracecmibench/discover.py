"""Run one arm on one corpus with one frozen model (D12; PRD scenarios 4, 6, 22, 23).

    python -m tracecmibench.discover --corpus <dir> --ordering end --grain request --split test \
        --model <model.pt> --arm faithful/library --particles 8 --guidance 3 --context 2 \
        --max-len 64 --max-lag 62 --num-sequences 0 --sequence-sample head \
        --microbatch-rows 2048 --memory-cap-gb 20 --divergence log1mexp --clamp-eps 1e-9 \
        --probe-amp bf16 --aggs max mean --taus 3e-2 1e-2 --per-lag-files 8 \
        --freeze freezes/<date>-xs-latent-s0.json --gate gate/<date>-gate-report.json \
        --seed 0 --device cuda --output-folder out/<run>

Refuses without a passing gate report at the current engine hash (scenario 4)
and refuses an output folder that already holds a completed run. With
`--split test` a freeze is mandatory: the run asserts that the commit which
added the freeze file is an ancestor of HEAD and predates the run's start,
and that `(tau, c, N, g, model_sha256, corpus_id)` in the freeze equal the
command line and the loaded model for every sub-arm written (scenarios 22,
23). The freeze selects `(c, N, τ)` per sub-arm, so a read writes the
sub-arms (`--aggs`) whose frozen `(c, N)` equal this pass's; when the two
sub-arms froze at different `(c, N)`, the job runs `discover` once per
sub-arm. With `--split val` the
freeze is optional (an exploratory read; the blind path is `sweep`).

Writes `matrices-<grain>.npz` (the projected cells of every probed sequence
with all four statistics), `scores-<grain>.npz` (the corpus-level table both
sub-arms read), and per sub-arm the thresholded prediction, the full ranking
and the per-lag files (D-CB-18); `results.json` carries the corrupted-cell
counts (scenario 15), the sequence and cell counts, throughput and wall
clock. The probing core (`probe_store`) is shared with `sweep`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .arms import arm_slug, arm_spec, cell_key
from .constants import (
    AGGREGATIONS, DEVICES, DIVERGENCES, GRAINS, MATRICES_NPZ_FMT, ORDERINGS, PREDICTION_JSON_FMT,
    PREDICTION_LAG_JSON_FMT, RANKING_JSON_FMT, RANKING_LAG_JSON_FMT, SCORES_NPZ_FMT, SEQUENCE_SAMPLES, SPLITS,
    STATISTICS,
)
from .corpus import Corpus
from .data import SequenceStore
from .hashes import require_gate
from .log import log
from .model import DecoderLM, sha256_of
from .prediction import write_prediction
from .pretrain import AMP_MODES
from .probe import check_memory, probe_sequence
from .project import PairAccumulator, sequence_cells, write_scores_npz
from .record import RunRecord, read_json
from .select import per_lag_ranking, ranking, select_edges
from .statistics import cell_statistics, stats_for
from .vocab import Vocab

SPLIT_STREAM = {"train": 0, "val": 1, "test": 2}


class FreezeRefusal(RuntimeError):
    pass


# --- the probing core -----------------------------------------------------------------------
def probe_store(model, vocab, store, arm, c, g, N, max_lag, microbatch_rows, memory_cap_gb, device, amp, divergence,
                clamp_eps, seed, split, keep_cells=True, progress_every=256):
    """One arm at one `(c, g, N)` over one sequence store. Returns the corpus-level table
    (`project.PairAccumulator.result()`), the flat projected-cell table (or None), the
    corrupted-cell counts and the run facts."""
    spec = arm_spec(arm)
    if not 1 <= g <= c:
        raise ValueError(f"guidance g = {g} must satisfy 1 <= g <= c = {c} (D-CB-9)")
    if len(store) == 0:
        raise ValueError("the sequence store is empty")
    max_L = int(store.lengths.max())
    mem = check_memory(model, microbatch_rows, max_L, N, max(max_L - c, 1), memory_cap_gb)
    log({"event": "memory", "split": split, "arm": arm, "c": c, "N": N, **mem})
    gen = torch.Generator(device=device).manual_seed(int(seed) + SPLIT_STREAM[split] * 1000003)
    acc = PairAccumulator(vocab.size, max_lag)
    cells_tab = {"seq": [], "j": [], "q": [], "lag": [], **{k: [] for k in STATISTICS}}
    counts = {"n_clamped_cells": 0,
              "compat_counts": {k: {"n_cell_to_zero": 0, "n_cell_to_fmax": 0} for k in STATISTICS}}
    n_skipped = 0
    tok_pos = 0
    t0 = time.monotonic()
    for s in range(len(store)):
        ids_np = store.get(s)
        if len(ids_np) - c < 2:                      # no position pair to test inside the window
            n_skipped += 1
            continue
        ids = torch.as_tensor(ids_np, dtype=torch.long, device=device)
        logp, X = probe_sequence(model, ids, c, spec["history"], g, N, max_lag, vocab.real_ids, gen,
                                 microbatch_rows, device, amp)
        tok_pos += X.shape[0] * X.shape[1] * X.shape[2]
        cells = cell_statistics(logp, c, max_lag, clamp_eps)
        stats = {k: v.cpu().numpy() for k, v in stats_for(cells, divergence).items()}
        counts["n_clamped_cells"] += int(cells["n_clamped_cells"])
        for k in STATISTICS:
            for d in ("n_cell_to_zero", "n_cell_to_fmax"):
                counts["compat_counts"][k][d] += int(cells["compat_counts"][k][d])
        window = np.asarray(ids_np[c:])
        j, q, lag, u, v, values = sequence_cells(stats, window, vocab, max_lag)
        acc.add(u, v, lag, values)
        if keep_cells:
            cells_tab["seq"].append(np.full(len(j), s, dtype=np.int32))
            cells_tab["j"].append(j.astype(np.int16))
            cells_tab["q"].append(q.astype(np.int16))
            cells_tab["lag"].append(lag.astype(np.int16))
            for k in STATISTICS:
                cells_tab[k].append(np.asarray(values[k], dtype=np.float32))
        del logp, X
        if (s + 1) % progress_every == 0:
            log({"event": "probe_progress", "split": split, "arm": arm, "c": c, "N": N, "done": s + 1, "of": len(store),
                 "tok_pos_per_s": tok_pos / max(time.monotonic() - t0, 1e-9)})
    wall = time.monotonic() - t0
    table = None
    if keep_cells:
        table = {k: (np.concatenate(v) if v else np.zeros(0, dtype=np.float32 if k in STATISTICS else np.int32))
                 for k, v in cells_tab.items()}
    facts = {"n_sequences": len(store), "n_skipped_short": n_skipped, "n_probed": len(store) - n_skipped,
             "n_cells": int(acc.n_cells), "tok_pos": int(tok_pos), "wall_clock_s": round(wall, 3),
             "tok_pos_per_s": tok_pos / max(wall, 1e-9), "memory": mem, "counts": counts}
    log({"event": "probe_done", "split": split, "arm": arm, "c": c, "N": N,
         **{k: v for k, v in facts.items() if k not in ("memory", "counts")}})
    return acc.result(), table, facts


def write_matrices(path, table):
    np.savez(path, **table)


# --- the prediction files of both sub-arms ----------------------------------------------------
def write_arm_outputs(out_dir, scores, arm, vocab, grain, taus, per_lag_files, max_lag, aggs=AGGREGATIONS, **meta):
    """Per aggregation: `prediction-<grain>-<agg>.json` (score > τ), `ranking-<grain>-<agg>.json`
    (every scored pair) and, for lags `1..per_lag_files`, the per-lag ranking and the per-lag
    thresholded prediction (D-CB-18). Returns a summary per aggregation."""
    out_dir = Path(out_dir)
    stat = arm_spec(arm)["statistic"]
    n_lags = min(int(per_lag_files), int(max_lag))
    summary = {}
    for agg in aggs:
        tau = float(taus[agg])
        common = {"arm": arm, "agg": agg, "grain": grain, "tau": tau, "statistic": stat, **meta}
        src, dst, s = select_edges(scores, stat, agg, tau)
        n = write_prediction(out_dir / PREDICTION_JSON_FMT.format(grain=grain, agg=agg), src, dst, s, vocab, kind="prediction", **common)
        rs, rd, rv = ranking(scores, stat, agg)
        write_prediction(out_dir / RANKING_JSON_FMT.format(grain=grain, agg=agg), rs, rd, rv, vocab, kind="ranking", **common)
        for lag in range(1, n_lags + 1):
            ls, ld, lv = per_lag_ranking(scores, stat, lag)
            write_prediction(out_dir / RANKING_LAG_JSON_FMT.format(grain=grain, agg=agg, lag=lag), ls, ld, lv, vocab,
                             kind="ranking", lag=lag, **common)
            hit = lv > tau
            write_prediction(out_dir / PREDICTION_LAG_JSON_FMT.format(grain=grain, agg=agg, lag=lag), ls[hit], ld[hit], lv[hit],
                             vocab, kind="prediction", lag=lag, **common)
        summary[agg] = {"tau": tau, "n_edges": int(n), "n_pairs": int(len(rs)), "per_lag_files": n_lags}
    return summary


# --- the freeze assertions (scenarios 22, 23) ----------------------------------------------------
def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def freeze_commit(freeze_path):
    """`(sha, committed_at)` of the commit that added the freeze file; refuses an uncommitted or
    modified freeze."""
    path = Path(freeze_path).resolve()
    if not path.exists():
        raise FreezeRefusal(f"freeze {path} does not exist")
    try:
        top = Path(_git(["rev-parse", "--show-toplevel"], cwd=path.parent))
    except subprocess.CalledProcessError as e:
        raise FreezeRefusal(f"freeze {path} is not inside a git repository") from e
    rel = path.relative_to(top).as_posix()
    if _git(["status", "--porcelain", "--", rel], cwd=top):
        raise FreezeRefusal(f"freeze {rel} has uncommitted changes; commit it before the test read (PRD scenario 22)")
    lines = _git(["log", "--diff-filter=A", "--format=%H %cI", "--", rel], cwd=top).splitlines()
    if not lines:
        raise FreezeRefusal(f"freeze {rel} is not committed; commit it before the test read (PRD scenario 22)")
    sha, committed_at = lines[-1].split()          # the commit that first added the file
    ok = subprocess.run(["git", "merge-base", "--is-ancestor", sha, "HEAD"], cwd=top).returncode == 0
    if not ok:
        raise FreezeRefusal(f"freeze commit {sha[:8]} is not an ancestor of HEAD (PRD scenario 22)")
    return sha, committed_at


def _iso(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def assert_freeze(freeze, freeze_path, arm, grain, taus, c, N, g, model_sha256, corpus_id, started, aggs=AGGREGATIONS):
    """The freeze names this run's values exactly and predates it."""
    sha, committed_at = freeze_commit(freeze_path)
    if not _iso(committed_at) < _iso(started):
        raise FreezeRefusal(f"the run started at {started} but the freeze was committed at {committed_at} (PRD scenario 22)")
    for agg in aggs:
        key = cell_key(arm, agg, grain)
        cell = freeze.get("cells", {}).get(key)
        if cell is None:
            raise FreezeRefusal(f"freeze has no cell {key}")
        want = {"tau": float(taus[agg]), "c": int(c), "N": int(N), "g": int(g)}
        got = {k: cell.get(k) for k in want}
        if got != want:
            raise FreezeRefusal(f"freeze cell {key} = {got} but the command line says {want} (PRD scenario 23)")
    if freeze.get("model_sha256") != model_sha256:
        raise FreezeRefusal(f"freeze binds model {freeze.get('model_sha256')} but the loaded model hashes to {model_sha256}")
    if freeze.get("corpus_id") != corpus_id:
        raise FreezeRefusal(f"freeze is for corpus {freeze.get('corpus_id')!r}, this run reads {corpus_id!r}")
    return {"sha": sha, "committed_at": committed_at, "path": str(freeze_path)}


# --- the command ------------------------------------------------------------------------------------
def run_discover(args, rec):
    require_gate(args.gate)
    if args.split == "test" and not args.freeze:
        raise FreezeRefusal("a test read needs --freeze <committed freeze record> (PRD scenario 22)")
    device = args.device
    model, _ = DecoderLM.load(args.model, device)
    model_sha = sha256_of(args.model)
    corpus = Corpus(args.corpus, args.ordering, args.grain)
    vocab = Vocab.from_model_vocab(corpus.vocab_json())
    if vocab.size != model.config["vocab_size"]:
        raise ValueError(f"the view's vocabulary has {vocab.size} ids but the model was trained on {model.config['vocab_size']}")
    if len(args.taus) != len(args.aggs) or len(set(args.aggs)) != len(args.aggs):
        raise ValueError("--taus takes one value per --aggs entry, in order, without repeats")
    taus = dict(zip(args.aggs, (float(t) for t in args.taus)))
    freeze_facts = None
    if args.freeze:
        freeze = read_json(args.freeze)
        freeze_facts = assert_freeze(freeze, args.freeze, args.arm, args.grain, taus, args.context, args.particles,
                                     args.guidance, model_sha, corpus.corpus_id, rec.started, aggs=args.aggs)
        log({"event": "freeze_ok", **freeze_facts})
    store = SequenceStore.from_corpus(corpus, args.split, vocab, args.max_len, n=args.num_sequences,
                                      mode=args.sequence_sample, seed=args.seed)
    scores, table, facts = probe_store(model, vocab, store, args.arm, args.context, args.guidance, args.particles,
                                       args.max_lag, args.microbatch_rows, args.memory_cap_gb, device, args.probe_amp,
                                       args.divergence, args.clamp_eps, args.seed, args.split, keep_cells=True)
    out = rec.out_dir
    write_matrices(out / MATRICES_NPZ_FMT.format(grain=args.grain), table)
    meta = {"arm": args.arm, "c": args.context, "N": args.particles, "g": args.guidance, "grain": args.grain,
            "split": args.split, "ordering": args.ordering, "model_sha256": model_sha, "corpus_id": corpus.corpus_id,
            "n_sequences": facts["n_probed"], "max_lag": args.max_lag, "divergence": args.divergence}
    write_scores_npz(out / SCORES_NPZ_FMT.format(grain=args.grain), scores, **meta)
    summary = write_arm_outputs(out, scores, args.arm, vocab, args.grain, taus, args.per_lag_files, args.max_lag, aggs=args.aggs,
                                c=args.context, N=args.particles, g=args.guidance, split=args.split,
                                model_sha256=model_sha, corpus_id=corpus.corpus_id, n_sequences=facts["n_probed"])
    rec.note(tok_pos_per_s=facts["tok_pos_per_s"])
    return {
        "arm": args.arm, "arm_slug": arm_slug(args.arm), "grain": args.grain, "split": args.split,
        "corpus_id": corpus.corpus_id, "model_sha256": model_sha, "c": args.context, "N": args.particles, "g": args.guidance,
        "n_sequences": facts["n_sequences"], "n_probed": facts["n_probed"], "n_skipped_short": facts["n_skipped_short"],
        "n_cells": facts["n_cells"], "n_pairs": int(len(scores["src"])), "counts": facts["counts"], "memory": facts["memory"],
        "tok_pos": facts["tok_pos"], "tok_pos_per_s": facts["tok_pos_per_s"], "probe_wall_clock_s": facts["wall_clock_s"],
        "predictions": summary, "freeze": freeze_facts, "gate": str(args.gate),
        "files": {"matrices": MATRICES_NPZ_FMT.format(grain=args.grain), "scores": SCORES_NPZ_FMT.format(grain=args.grain)},
    }


def add_probe_knobs(p):
    p.add_argument("--max-len", required=True, type=int, help="cap on real tokens per sequence (D-CB-5)")
    p.add_argument("--max-lag", required=True, type=int)
    p.add_argument("--sequence-sample", required=True, choices=SEQUENCE_SAMPLES, help="D-CB-14")
    p.add_argument("--microbatch-rows", required=True, type=int)
    p.add_argument("--memory-cap-gb", required=True, type=float)
    p.add_argument("--divergence", required=True, choices=DIVERGENCES, help="log1mexp (safe) or compat (the published clamp)")
    p.add_argument("--clamp-eps", required=True, type=float, help="ε of the published float32 clamp (scenario 15)")
    p.add_argument("--probe-amp", required=True, choices=AMP_MODES, help="autocast mode of the probe (D-CB-20)")
    p.add_argument("--gate", required=True, help="a committed gate report that passed at the current engine hash (scenario 4)")
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--device", required=True, choices=DEVICES)
    p.add_argument("--output-folder", required=True)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True)
    p.add_argument("--ordering", required=True, choices=ORDERINGS)
    p.add_argument("--grain", required=True, choices=GRAINS)
    p.add_argument("--split", required=True, choices=SPLITS)
    p.add_argument("--model", required=True, help="the frozen model.pt (D-CB-17)")
    p.add_argument("--arm", required=True, help="a registered arm, e.g. faithful/library")
    p.add_argument("--particles", required=True, type=int, help="N")
    p.add_argument("--guidance", required=True, type=int, help="g, 1 <= g <= c (D-CB-9)")
    p.add_argument("--context", required=True, type=int, help="c, BOS-counted")
    p.add_argument("--num-sequences", required=True, type=int, help="sequences probed; 0 = the whole split")
    p.add_argument("--aggs", required=True, nargs="+", choices=AGGREGATIONS, help="the sub-arms this pass writes (D-CB-8)")
    p.add_argument("--taus", required=True, nargs="+", type=float, help="one threshold per --aggs entry, in order")
    p.add_argument("--per-lag-files", required=True, type=int, help="write per-lag ranking/prediction files for lags 1..K")
    p.add_argument("--freeze", required=True, help="the committed freeze record, or '' (allowed on val only)")
    add_probe_knobs(p)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.freeze = args.freeze or None
    with RunRecord(args.output_folder, "discover", vars(args)) as rec:
        res = run_discover(args, rec)
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
