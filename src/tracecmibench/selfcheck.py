"""The reproduction gate (PRD Goal 8, scenarios 1–4; D11): the paper's synthetic
experiment, read against the replication note's registered findings.

One job: generate Eq. 21 worlds (refusing below the redundancy band) →
pretrain the backbone on them → probe the head sequences of the validation
split with both faithful arms at the gate's `c`, `g` and `N` → choose τ on
validation by the mean per-sequence directed F1 of the Def. 3.2 projection
against the exact truth → freeze in-process → probe the head sequences of the
test split once → assert on `faithful/library` (the reading the note
characterised):

    F1(τ = 3e-5)  <= 0.55           the printed threshold does NOT replicate (scenario 2; one-sided since 2026-09-24)
    F1(τ*)        >= 0.86           the blind validation-selected threshold does (scenario 3)

`faithful/paper` is reported ungated. The report carries `engine_sha256`, and
`discover` / `sweep` refuse any benchmark corpus without a passing report at
the current engine hash (scenario 4). The truth for a sequence is restricted
to position pairs whose cause lies in the window (positions `>= c`), the
same set the staircase can test; per-lag recall is reported at the position
level; the oracle score is computed against the generator's exact
conditional entropy (D-CB-7) beside the n-gram estimate.

    python -m tracecmibench.selfcheck <every generator, pretrain and probe knob> --output-folder <dir>
"""
from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from . import __version__
from .constants import (
    ARM_FAITHFUL_LIBRARY, ARM_FAITHFUL_PAPER, ARMS, DEVICES, GATE_ARM, GATE_DIR, GATE_F1_PRINTED_MAX,
    GATE_F1_SELECTED_MIN, GATE_TAU_GRID, GATE_TAU_PRINTED, ORACLE_IN_REGIME,
)
from .corpus import Corpus
from .data import SequenceStore
from .generator import GENERATOR_JSON, TRUTH_DIR, GeneratorRefusal, generate
from .generator import build_parser as generator_parser
from .hashes import engine_sha256
from .log import log
from .model import DecoderLM
from .pretrain import AMP_MODES
from .pretrain import build_parser as pretrain_parser
from .pretrain import pretrain
from .probe import check_memory, probe_sequence
from .project import sequence_type_edges
from .record import RunRecord, git_commit, read_json, write_json
from .select import f1_at
from .statistics import cell_statistics
from .vocab import Vocab

GATE_ARMS = (ARM_FAITHFUL_LIBRARY, ARM_FAITHFUL_PAPER)


# --- truth per sequence ---------------------------------------------------------------------
def load_truth(corpus_dir, split):
    z = np.load(Path(corpus_dir) / TRUTH_DIR / f"{split}-truth.npz")
    return {k: z[k] for k in z.files}


def truth_for_sequence(truth, s, c, vocab, x_row):
    """Type edges (token ids, u != v) and position pairs (window indices) of sequence `s`
    whose cause sits in the window: generator position `i` ↔ sequence position `i + 1`
    ↔ window index `i + 1 − c`."""
    sel = (truth["seq"] == s) & truth["edge"]
    i = truth["i"][sel].astype(int)
    j = truth["j"][sel].astype(int)
    inside = (i + 1) >= c
    i, j = i[inside], j[inside]
    wi, wj = i + 1 - c, j + 1 - c
    u = np.array([vocab.base_id(int(x_row[a])) for a in i], dtype=np.int64)
    v = np.array([vocab.base_id(int(x_row[b])) for b in j], dtype=np.int64)
    keep = u != v
    edges = set(zip(u[keep].tolist(), v[keep].tolist()))
    return edges, list(zip(wi.tolist(), wj.tolist()))


# --- probing a split ------------------------------------------------------------------------------
def probe_split(model, vocab, corpus, split, truth, args, device):
    """Per sequence and arm: the arm's statistic matrix; plus the truth. Returns a list of dicts."""
    store = SequenceStore.from_corpus(corpus, split, vocab, args.seq_len, n=args.num_sequences, mode="head", seed=args.seed)
    if len(store) > truth["x"].shape[0]:
        raise ValueError(f"{split}: {len(store)} sequences requested but the truth covers {truth['x'].shape[0]} (raise --truth-head)")
    max_L = int(store.lengths.max())
    mem = check_memory(model, args.microbatch_rows, max_L, args.particles, max_L - args.context, args.memory_cap_gb)
    log({"event": "memory", "split": split, **mem})
    gen = torch.Generator(device=device).manual_seed(args.seed + {"val": 1, "test": 2}[split] * 1000003)
    out = []
    t0 = time.monotonic()
    tok_pos = 0
    counts = {arm: {"n_clamped_cells": 0, "compat_counts": None} for arm in GATE_ARMS}
    for s in range(len(store)):
        ids = torch.as_tensor(store.get(s), dtype=torch.long, device=device)
        x_row = truth["x"][s]
        assert (ids[1:-1].cpu().numpy() - 4 == x_row).all(), "truth row does not match the stored sequence"
        edges, pairs = truth_for_sequence(truth, s, args.context, vocab, x_row)
        rec = {"truth_edges": edges, "truth_pairs": pairs, "window_tokens": store.get(s)[args.context:], "stats": {}}
        for arm in GATE_ARMS:
            spec = ARMS[arm]
            logp, X = probe_sequence(model, ids, args.context, spec["history"], args.guidance, args.particles, args.max_lag,
                                     vocab.real_ids, gen, args.microbatch_rows, device, args.probe_amp)
            tok_pos += X.shape[0] * X.shape[1] * X.shape[2]
            cells = cell_statistics(logp, args.context, args.max_lag, args.clamp_eps)
            rec["stats"][arm] = cells["safe"][spec["statistic"]].cpu().numpy()
            counts[arm]["n_clamped_cells"] += cells["n_clamped_cells"]
            cc = cells["compat_counts"][spec["statistic"]]
            if counts[arm]["compat_counts"] is None:
                counts[arm]["compat_counts"] = dict(cc)
            else:
                for k in cc:
                    counts[arm]["compat_counts"][k] += cc[k]
            del logp, X
        out.append(rec)
        if (s + 1) % 128 == 0:
            log({"event": "probe_progress", "split": split, "done": s + 1, "of": len(store),
                 "tok_pos_per_s": tok_pos / max(time.monotonic() - t0, 1e-9)})
    wall = time.monotonic() - t0
    log({"event": "probe_done", "split": split, "n": len(store), "wall_clock_s": round(wall, 1), "tok_pos_per_s": tok_pos / max(wall, 1e-9)})
    return out, {"n_sequences": len(store), "wall_clock_s": wall, "tok_pos_per_s": tok_pos / max(wall, 1e-9), "counts": counts}


def f1_curve(records, arm, taus, vocab, max_lag):
    """Mean per-sequence directed precision / recall / F1 of the Def. 3.2 projection at every τ."""
    out = {}
    for tau in taus:
        acc = {"precision": [], "recall": [], "f1": []}
        for r in records:
            pred = sequence_type_edges(r["stats"][arm], r["window_tokens"], vocab, tau, max_lag)
            m = f1_at(pred, r["truth_edges"])
            for k in acc:
                acc[k].append(m[k])
        out[tau] = {k: (float(np.mean(v)) if v else 0.0) for k, v in acc.items()}
    return out


def per_lag_recall(records, arm, tau, max_lag):
    """Position-level recall of the truth pairs at each lag `1..max_lag`, at threshold τ."""
    hit = {k: [0, 0] for k in range(1, max_lag + 1)}
    for r in records:
        S = r["stats"][arm]
        for (i, j) in r["truth_pairs"]:
            k = j - i
            if k < 1 or k > max_lag or j >= S.shape[1]:
                continue
            hit[k][1] += 1
            hit[k][0] += bool(S[i, j] > tau)
    return {str(k): (n and h / n) if n else None for k, (h, n) in hit.items()}


def select_tau(curve):
    """Argmax of F1 over the grid; ties → the larger τ."""
    best = max(v["f1"] for v in curve.values())
    return max(t for t, v in curve.items() if v["f1"] == best)


def write_matrices(path, records, arms):
    """The per-sequence statistic matrices of a split as a flat cell table (the instrumentation
    that lets a corrected truth be re-scored without re-probing)."""
    seq, jj, qq = [], [], []
    vals = {arm: [] for arm in arms}
    for s, r in enumerate(records):
        S = r["stats"][arms[0]]
        j, q = np.nonzero(~np.isnan(S))
        seq.append(np.full(len(j), s, dtype=np.int32)); jj.append(j.astype(np.int16)); qq.append(q.astype(np.int16))
        for arm in arms:
            vals[arm].append(r["stats"][arm][j, q].astype(np.float32))
    np.savez(path, seq=np.concatenate(seq), j=np.concatenate(jj), q=np.concatenate(qq),
             **{arm.replace("/", "_"): np.concatenate(vals[arm]) for arm in arms})


def assertions_for(f1_printed, f1_selected):
    return [
        {"name": "printed_threshold_does_not_replicate", "value": f1_printed, "band": [0.0, GATE_F1_PRINTED_MAX],
         "passed": f1_printed <= GATE_F1_PRINTED_MAX, "scenario": 2, "arm": GATE_ARM, "tau": GATE_TAU_PRINTED},
        {"name": "selected_threshold_replicates", "value": f1_selected, "band": [GATE_F1_SELECTED_MIN, 1.0],
         "passed": f1_selected >= GATE_F1_SELECTED_MIN, "scenario": 3, "arm": GATE_ARM},
    ]


# --- the gate --------------------------------------------------------------------------------------
def run_gate(args, out_dir):
    out_dir = Path(out_dir)
    device = args.device
    report = {"schema": "tracecmibench/gate-report@1", "tool_version": __version__, "commit": git_commit(),
              "date": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"), "engine_sha256": engine_sha256(),
              "gate_arm": GATE_ARM, "context": args.context, "guidance": args.guidance, "particles": args.particles,
              "max_lag": args.max_lag, "probe_amp": args.probe_amp, "tau_grid": list(GATE_TAU_GRID), "tau_printed": GATE_TAU_PRINTED,
              "passed": False}
    # 1. worlds
    scm_dir = out_dir / "scm"
    try:
        gen = generate(args, scm_dir)
    except GeneratorRefusal as e:
        report.update({"reason": str(e), "assertions": [{"name": "redundancy_band", "passed": False, "value": str(e), "scenario": 1}]})
        return report
    report["redundancy"] = gen["redundancy"]["redundancy"]
    report["w_scale"] = args.w_scale
    report["generator"] = {k: gen[k] for k in ("scm", "seq_len", "entropy", "truth", "log_alphabet")}
    # 2. backbone
    pre_args = SimpleNamespace(**vars(args), corpus=str(scm_dir), ordering="end", grain="request", max_len=args.seq_len)
    pre_dir = out_dir / "pretrain"
    pre_dir.mkdir(parents=True, exist_ok=True)
    pre = pretrain(pre_args, pre_dir)
    write_json(pre_dir / "pretrain-results.json", pre)
    h_exact = gen["entropy"]["train"]["mean_entropy"]
    val_loss = pre["oracle"]["val_loss"]
    eps_exact = (val_loss - h_exact) / (gen["log_alphabet"] - h_exact)
    report["model_sha256"] = pre["model_sha256"]
    report["eps_hat"] = {"exact": eps_exact, "in_regime_exact": eps_exact < ORACLE_IN_REGIME, "ngram": pre["oracle"]["eps_hat"],
                         "val_loss": val_loss, "entropy_exact": h_exact, "entropy_ngram": pre["oracle"]["entropy_floor"]}
    report["budget"] = pre["budget"]
    model, _ = DecoderLM.load(pre_dir / pre["model_file"], device)
    corpus = Corpus(scm_dir, "end", "request")
    vocab = Vocab.from_model_vocab(corpus.vocab_json())
    # 3. validation probe, τ chosen blind per arm, in-process freeze
    val_truth = load_truth(scm_dir, "val")
    truth_h = int(val_truth["history"])                     # per-lag recall is reported over the truth's own lag range
    val_recs, val_meta = probe_split(model, vocab, corpus, "val", val_truth, args, device)
    write_matrices(out_dir / "matrices-val.npz", val_recs, GATE_ARMS)
    freeze = {"model_sha256": pre["model_sha256"], "c": args.context, "N": args.particles, "g": args.guidance, "cells": {}}
    curves = {}
    for arm in GATE_ARMS:
        curves[arm] = f1_curve(val_recs, arm, GATE_TAU_GRID, vocab, args.max_lag)
        freeze["cells"][arm] = {"tau": select_tau(curves[arm])}
        log({"event": "val_curve", "arm": arm, "tau_selected": freeze["cells"][arm]["tau"], **curves[arm][freeze["cells"][arm]["tau"]]})
    write_json(out_dir / "freeze.json", freeze)
    report["val"] = {"meta": {k: v for k, v in val_meta.items()}, "curves": {a: {str(t): v for t, v in c.items()} for a, c in curves.items()}}
    del val_recs
    # 4. one test read
    test_recs, test_meta = probe_split(model, vocab, corpus, "test", load_truth(scm_dir, "test"), args, device)
    write_matrices(out_dir / "matrices-test.npz", test_recs, GATE_ARMS)
    report["test"] = {"meta": test_meta, "arms": {}}
    for arm in GATE_ARMS:
        tau_star = freeze["cells"][arm]["tau"]
        curve = f1_curve(test_recs, arm, sorted(set(GATE_TAU_GRID) | {tau_star, GATE_TAU_PRINTED}), vocab, args.max_lag)
        report["test"]["arms"][arm] = {
            "tau_selected": tau_star, "f1_selected_tau": curve[tau_star]["f1"], "f1_printed_tau": curve[GATE_TAU_PRINTED]["f1"],
            "precision_recall_selected": {k: curve[tau_star][k] for k in ("precision", "recall")},
            "precision_recall_printed": {k: curve[GATE_TAU_PRINTED][k] for k in ("precision", "recall")},
            "curve": {str(t): v for t, v in curve.items()},
            "per_lag_recall_selected": per_lag_recall(test_recs, arm, tau_star, truth_h),
            "per_lag_recall_printed": per_lag_recall(test_recs, arm, GATE_TAU_PRINTED, truth_h),
        }
    g = report["test"]["arms"][GATE_ARM]
    report["tau_selected"] = g["tau_selected"]
    report["f1_printed_tau"] = g["f1_printed_tau"]
    report["f1_selected_tau"] = g["f1_selected_tau"]
    report["per_lag_recall"] = g["per_lag_recall_selected"]
    report["assertions"] = [{"name": "redundancy_band", "value": report["redundancy"], "band": [args.redundancy_min, 1.0],
                             "passed": True, "scenario": 1}] + assertions_for(g["f1_printed_tau"], g["f1_selected_tau"])
    report["passed"] = all(a["passed"] for a in report["assertions"])
    paper = report["test"]["arms"][ARM_FAITHFUL_PAPER]
    report["paper_vs_library"] = {"f1_selected_tau": paper["f1_selected_tau"] - g["f1_selected_tau"],
                                  "f1_printed_tau": paper["f1_printed_tau"] - g["f1_printed_tau"]}
    return report


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                parents=[_without(generator_parser(), {"--output-folder", "--seed"}),
                                         _without(pretrain_parser(), {"--output-folder", "--seed", "--corpus", "--ordering", "--grain",
                                                                      "--max-len", "--device"})],
                                conflict_handler="resolve")
    p.add_argument("--particles", required=True, type=int, help="N")
    p.add_argument("--context", required=True, type=int, help="c, BOS-counted (the note: 6)")
    p.add_argument("--guidance", required=True, type=int, help="g < c real positions before the sampled history (the note: 3)")
    p.add_argument("--num-sequences", required=True, type=int, help="head sequences probed per split (the note: 1024)")
    p.add_argument("--max-lag", required=True, type=int)
    p.add_argument("--microbatch-rows", required=True, type=int)
    p.add_argument("--memory-cap-gb", required=True, type=float)
    p.add_argument("--clamp-eps", required=True, type=float, help="ε of the published float32 clamp, for the corrupted-cell count")
    p.add_argument("--probe-amp", required=True, choices=AMP_MODES,
                   help="autocast mode of the probe's forward passes, independent of --amp (pretraining); D-CB-20")
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--device", required=True, choices=DEVICES)
    p.add_argument("--output-folder", required=True)
    return p


def _without(parser, drop):
    """A parent parser minus the options the gate itself supplies."""
    parser.add_help = False
    for a in list(parser._actions):
        if any(o in drop for o in a.option_strings):
            parser._remove_action(a)
            for o in a.option_strings:
                parser._option_string_actions.pop(o, None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, "selfcheck", vars(args)) as rec:
        report = run_gate(args, rec.out_dir)
        gate_dir = rec.out_dir / GATE_DIR
        path = write_json(gate_dir / f"{report['date']}-gate-report.json", report)
        log({"event": "gate", "passed": report["passed"], "report": str(path),
             "assertions": [(a["name"], a["passed"]) for a in report.get("assertions", [])]})
        rec.finish({"passed": report["passed"], "report": str(path.relative_to(rec.out_dir)),
                    "assertions": report.get("assertions"), "engine_sha256": report["engine_sha256"]},
                   status="ok" if report["passed"] else "failed")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
