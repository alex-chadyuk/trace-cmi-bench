"""The paper's synthetic nonlinear SCM (Eq. 21) and its exact interventional truth
(App. E.1) — the substrate of the reproduction gate (PRD scenarios 1–3).

    python -m tracecmibench.generator --vocab-size 1000 --seq-len 64 --history 6 --sparsity 0.9 \
        --w-scale 3.2 --decay-rate 1.0 --embed-dim 16 --hidden 64 --redundancy-min 0.58 \
        --redundancy-mc 100000 --n-train 300000 --n-val 5000 --n-test 5000 --truth-head 1024 \
        --seed 0 --output-folder <corpus-dir>

    P(X_t | X_{t-h:t-1}) = softmax( b + sum_{k=1..h} e^{-r(k-1)} W[x_{t-k}]
                                    + ReLU([E_{x_{t-h}}, ..., E_{x_{t-1}}] W1) W2 )

`W` is sparse (Bernoulli(1 − sparsity) mask), mixed-sign (N(0, 1)) and
multiplied by `--w-scale` (D-CB-1: the released library cannot reach the
paper's redundancy band at unit scale); `E`, `W1`, `W2`, `b` are fixed random
draws of the world; positions with fewer than `h` predecessors use the lags
that exist and zero embeddings for the rest. The Shannon redundancy
`R = 1 − H(P) / log|X|` is estimated by Monte Carlo over `--redundancy-mc`
positions of the process's own contexts before any sequence is written; a
world below `--redundancy-min` is refused naming the value reached.

The corpus is written in the benchmark's method-readable views layout
(`views/end-request/…`, one op per symbol, outcome `ok`, a `model-vocab.json`)
so `pretrain` and `discover` run on it unchanged. The truth (D-CB-13) lives
outside the method-readable set under `truth/`: for the first `--truth-head`
sequences of `val` and `test`, every position pair `(i, j)` with
`1 <= j − i <= h` carries the Bernoulli KL between the event distributions of
`E_j = 1{X_j = x_j}`, factual first and the mixture over 10 uniform
replacements of `x_i` second (D-CB-13 as amended 2026-09-24);
an edge is `kl > δ = 0.05`. The exact conditional entropy of every generated
sequence is stored too, for the gate's oracle score (D-CB-7).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .constants import (
    BOS, EOS, GATE_TRUTH_COUNTERFACTUALS, GATE_TRUTH_DELTA, N_SPECIALS, OUTCOME_OK, PAD, UNK,
)
from .log import log
from .record import RunRecord, write_json

TRUTH_DIR = "truth"
GENERATOR_JSON = "generator.json"
VIEW = "end-request"
SPLIT_INDEX = {"train": 0, "val": 1, "test": 2}
GEN_BATCH = 4096
TRUTH_BATCH = 128


class GeneratorRefusal(RuntimeError):
    pass


def bernoulli_kl(a, b):
    """`KL(Bern(a) || Bern(b))` in nats, float64, inputs in (0, 1); the probability of an
    observed token under a softmax is never exactly 0 or 1 in float64 at these scales."""
    a = np.clip(a, 1e-300, 1 - 1e-16)
    b = np.clip(b, 1e-300, 1 - 1e-16)
    return a * np.log(a / b) + (1.0 - a) * np.log((1.0 - a) / (1.0 - b))


class SCM:
    """One world: the fixed random draws of Eq. 21."""

    def __init__(self, vocab_size, history, sparsity, w_scale, decay_rate, embed_dim, hidden, seed):
        self.V, self.h = int(vocab_size), int(history)
        self.sparsity, self.w_scale, self.decay_rate = float(sparsity), float(w_scale), float(decay_rate)
        self.embed_dim, self.hidden = int(embed_dim), int(hidden)
        self.seed = int(seed)
        rng = np.random.default_rng([seed, 0])          # world stream: construction only
        mask = rng.random((self.V, self.V)) >= sparsity
        self.W = rng.standard_normal((self.V, self.V)) * mask * w_scale
        self.E = rng.standard_normal((self.V, self.embed_dim))
        self.W1 = rng.standard_normal((self.h * self.embed_dim, self.hidden)) / math.sqrt(self.h * self.embed_dim)
        self.W2 = rng.standard_normal((self.hidden, self.V)) / math.sqrt(self.hidden)
        self.b = rng.standard_normal(self.V)
        self.decay = np.exp(-decay_rate * np.arange(self.h))      # lag k = 1..h -> decay[k-1]

    def config(self):
        return {"vocab_size": self.V, "history": self.h, "sparsity": self.sparsity, "w_scale": self.w_scale,
                "decay_rate": self.decay_rate, "embed_dim": self.embed_dim, "hidden": self.hidden, "seed": self.seed}

    # --- the conditional -------------------------------------------------------------------
    def logits(self, context):
        """`context`: int array `[n, h]` of the h most recent tokens, oldest first; −1 = absent.
        Returns `[n, V]` logits of the next token."""
        n = context.shape[0]
        out = np.broadcast_to(self.b, (n, self.V)).copy()
        emb = np.zeros((n, self.h * self.embed_dim))
        for k in range(1, self.h + 1):            # lag k = the k-th most recent = column h - k
            tok = context[:, self.h - k]
            present = tok >= 0
            if present.any():
                rows = np.where(present, tok, 0)
                out += (self.decay[k - 1] * self.W[rows]) * present[:, None]
                emb[:, (self.h - k) * self.embed_dim:(self.h - k + 1) * self.embed_dim] = self.E[rows] * present[:, None]
        out += np.maximum(emb @ self.W1, 0.0) @ self.W2
        return out

    @staticmethod
    def log_softmax(logits):
        m = logits.max(-1, keepdims=True)
        z = logits - m
        return z - np.log(np.exp(z).sum(-1, keepdims=True))

    @staticmethod
    def entropy(logp):
        p = np.exp(logp)
        return -(p * logp).sum(-1)

    # --- sampling ---------------------------------------------------------------------------
    def sample(self, n, L, rng):
        """`(x [n, L] int, entropy [n, L] float)`: sequences and the exact conditional entropy at every position."""
        x = np.full((n, L), -1, dtype=np.int64)
        ent = np.zeros((n, L))
        for t in range(L):
            ctx = self._context(x, t)
            logp = self.log_softmax(self.logits(ctx))
            ent[:, t] = self.entropy(logp)
            u = rng.random(n)
            cdf = np.cumsum(np.exp(logp), axis=-1)
            x[:, t] = np.minimum((cdf < u[:, None]).sum(-1), self.V - 1)
        return x, ent

    def _context(self, x, t):
        """The h tokens before position t (oldest first), −1 where absent."""
        n = x.shape[0]
        ctx = np.full((n, self.h), -1, dtype=np.int64)
        lo = max(0, t - self.h)
        if t > lo:
            ctx[:, self.h - (t - lo):] = x[:, lo:t]
        return ctx

    def redundancy(self, n_positions, L, rng):
        """`R = 1 − H(P) / log|X|` by Monte Carlo over the process's own contexts."""
        n_seq = max(1, math.ceil(n_positions / L))
        ents = []
        done = 0
        while done < n_seq:
            b = min(GEN_BATCH, n_seq - done)
            _, ent = self.sample(b, L, rng)
            ents.append(ent.ravel())
            done += b
        h = float(np.concatenate(ents)[:n_positions].mean())
        return {"redundancy": 1.0 - h / math.log(self.V), "mean_entropy": h, "n_positions": int(min(n_positions, n_seq * L))}

    # --- exact truth (D-CB-13, amended 2026-09-24) -------------------------------------------------
    def truth(self, x, delta, n_counterfactuals, rng):
        """For every `(i, j)` with `1 <= j − i <= h` (0-based positions): the Bernoulli KL
        between the event distributions of `E_j = 1{X_j = x_j}` — factual
        `p = P(X_j = x_j | x_<j)` first, the counterfactual mixture
        `q = mean_u P(X_j = x_j | x_<j, x_i := u)` over `n_counterfactuals` uniform `u` second
        (paper E.1: the KL "between post-intervention and observational distributions of
        E_t", the expectation over the do-operator inside as in Eq. 9). The 2026-09-24
        gate attempt had used the full categorical KL, which marks 68 % of pairs as edges;
        the Bernoulli reading gives 15 % with the note's lag-1 share (findings/gate-attempt-1-26-09-24.md).
        Returns arrays `seq, i, j, kl` (all candidate pairs) and the boolean edge mask."""
        n, L = x.shape
        seqs, iis, jjs, kls = [], [], [], []
        rows = np.arange(n)
        for j in range(1, L):
            ctx = self._context(x, j)                                   # [n, h]
            base = self.log_softmax(self.logits(ctx))                   # [n, V]
            obs = x[:, j]
            p_obs = np.exp(base[rows, obs])                             # factual probability of the observed token
            for k in range(1, min(self.h, j) + 1):
                i = j - k
                col = self.h - k
                u = rng.integers(0, self.V, size=(n, n_counterfactuals))
                cf = np.repeat(ctx, n_counterfactuals, axis=0)          # [n*C, h]
                cf[:, col] = u.ravel()
                lq = self.log_softmax(self.logits(cf)).reshape(n, n_counterfactuals, self.V)
                q_obs = np.exp(lq[rows[:, None], np.arange(n_counterfactuals)[None, :], obs[:, None]]).mean(1)
                kl = bernoulli_kl(p_obs, q_obs)                         # KL_B(p || q_bar)
                seqs.append(np.arange(n)); iis.append(np.full(n, i)); jjs.append(np.full(n, j)); kls.append(kl)
        seq = np.concatenate(seqs); i_ = np.concatenate(iis); j_ = np.concatenate(jjs); kl = np.concatenate(kls)
        return {"seq": seq.astype(np.int32), "i": i_.astype(np.int16), "j": j_.astype(np.int16),
                "kl": kl.astype(np.float64), "edge": kl > delta}


# --- corpus writer ------------------------------------------------------------------------------
def model_vocab(V):
    return {"version": 1, "normalizer_version": "synthetic-scm", "n_specials": N_SPECIALS,
            "specials": {"PAD": PAD, "BOS": BOS, "EOS": EOS, "UNK": UNK}, "vocab_size": N_SPECIALS + V,
            "base_ops": [{"id": i, "service": "scm", "name": f"/x/{i}", "kind": 1} for i in range(V)], "variants": []}


def _rows(x, split, offset):
    n, L = x.shape
    rows = []
    for r in range(n):
        ops = x[r].tolist()
        rows.append({"trace_id": f"{split}-{offset + r:08d}", "ops": ops, "outcomes": [OUTCOME_OK] * L,
                     "offsets": list(range(L)), "durations": [0] * L, "parent_pos": [-1] * L, "healthy": True,
                     "n_err_spans": 0, "n_spans": L, "truncated": False, "scenario_hash": "", "scenario_sid": -1,
                     "start_ns": 0, "attribution_level": 0, "sequence_kind": "request"})
    return rows


def write_corpus(scm, out, seq_len, n_by_split, truth_head, delta, n_counterfactuals):
    from tracebench.correlate.views import ARROW_SCHEMA

    out = Path(out)
    view = out / "views" / VIEW
    truth_dir = out / TRUTH_DIR
    truth_dir.mkdir(parents=True, exist_ok=True)
    rows_stat = {}
    entropy_by_split = {}
    truth_meta = {}
    for split, n in n_by_split.items():
        rng = np.random.default_rng([scm.seed, 1, SPLIT_INDEX[split]])
        part_dir = view / "sequences" / f"split={split}" / "date=synthetic"
        part_dir.mkdir(parents=True, exist_ok=True)
        writer = None
        ent_sum, ent_n = 0.0, 0
        head_x = []
        done = 0
        part = 0
        while done < n:
            b = min(GEN_BATCH, n - done)
            x, ent = scm.sample(b, seq_len, rng)
            ent_sum += float(ent.sum()); ent_n += ent.size
            if split != "train" and len(head_x) * GEN_BATCH < truth_head:
                head_x.append(x)
            table = pa.Table.from_pylist(_rows(x, split, done), schema=ARROW_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(part_dir / f"part-{part:04d}.parquet", ARROW_SCHEMA)
            writer.write_table(table)
            done += b
        if writer is not None:
            writer.close()
        rows_stat[f"{VIEW}-{split}"] = n
        entropy_by_split[split] = {"mean_entropy": ent_sum / max(ent_n, 1), "n_positions": ent_n}
        log({"event": "generated", "split": split, "n": n, "mean_entropy": entropy_by_split[split]["mean_entropy"]})
        if split != "train" and truth_head > 0 and head_x:
            hx = np.concatenate(head_x)[:truth_head]
            trng = np.random.default_rng([scm.seed, 2, SPLIT_INDEX[split]])
            t = scm.truth(hx, delta, n_counterfactuals, trng)
            np.savez(truth_dir / f"{split}-truth.npz", x=hx.astype(np.int32), **t,
                     delta=np.float64(delta), n_counterfactuals=np.int64(n_counterfactuals), history=np.int64(scm.h))
            truth_meta[split] = {"n_sequences": int(hx.shape[0]), "n_pairs": int(len(t["kl"])), "n_edges": int(t["edge"].sum()),
                                 "edge_rate": float(t["edge"].mean()) if len(t["kl"]) else None,
                                 "lag_share": {str(k): float(((t["j"] - t["i"]) == k)[t["edge"]].mean()) if t["edge"].any() else None
                                               for k in range(1, scm.h + 1)}}
            log({"event": "truth", "split": split, **{k: v for k, v in truth_meta[split].items() if k != "lag_share"}})
    (view / "model-vocab.json").write_text(json.dumps(model_vocab(scm.V), sort_keys=True))
    write_json(view / "export-stats.json", {"rows": rows_stat, "vocab_size": N_SPECIALS + scm.V, "k_ops": scm.V, "n_variants": 0,
                                            "alphabet_size_realized_train": scm.V, "ordering": "end", "grain": "request"})
    return rows_stat, entropy_by_split, truth_meta


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vocab-size", required=True, type=int)
    p.add_argument("--seq-len", required=True, type=int)
    p.add_argument("--history", required=True, type=int, help="h, the lag window of Eq. 21")
    p.add_argument("--sparsity", required=True, type=float)
    p.add_argument("--w-scale", required=True, type=float, help="multiplier on W (D-CB-1)")
    p.add_argument("--decay-rate", required=True, type=float, help="r in e^{-r(k-1)}; 1.0 is the paper")
    p.add_argument("--embed-dim", required=True, type=int)
    p.add_argument("--hidden", required=True, type=int)
    p.add_argument("--redundancy-min", required=True, type=float, help="refuse a world below this R (scenario 1)")
    p.add_argument("--redundancy-mc", required=True, type=int, help="positions sampled for the R estimate")
    p.add_argument("--n-train", required=True, type=int)
    p.add_argument("--n-val", required=True, type=int)
    p.add_argument("--n-test", required=True, type=int)
    p.add_argument("--truth-head", required=True, type=int, help="head sequences of val and test given an exact truth (0 = none)")
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--output-folder", required=True)
    return p


def generate(args, out_dir):
    scm = SCM(args.vocab_size, args.history, args.sparsity, args.w_scale, args.decay_rate, args.embed_dim, args.hidden, args.seed)
    red = scm.redundancy(args.redundancy_mc, args.seq_len, np.random.default_rng([args.seed, 3]))
    log({"event": "redundancy", **red, "redundancy_min": args.redundancy_min})
    if red["redundancy"] < args.redundancy_min:
        raise GeneratorRefusal(
            f"redundancy {red['redundancy']:.4f} is below the required {args.redundancy_min} "
            f"(w_scale {args.w_scale}, sparsity {args.sparsity}); raise --w-scale")
    rows, ent, truth = write_corpus(scm, out_dir, args.seq_len,
                                    {"train": args.n_train, "val": args.n_val, "test": args.n_test},
                                    args.truth_head, GATE_TRUTH_DELTA, GATE_TRUTH_COUNTERFACTUALS)
    res = {"scm": scm.config(), "seq_len": args.seq_len, "redundancy": red, "rows": rows, "entropy": ent,
           "log_alphabet": math.log(args.vocab_size), "truth": truth,
           "truth_delta": GATE_TRUTH_DELTA, "truth_counterfactuals": GATE_TRUTH_COUNTERFACTUALS, "view": VIEW}
    write_json(Path(out_dir) / GENERATOR_JSON, res)
    return res


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, "generator", vars(args)) as rec:
        try:
            res = generate(args, rec.out_dir)
        except GeneratorRefusal as e:
            rec.finish({"error": str(e), "refused": True}, status="failed")
            log({"event": "generator_refused", "reason": str(e)})
            return 3
        rec.finish({k: v for k, v in res.items() if k != "truth"} | {"truth": res["truth"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
