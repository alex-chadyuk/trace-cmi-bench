"""The Eq. 21 generator is a pure function of its arguments and seed, writes a
corpus the method-side adapter reads unchanged, and its worlds differ across seeds."""
import numpy as np
import pyarrow.parquet as pq

from tracecmibench import generator as gen
from tracecmibench.constants import RESULTS_JSON, RUN_DIR
from tracecmibench.corpus import Corpus
from tracecmibench.data import SequenceStore
from tracecmibench.record import read_json
from tracecmibench.vocab import Vocab

ARGS = ["--vocab-size", "30", "--seq-len", "12", "--history", "3", "--sparsity", "0.5", "--w-scale", "3.0",
        "--decay-rate", "1.0", "--embed-dim", "4", "--hidden", "8", "--redundancy-min", "0.0", "--redundancy-mc", "2000",
        "--n-train", "300", "--n-val", "40", "--n-test", "40", "--truth-head", "16"]


def _run(tmp_path, name, seed):
    out = tmp_path / name
    assert gen.main([*ARGS, "--seed", str(seed), "--output-folder", str(out)]) == 0
    return out


def test_same_seed_same_bytes_different_seed_different_world(tmp_path):
    a, b, c = _run(tmp_path, "a", 0), _run(tmp_path, "b", 0), _run(tmp_path, "c", 1)
    for split in ("train", "val", "test"):
        rel = f"views/end-request/sequences/split={split}/date=synthetic/part-0000.parquet"
        ta, tb, tc = (pq.read_table(p / rel) for p in (a, b, c))
        assert ta.equals(tb) and not ta.equals(tc)
    ta, tb = np.load(a / "truth" / "val-truth.npz"), np.load(b / "truth" / "val-truth.npz")
    assert np.array_equal(ta["kl"], tb["kl"]) and np.array_equal(ta["x"], tb["x"])
    ra, rc = read_json(a / "generator.json"), read_json(c / "generator.json")
    assert ra["redundancy"] == read_json(b / "generator.json")["redundancy"] != rc["redundancy"]


def test_corpus_is_readable_through_the_adapter(tmp_path):
    out = _run(tmp_path, "d", 0)
    c = Corpus(out, "end", "request")
    v = Vocab.from_model_vocab(c.vocab_json())
    assert v.size == 34 and v.tokens()[0] == "0:ok"
    store = SequenceStore.from_corpus(c, "val", v, 64)
    assert len(store) == 40 and (store.lengths == 14).all()          # BOS + 12 + EOS
    assert c.n_rows("train") == 300 == c.export_stats()["rows"]["end-request-train"]
    r = read_json(out / RUN_DIR / RESULTS_JSON)
    assert r["status"] == "ok" and r["rows"]["end-request-test"] == 40
    assert 0 < r["entropy"]["train"]["mean_entropy"] < r["log_alphabet"]
    assert r["truth"]["val"]["n_sequences"] == 16 and r["truth"]["test"]["n_sequences"] == 16
