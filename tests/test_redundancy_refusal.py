"""PRD scenario 1: a world whose Shannon redundancy falls below the required band
is refused naming the value it reached, before any sequence is written; the
scale knob (D-CB-1) is what moves R."""
import numpy as np
import pytest

from tracecmibench import generator as gen
from tracecmibench.constants import RESULTS_JSON, RUN_DIR
from tracecmibench.record import read_json


def _scm(w_scale, seed=0):
    return gen.SCM(vocab_size=50, history=3, sparsity=0.9, w_scale=w_scale, decay_rate=1.0, embed_dim=4, hidden=8, seed=seed)


def test_redundancy_grows_with_the_scale():
    r = [_scm(w).redundancy(3000, 16, np.random.default_rng(0))["redundancy"] for w in (0.0, 2.0, 8.0)]
    assert 0.0 <= r[0] < r[1] < r[2] <= 1.0
    assert r[0] < 0.58                     # without the interaction matrix the world sits below the paper's band


def test_refusal_names_the_value_and_writes_a_failed_record(tmp_path):
    out = tmp_path / "low"
    rc = gen.main(["--vocab-size", "50", "--seq-len", "16", "--history", "3", "--sparsity", "0.9", "--w-scale", "0.5",
                   "--decay-rate", "1.0", "--embed-dim", "4", "--hidden", "8", "--redundancy-min", "0.58",
                   "--redundancy-mc", "3000", "--n-train", "10", "--n-val", "5", "--n-test", "5", "--truth-head", "0",
                   "--seed", "0", "--output-folder", str(out)])
    assert rc == 3
    r = read_json(out / RUN_DIR / RESULTS_JSON)
    assert r["status"] == "failed" and r["refused"] and "redundancy 0." in r["error"] and "0.58" in r["error"]
    assert not (out / "views").exists()


def test_refusal_is_an_exception_in_process():
    class A:
        vocab_size, seq_len, history, sparsity, w_scale, decay_rate, embed_dim, hidden = 50, 16, 3, 0.9, 0.5, 1.0, 4, 8
        redundancy_min, redundancy_mc, n_train, n_val, n_test, truth_head, seed = 0.9, 2000, 1, 1, 1, 0, 0
    with pytest.raises(gen.GeneratorRefusal, match="below the required 0.9"):
        gen.generate(A(), "/nonexistent")
