"""PRD scenarios 2–3: the gate passes when the printed threshold fails to
replicate (F1 at most the registered ceiling; one-sided since the 2026-09-24
amendment) AND the blind-selected threshold replicates; a printed threshold
that unexpectedly replicates fails the gate. Plus the truth mapping,
τ selection and per-lag recall helpers."""
import numpy as np

from tracecmibench.constants import GATE_F1_PRINTED_MAX, GATE_F1_SELECTED_MIN
from tracecmibench.selfcheck import assertions_for, per_lag_recall, select_tau, truth_for_sequence
from tracecmibench.vocab import Vocab
from tracecmibench.generator import model_vocab


def test_gate_bands_logic():
    ok = assertions_for(0.46, 0.90)
    assert all(a["passed"] for a in ok)
    unexpected = assertions_for(0.91, 0.90)                      # the printed threshold replicates: a different engine
    assert not unexpected[0]["passed"] and unexpected[1]["passed"]
    low = assertions_for(0.46, 0.80)
    assert low[0]["passed"] and not low[1]["passed"]
    hi = GATE_F1_PRINTED_MAX
    assert assertions_for(hi, GATE_F1_SELECTED_MIN)[0]["passed"] and assertions_for(0.0, 1.0)[0]["passed"]
    assert assertions_for(0.35, 0.90)[0]["passed"]                # attempt 2's reading: below the former lower edge, still a non-replication
    assert not assertions_for(hi + 1e-9, 1.0)[0]["passed"]
    assert assertions_for(0.46, 0.90)[0]["band"] == [0.0, hi]
    assert {a["scenario"] for a in ok} == {2, 3}


def test_select_tau_prefers_larger_on_ties():
    c = {t: {"f1": v, "precision": 0.0, "recall": 0.0} for t, v in {1e-5: 0.5, 3e-5: 0.9, 1e-4: 0.9, 1e-3: 0.2}.items()}
    assert select_tau(c) == 1e-4


def test_truth_mapping_restricts_to_the_window():
    v = Vocab.from_model_vocab(model_vocab(10))
    # one sequence of 6 symbols; truth pairs (i, j) in generator positions
    truth = {"seq": np.array([0, 0, 0, 0]), "i": np.array([0, 1, 2, 3]), "j": np.array([1, 3, 4, 5]),
             "edge": np.array([True, True, True, False]), "kl": np.zeros(4)}
    x = np.array([7, 3, 5, 7, 2, 9])
    edges, pairs = truth_for_sequence(truth, 0, c=3, vocab=v, x_row=x)
    # cause at generator position i sits in the window iff i + 1 >= c: i = 2 only (0, 1 are context); pair (3, 5) is not an edge
    assert pairs == [(0, 2)] and edges == {(v.base_id(5), v.base_id(2))}
    edges2, pairs2 = truth_for_sequence(truth, 0, c=1, vocab=v, x_row=x)
    assert pairs2 == [(0, 1), (1, 3), (2, 4)]
    assert (v.base_id(7), v.base_id(3)) in edges2 and (v.base_id(3), v.base_id(7)) in edges2


def test_per_lag_recall_counts_hits_by_lag():
    S = np.full((5, 5), np.nan)
    S[0, 1] = 1.0; S[1, 2] = 0.0; S[0, 2] = 1.0; S[1, 3] = 0.0
    recs = [{"stats": {"a": S}, "truth_pairs": [(0, 1), (1, 2), (0, 2), (1, 3), (0, 4)]}]
    r = per_lag_recall(recs, "a", tau=0.5, max_lag=3)
    assert r == {"1": 0.5, "2": 0.5, "3": None}                  # (0, 4) has lag 4 > max_lag and is ignored
