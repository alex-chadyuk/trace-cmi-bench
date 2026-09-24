"""PRD scenario 7: a method run reads a corpus only through the benchmark's
accessor; any read outside the raw feed and the views is refused and no
method module even names a ground-truth path."""
import re
from pathlib import Path

import pytest
from tracebench.allowlist import NotMethodReadable

from fixture_corpus import fixture_corpus
from tracecmibench.corpus import Corpus

SRC = Path(__file__).resolve().parents[1] / "src" / "tracecmibench"
# modules on a method code path: none may name a non-method-readable corpus path
METHOD_MODULES = ("corpus", "vocab", "data", "model", "staircase", "probe", "statistics", "project", "select",
                  "prediction", "pretrain", "entropy", "discover", "sweep", "prepare")
FORBIDDEN = re.compile(r"graphs/|labels/|oracle/|topology/|manifest\.json|alphabet\.json|scoring-target|instantiation\.json")


def test_reads_go_through_the_door():
    c = Corpus(fixture_corpus("latent"), "end", "session")
    assert c.vocab_json()["vocab_size"] == 14
    assert c.export_stats()["rows"]["end-session-train"] == c.n_rows("train")
    rows = list(c.sequences("val"))
    assert rows and all(len(ops) == len(outs) == n for _, ops, outs, n in rows)
    assert all(isinstance(ops[0], int) for _, ops, _, _ in rows)


@pytest.mark.parametrize("rel", ["graphs/scoring-target.json", "graphs/alphabet.json", "labels/cases.json",
                                 "topology/prior.json", "manifest.json", "instantiation.json", "COMPLETE",
                                 "../outside", "views/../graphs/alphabet.json"])
def test_refuses_everything_outside_the_views(rel):
    c = Corpus(fixture_corpus("latent"), "end", "request")
    with pytest.raises(NotMethodReadable):
        c.open(rel)


def test_bad_view_names_are_refused():
    with pytest.raises(ValueError):
        Corpus(fixture_corpus("latent"), "middle", "request")
    with pytest.raises(ValueError):
        Corpus(fixture_corpus("latent"), "end", "trace")
    with pytest.raises(ValueError):
        Corpus(fixture_corpus("latent"), "end", "request").parts("holdout")


def test_method_modules_never_name_a_ground_truth_path():
    offenders = []
    for name in METHOD_MODULES:
        p = SRC / f"{name}.py"
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if FORBIDDEN.search(line):
                offenders.append(f"{name}.py:{i}: {line.strip()}")
    assert not offenders, "\n".join(offenders)
