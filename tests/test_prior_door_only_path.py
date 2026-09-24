"""PRD non-negotiable 2 and scenario 19: the prior door opens exactly one
corpus path, records the read, and no module outside the prior arms imports it."""
from pathlib import Path

import pytest

from fixture_corpus import fixture_corpus
from tracecmibench import prior_door
from tracecmibench.constants import PRIOR_RELATIVE_PATH

SRC = Path(__file__).resolve().parents[1] / "src" / "tracecmibench"


def test_opens_the_prior_and_records_the_read(capsys):
    prior, reads = prior_door.open_prior(fixture_corpus("latent"))
    assert reads == [PRIOR_RELATIVE_PATH] == ["topology/prior.json"]
    assert prior["edges"] and prior["default_prob"] == 0.01 and prior["reverse_prob"] == 0.005
    assert '"event": "prior_read"' in capsys.readouterr().out


@pytest.mark.parametrize("rel", ["graphs/scoring-target.json", "graphs/alphabet.json", "labels/cases.json",
                                 "topology/callgraph.json", "manifest.json", "views/end-request/model-vocab.json",
                                 "topology/../graphs/alphabet.json"])
def test_refuses_every_other_path(rel):
    with pytest.raises(prior_door.NotPriorReadable):
        prior_door.open_prior_path(fixture_corpus("latent"), rel)


def test_only_prior_arms_import_the_door():
    offenders = []
    for p in sorted(SRC.glob("*.py")):
        if p.name == "prior_door.py" or p.name.startswith("prior_"):
            continue
        if "prior_door" in p.read_text(encoding="utf-8"):
            offenders.append(p.name)
    assert not offenders, offenders
