"""M6 end to end on the fixture, CPU, seconds: pretrain → blind val sweep → scoresweep
(the benchmark's scorer) → freeze in a git repository → one test read under the
freeze → annotate → five-seed report with paired differences; plus every refusal
the PRD names (scenarios 4, 6, 8, 9, 12, 14, 22, 23) and the freeze overwrite guard."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from tracebench.score import score_corpus

from fixture_corpus import fixture_corpus
from tracecmibench import annotate as an
from tracecmibench import discover as dc
from tracecmibench import freeze as fz
from tracecmibench import pretrain as pt
from tracecmibench import report as rp
from tracecmibench import scoresweep as ss
from tracecmibench import sweep as sw
from tracecmibench.constants import AGGREGATIONS, GRAINS, RESULTS_JSON, RUN_DIR, VAL_TABLE_JSON
from tracecmibench.hashes import GateRefusal, engine_sha256
from tracecmibench.record import read_json, write_json

PRETRAIN = ["--ordering", "end", "--grain", "session", "--max-len", "24", "--n-layers", "1", "--d-model", "16",
            "--n-heads", "2", "--ff-mult", "2", "--dropout", "0.0", "--rope-theta", "10000", "--lr", "3e-3",
            "--adam-beta1", "0.9", "--adam-beta2", "0.95", "--warmup-steps", "2", "--lr-schedule", "cosine",
            "--weight-decay", "0.0", "--batch-size", "16", "--steps", "8", "--grad-clip", "1.0", "--amp", "none",
            "--val-every", "4", "--val-batches", "2", "--checkpoint-every", "4", "--entropy-order", "1",
            "--seed", "0", "--device", "cpu"]
PROBE = ["--max-len", "24", "--max-lag", "6", "--sequence-sample", "head", "--microbatch-rows", "64",
         "--memory-cap-gb", "2", "--divergence", "log1mexp", "--clamp-eps", "1e-9", "--probe-amp", "none",
         "--seed", "0", "--device", "cpu"]
TAUS = ["1e-6", "1e-4", "1e-2", "1e-1"]
ARMS = ["faithful/paper", "faithful/library"]


def _git(args, cwd, env=None):
    e = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x", **(env or {})}
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, env=e).stdout.strip()


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Everything up to and including a committed freeze, shared by the tests below."""
    root = tmp_path_factory.mktemp("m6")
    corpus = fixture_corpus("latent")
    assert pt.main(["--corpus", str(corpus), *PRETRAIN, "--output-folder", str(root / "pre")]) == 0
    gate = root / "gate-report.json"
    write_json(gate, {"passed": True, "engine_sha256": engine_sha256(), "assertions": []})
    model = str(root / "pre" / "model.pt")
    assert sw.main(["--corpus", str(corpus), "--ordering", "end", "--grains", *GRAINS, "--split", "val", "--model", model,
                    "--arms", *ARMS, "--contexts", "2", "3", "--particles-grid", "2", "--guidance", "3",
                    "--num-sequences", "0", *PROBE, "--gate", str(gate), "--output-folder", str(root / "sweep")]) == 0
    assert ss.main(["--corpus", str(corpus), "--sweep-dir", str(root / "sweep"), "--grains", *GRAINS, "--taus", *TAUS,
                    "--output-folder", str(root / "scoresweep")]) == 0
    repo = root / "repo"
    repo.mkdir()
    _git(["init", "-q"], repo)
    assert fz.main(["--val-tables", str(root / "scoresweep" / VAL_TABLE_JSON), "--rung", "xs", "--variant", "latent", "--seed", "0",
                    "--freezes-dir", str(repo / "freezes"), "--output-folder", str(root / "freeze-run")]) == 0
    freeze = next((repo / "freezes").glob("*-xs-latent-s0.json"))
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "freeze"], repo, env={"GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z", "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z"})
    return {"root": root, "corpus": corpus, "gate": gate, "model": model, "freeze": freeze, "repo": repo,
            "freeze_doc": read_json(freeze), "pretrain_results": root / "pre" / RUN_DIR / RESULTS_JSON}


def _discover(w, arm, grain, split, out, aggs, freeze=None, extra=()):
    cells = {a: w["freeze_doc"]["cells"][f"{arm}/{a}/{grain}"] for a in aggs}
    c = {v["c"] for v in cells.values()}; N = {v["N"] for v in cells.values()}
    assert len(c) == 1 and len(N) == 1, "the test fixture's sub-arms froze at different (c, N); write one sub-arm per read"
    return dc.main(["--corpus", str(w["corpus"]), "--ordering", "end", "--grain", grain, "--split", split, "--model", w["model"],
                    "--arm", arm, "--particles", str(N.pop()), "--guidance", str(min(3, list(c)[0])), "--context", str(c.pop()),
                    "--num-sequences", "0", "--aggs", *aggs, "--taus", *[str(cells[a]["tau"]) for a in aggs],
                    "--per-lag-files", "3", "--freeze", str(freeze or ""), *PROBE, *extra, "--gate", str(w["gate"]),
                    "--output-folder", str(out)])


# --- sweep / scoresweep / freeze --------------------------------------------------------------------
def test_sweep_and_val_table(world):
    r = read_json(world["root"] / "sweep" / RUN_DIR / RESULTS_JSON)
    assert r["status"] == "ok" and r["n_cells"] == 2 * 2 * 1 * 2 and all(c["g"] == min(3, c["c"]) for c in r["cells"])
    assert sorted(Path(world["root"] / "sweep").glob("scores-faithful-*-c*-N2-*.npz"))
    t = read_json(world["root"] / "scoresweep" / VAL_TABLE_JSON)
    assert t["n_rows"] == 8 * len(AGGREGATIONS) * len(TAUS) and t["tool_version"] == "0.3.0" and t["config_hash"] == "fixture-latent-0"
    assert all(set(row["directed"]) >= {"precision", "recall", "f1", "shd"} for row in t["cells"])
    cov = next(iter(t["coverage"].values()))
    assert 0 <= cov["coverage"]["reachable_recall_ceiling"] <= 1 and cov["coverage"]["universe_ordered_pairs"] > 0


def test_sweep_refuses_test(world, tmp_path):
    with pytest.raises(sw.SweepRefusal):
        sw.main(["--corpus", str(world["corpus"]), "--ordering", "end", "--grains", "request", "--split", "test", "--model", world["model"],
                 "--arms", ARMS[0], "--contexts", "2", "--particles-grid", "2", "--guidance", "3", "--num-sequences", "0", *PROBE,
                 "--gate", str(world["gate"]), "--output-folder", str(tmp_path / "s")])
    assert read_json(tmp_path / "s" / RUN_DIR / RESULTS_JSON)["status"] == "failed"     # the refusal leaves a record


def test_freeze_rule_and_refuses_overwrite(world, tmp_path):
    f = world["freeze_doc"]
    assert set(f["cells"]) == {f"{a}/{g}/{gr}" for a in ARMS for g in AGGREGATIONS for gr in GRAINS}
    t = read_json(world["root"] / "scoresweep" / VAL_TABLE_JSON)
    for key, cell in f["cells"].items():
        arm, agg, grain = key.rsplit("/", 2)
        rows = [r for r in t["cells"] if (r["arm"], r["agg"], r["grain"]) == (arm, agg, grain)]
        best = max(r["directed"]["f1"] for r in rows)
        ties = [r for r in rows if r["directed"]["f1"] == best]
        pick = min(ties, key=lambda r: (r["N"], r["c"], -r["tau"]))            # smaller N, smaller c, larger tau
        assert (cell["tau"], cell["c"], cell["N"]) == (pick["tau"], pick["c"], pick["N"]) and cell["val_directed_f1"] == best
    assert f["model_sha256"] == read_json(world["pretrain_results"])["model_sha256"] and f["corpus_id"] == world["corpus"].name
    with pytest.raises(fz.FreezeExists):
        fz.main(["--val-tables", str(world["root"] / "scoresweep" / VAL_TABLE_JSON), "--rung", "xs", "--variant", "latent", "--seed", "0",
                 "--freezes-dir", str(world["repo"] / "freezes"), "--output-folder", str(tmp_path / "again")])


def test_freeze_merges_per_grain_tables(world, tmp_path):
    t = read_json(world["root"] / "scoresweep" / VAL_TABLE_JSON)
    parts = []
    for grain in GRAINS:
        part = {**t, "grains": [grain], "cells": [r for r in t["cells"] if r["grain"] == grain],
                "coverage": {k: v for k, v in t["coverage"].items() if f"/{grain}/" in k}}
        write_json(tmp_path / f"val-table-{grain}.json", part)
        parts.append(str(tmp_path / f"val-table-{grain}.json"))
    assert fz.main(["--val-tables", *parts, "--rung", "s", "--variant", "latent", "--seed", "1",
                    "--freezes-dir", str(tmp_path / "fr"), "--output-folder", str(tmp_path / "run")]) == 0
    merged = read_json(next((tmp_path / "fr").glob("*-s-latent-s1.json")))
    assert merged["cells"] == world["freeze_doc"]["cells"] and merged["grains"] == list(GRAINS) and len(merged["val_tables"]) == 2
    bad = {**t, "model_sha256": "0" * 64}
    write_json(tmp_path / "bad.json", bad)
    with pytest.raises(ValueError, match="disagree on model_sha256"):
        fz.main(["--val-tables", parts[0], str(tmp_path / "bad.json"), "--rung", "s", "--variant", "latent", "--seed", "2",
                 "--freezes-dir", str(tmp_path / "fr"), "--output-folder", str(tmp_path / "run2")])


# --- discover: gate, freeze and the test read -------------------------------------------------------------
def test_discover_refuses_without_gate(world, tmp_path):
    stale = tmp_path / "stale.json"
    write_json(stale, {"passed": True, "engine_sha256": "0" * 64, "assertions": []})
    with pytest.raises(GateRefusal):
        dc.main(["--corpus", str(world["corpus"]), "--ordering", "end", "--grain", "request", "--split", "val", "--model", world["model"],
                 "--arm", ARMS[1], "--particles", "2", "--guidance", "2", "--context", "2", "--num-sequences", "4", "--aggs", "max",
                 "--taus", "0.01", "--per-lag-files", "2", "--freeze", "", *PROBE, "--gate", str(stale), "--output-folder", str(tmp_path / "d")])


def test_discover_test_requires_freeze(world, tmp_path):
    with pytest.raises(dc.FreezeRefusal, match="needs --freeze"):
        _discover(world, ARMS[1], "request", "test", tmp_path / "nofreeze", ["max"], freeze=None)


def test_freeze_commit_assert(world, tmp_path):
    w = world
    # an uncommitted copy of the freeze is refused
    loose = w["repo"] / "freezes" / "loose.json"
    shutil.copy(w["freeze"], loose)
    with pytest.raises(dc.FreezeRefusal, match="commit it before the test read"):
        _discover(w, ARMS[1], "request", "test", tmp_path / "loose", ["max"], freeze=loose)
    loose.unlink()
    # a freeze committed after the run started is refused
    late = w["repo"] / "freezes" / "late.json"
    shutil.copy(w["freeze"], late)
    _git(["add", "-A"], w["repo"])
    _git(["commit", "-q", "-m", "late"], w["repo"], env={"GIT_COMMITTER_DATE": "2035-01-01T00:00:00Z", "GIT_AUTHOR_DATE": "2035-01-01T00:00:00Z"})
    with pytest.raises(dc.FreezeRefusal, match="started at"):
        _discover(w, ARMS[1], "request", "test", tmp_path / "late", ["max"], freeze=late)
    _git(["rm", "-q", "--cached", str(late)], w["repo"]); late.unlink(); _git(["commit", "-q", "-m", "rm"], w["repo"])
    # a freeze outside any repository is refused
    outside = tmp_path / "outside.json"
    shutil.copy(w["freeze"], outside)
    with pytest.raises(dc.FreezeRefusal):
        _discover(w, ARMS[1], "request", "test", tmp_path / "outside", ["max"], freeze=outside)
    # values that disagree with the freeze are refused (scenario 23)
    with pytest.raises(dc.FreezeRefusal, match="command line says"):
        _discover(w, ARMS[1], "request", "test", tmp_path / "wrongtau", ["max"], freeze=w["freeze"], extra=["--taus", "0.123"])


def _read_cell(world, arm, agg, grain, out):
    aggs = [agg]
    assert _discover(world, arm, grain, "test", out, aggs, freeze=world["freeze"]) == 0
    r = read_json(out / RUN_DIR / RESULTS_JSON)
    assert r["status"] == "ok" and r["freeze"]["sha"] and r["predictions"][agg]["per_lag_files"] == 3
    return r


def test_discover_test_read_and_annotate(world, tmp_path):
    w = world
    for arm in ARMS:
        cell = w["freeze_doc"]["cells"][f"{arm}/max/request"]
        out = tmp_path / arm.replace("/", "-")
        r = _read_cell(w, arm, "max", "request", out)
        assert (out / "scores-request.npz").exists() and (out / "matrices-request.npz").exists()
        assert (out / "prediction-request-max.json").exists() and (out / "ranking-request-max-lag3.json").exists()
        assert set(r["counts"]["compat_counts"]) == {"kl_mean_bf", "kl_mean_df", "mean_kl_bf", "mean_kl_df"}
        m = np.load(out / "matrices-request.npz")
        assert set(m.files) >= {"seq", "j", "q", "lag", "kl_mean_df"} and (m["lag"] >= 1).all()
        res = tmp_path / "results" / "xs" / "latent" / "seed=0" / arm / "max" / "request"
        assert an.main(["--corpus", str(w["corpus"]), "--run-dir", str(out), "--arm", arm, "--agg", "max", "--grain", "request",
                        "--pretrain-results", str(w["pretrain_results"]), "--per-lag-files", "3", "--output-folder", str(res)]) == 0
        a = read_json(res / "annotate.json")
        # scenario 8: the structural limitation is machine-readable with the count of unreachable bidirected truth
        assert a["structural_limitation"]["assumption"] == "causal_sufficiency" and a["structural_limitation"]["truth_bidirected_edges"] == 2
        assert a["score"]["bidirected"]["recall"] == 0.0
        # metrics are the scorer's own, byte for byte (non-negotiable 8)
        direct = score_corpus(w["corpus"], read_json(out / "prediction-request-max.json"), grain="request")
        assert json.dumps(read_json(res / "score.json"), sort_keys=True) == json.dumps(direct, sort_keys=True)
        assert json.dumps(a["score"], sort_keys=True) == json.dumps(direct, sort_keys=True)
        assert a["frozen"]["tau"] == cell["tau"] and a["frozen"]["c"] == cell["c"] and a["config_hash"] == "fixture-latent-0"
        assert a["causal_validity"]["value"] is None and "ADMG" in a["causal_validity"]["reason"]       # scenario 11 on latent
        assert set(a["per_lag_recall"]) == {"1", "2", "3"} and a["unreachable_tokens"]["unreachable"] == 2
        assert a["coverage"]["reachable_recall_ceiling"] <= 1 and a["oracle"]["in_regime"] in (True, False)
    world["results"] = tmp_path / "results"


# --- report ------------------------------------------------------------------------------------------------
def _five_seeds(results, model_override=None):
    """Fabricate seeds 1–4 from seed 0 (the report needs five; the fixture has one)."""
    base = results / "xs" / "latent" / "seed=0"
    for k in range(1, 5):
        dst = results / "xs" / "latent" / f"seed={k}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(base, dst)
        for ann in dst.rglob("annotate.json"):
            a = read_json(ann)
            a["seed"] = k
            if model_override and model_override in str(ann):
                a["model_sha256"] = "f" * 64
            write_json(ann, a)


def test_report_five_seed_refusal(world, tmp_path):
    results = world["results"]
    reasons = tmp_path / "reasons.json"
    write_json(reasons, {f"{a}/{g}/{gr}": "not read in this test" for a in ARMS for g in AGGREGATIONS for gr in GRAINS if not (g == "max" and gr == "request")})
    for k in range(1, 5):
        shutil.rmtree(results / "xs" / "latent" / f"seed={k}", ignore_errors=True)
    with pytest.raises(rp.ReportRefusal, match="at least 5"):
        rp.main(["--results-dir", str(results), "--rung", "xs", "--variant", "latent", "--pairs", "faithful/paper:faithful/library",
                 "--reasons", str(reasons), "--output-folder", str(tmp_path / "r1")])
    _five_seeds(results)
    with pytest.raises(rp.ReportRefusal, match="absent on every seed"):                     # scenario 12: no reason given
        rp.main(["--results-dir", str(results), "--rung", "xs", "--variant", "latent", "--pairs", "faithful/paper:faithful/library",
                 "--reasons", "", "--output-folder", str(tmp_path / "r2")])
    assert rp.main(["--results-dir", str(results), "--rung", "xs", "--variant", "latent", "--pairs", "faithful/paper:faithful/library",
                    "--reasons", str(reasons), "--output-folder", str(tmp_path / "r3")]) == 0
    rep = read_json(tmp_path / "r3" / "xs-latent.json")
    assert set(rep["cells"]) == {f"{a}/max/request" for a in ARMS} and len(rep["absent"]) == 6
    cell = rep["cells"]["faithful/library/max/request"]
    assert cell["n_seeds"] == 5 and cell["metrics"]["directed.f1"]["n_seeds"] == 5 and "reason" in cell["metrics"]["causal_validity.sid"]
    assert cell["structural_limitation"]["truth_bidirected_edges"] == [2] * 5
    md = (tmp_path / "r3" / "xs-latent.md").read_text()
    assert "structural: causal_sufficiency" in md and "paired differences" in md
    pair = rep["pairs"]["faithful/paper:faithful/library/max/request"]
    assert pair["seeds"] == [0, 1, 2, 3, 4] and pair["metrics"]["directed.f1"]["n_seeds"] == 5 and all(t[0] == world["corpus"].name for t in pair["triples"])


def test_report_paired_cells(world, tmp_path):
    results = world["results"]
    reasons = tmp_path / "reasons.json"
    write_json(reasons, {f"{a}/{g}/{gr}": "not read" for a in ARMS for g in AGGREGATIONS for gr in GRAINS if not (g == "max" and gr == "request")})
    _five_seeds(results, model_override="faithful/paper")
    with pytest.raises(rp.ReportRefusal, match="model_sha256"):
        rp.main(["--results-dir", str(results), "--rung", "xs", "--variant", "latent", "--pairs", "faithful/paper:faithful/library",
                 "--reasons", str(reasons), "--output-folder", str(tmp_path / "r")])
    _five_seeds(results)
