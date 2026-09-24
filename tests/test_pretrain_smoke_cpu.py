"""`pretrain` end to end on the fixture, CPU, seconds: a model file whose sha256
the record carries, checkpoints before validations, the oracle score against
the independent floor, and the budget triple (PRD scenario 21)."""
import torch

from fixture_corpus import fixture_corpus
from tracecmibench import pretrain as pt
from tracecmibench.constants import RESULTS_JSON, RUN_DIR
from tracecmibench.model import DecoderLM, sha256_of
from tracecmibench.record import read_json

ARGS = ["--ordering", "end", "--grain", "session", "--max-len", "32", "--n-layers", "2", "--d-model", "32",
        "--n-heads", "4", "--ff-mult", "2", "--dropout", "0.0", "--rope-theta", "10000", "--lr", "3e-3",
        "--adam-beta1", "0.9", "--adam-beta2", "0.95", "--warmup-steps", "5", "--lr-schedule", "cosine",
        "--weight-decay", "0.01", "--batch-size", "16", "--steps", "40", "--grad-clip", "1.0", "--amp", "none",
        "--val-every", "20", "--val-batches", "4", "--checkpoint-every", "20", "--entropy-order", "2",
        "--seed", "0", "--device", "cpu"]


def test_pretrain_smoke(tmp_path):
    out = tmp_path / "pre"
    assert pt.main(["--corpus", str(fixture_corpus("latent")), *ARGS, "--output-folder", str(out)]) == 0
    r = read_json(out / RUN_DIR / RESULTS_JSON)
    assert r["status"] == "ok" and (out / "model.pt").exists()
    assert r["model_sha256"] == sha256_of(out / "model.pt")
    assert [c["step"] for c in r["checkpoints"]] == [20] and (out / "checkpoint-0000020.pt").exists()
    assert [v["step"] for v in r["val_curve"]] == [20, 40]
    assert r["val_curve"][-1]["loss"] < r["val_curve"][0]["loss"] + 0.5      # not diverging
    o = r["oracle"]
    assert set(o) >= {"eps_hat", "val_loss", "entropy_floor", "log_alphabet", "in_regime", "entropy_order"}
    assert o["entropy_floor"] == r["entropy"]["entropy_floor"] and o["entropy_order"] == 2
    assert 0 < o["entropy_floor"] < o["log_alphabet"]
    assert r["budget"] == {"configs_tried_on_val": 1, "total_steps": 40, "wall_clock_s": r["budget"]["wall_clock_s"]}
    assert r["tok_pos_per_s"] > 0 and r["params"] > 0 and r["vocab_size"] == 14
    m, extra = DecoderLM.load(out / "model.pt")
    assert extra["step"] == 40 and m.config["d_model"] == 32
    meta = read_json(out / RUN_DIR / "run_meta.json")
    assert meta["tok_pos_per_s"] == r["tok_pos_per_s"] and meta["status"] == "ok"


def test_seeded_runs_are_reproducible_on_cpu(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    args = [*ARGS]
    args[args.index("--steps") + 1] = "10"
    for out in (a, b):
        assert pt.main(["--corpus", str(fixture_corpus("latent")), *args, "--output-folder", str(out)]) == 0
    ma, _ = DecoderLM.load(a / "model.pt")
    mb, _ = DecoderLM.load(b / "model.pt")
    for pa, pb in zip(ma.parameters(), mb.parameters()):
        assert torch.equal(pa, pb)
