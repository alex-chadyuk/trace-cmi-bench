"""`selfcheck` end to end on a tiny world, CPU, seconds: the report carries every
registered field, the in-process freeze predates the test read, the run
record says whether the gate passed, and the parser accepts the composed
knob set with nothing defaulted."""
from tracecmibench import selfcheck as sc
from tracecmibench.constants import GATE_TAU_PRINTED, RESULTS_JSON, RUN_DIR
from tracecmibench.hashes import engine_sha256
from tracecmibench.record import read_json

ARGS = [
    # generator
    "--vocab-size", "30", "--seq-len", "12", "--history", "3", "--sparsity", "0.5", "--w-scale", "3.0",
    "--decay-rate", "1.0", "--embed-dim", "4", "--hidden", "8", "--redundancy-min", "0.0", "--redundancy-mc", "2000",
    "--n-train", "200", "--n-val", "24", "--n-test", "24", "--truth-head", "8",
    # pretrain
    "--n-layers", "1", "--d-model", "16", "--n-heads", "2", "--ff-mult", "2", "--dropout", "0.0", "--rope-theta", "10000",
    "--lr", "3e-3", "--adam-beta1", "0.9", "--adam-beta2", "0.95", "--warmup-steps", "2", "--lr-schedule", "cosine",
    "--weight-decay", "0.0", "--batch-size", "16", "--steps", "12", "--grad-clip", "1.0", "--amp", "none",
    "--val-every", "6", "--val-batches", "2", "--checkpoint-every", "6", "--entropy-order", "1",
    # probe
    "--particles", "3", "--context", "3", "--guidance", "2", "--num-sequences", "8", "--max-lag", "11",
    "--microbatch-rows", "64", "--memory-cap-gb", "2", "--clamp-eps", "1e-9",
    "--seed", "0", "--device", "cpu",
]


def test_selfcheck_smoke(tmp_path):
    out = tmp_path / "gate"
    rc = sc.main([*ARGS, "--output-folder", str(out)])
    r = read_json(out / RUN_DIR / RESULTS_JSON)
    reports = sorted((out / "gate").glob("*-gate-report.json"))
    assert len(reports) == 1
    rep = read_json(reports[0])
    assert rep["engine_sha256"] == engine_sha256() == r["engine_sha256"]
    assert rep["gate_arm"] == "faithful/library" and rep["context"] == 3 and rep["particles"] == 3
    for k in ("redundancy", "w_scale", "model_sha256", "eps_hat", "tau_selected", "f1_printed_tau", "f1_selected_tau",
              "per_lag_recall", "assertions", "passed", "val", "test", "paper_vs_library", "budget"):
        assert k in rep, k
    assert {a["name"] for a in rep["assertions"]} == {"redundancy_band", "printed_threshold_does_not_replicate",
                                                       "selected_threshold_replicates"}
    assert rep["passed"] == all(a["passed"] for a in rep["assertions"]) == (rc == 0) == (r["status"] == "ok")
    assert set(rep["test"]["arms"]) == {"faithful/library", "faithful/paper"}
    lib = rep["test"]["arms"]["faithful/library"]
    assert str(GATE_TAU_PRINTED) in lib["curve"] and 0 <= lib["f1_selected_tau"] <= 1
    assert set(rep["per_lag_recall"]) == {str(k) for k in range(1, 12)}
    assert rep["eps_hat"]["exact"] is not None and rep["eps_hat"]["entropy_exact"] > 0
    freeze = read_json(out / "freeze.json")
    assert freeze["model_sha256"] == rep["model_sha256"] and set(freeze["cells"]) == set(rep["test"]["arms"])
    assert (out / "scm" / "generator.json").exists() and (out / "pretrain" / "model.pt").exists()
    assert rep["val"]["meta"]["n_sequences"] == 8 and rep["test"]["meta"]["n_sequences"] == 8


def test_parser_composes_every_knob_required():
    p = sc.build_parser()
    for a in p._actions:
        if a.option_strings and a.option_strings[0] not in ("-h", "--help"):
            assert a.required, a.option_strings
    names = {a.option_strings[0] for a in p._actions}
    assert {"--w-scale", "--steps", "--particles", "--context", "--guidance", "--max-lag", "--clamp-eps",
            "--output-folder", "--seed", "--device"} <= names
    assert "--corpus" not in names and "--max-len" not in names
