"""D6: the probe's read-out is microbatch-invariant, the lag bound is a true
bound (cells within the bound are identical whether or not later positions are
forwarded), and the memory estimator refuses above the cap naming the largest
microbatch that fits."""
import pytest
import torch

from tracecmibench.constants import BOS, EOS
from tracecmibench.model import DecoderLM
from tracecmibench.probe import MemoryRefusal, check_memory, estimate_peak_bytes, largest_microbatch, n_cells, probe_sequence
from tracecmibench.statistics import cell_statistics

REAL = range(4, 24)


def _model():
    torch.manual_seed(0)
    return DecoderLM(24, 2, 32, 4, 2.0, 0.0, 10000.0).eval()


def _ids():
    return torch.tensor([BOS, 5, 9, 7, 12, 6, 20, 8, 11, EOS])


def test_microbatch_equivalence():
    m = _model()
    ids = _ids()
    out = []
    for mb in (1, 7, 64):
        gen = torch.Generator().manual_seed(5)
        logp, X = probe_sequence(m, ids, c=2, history_mode="sampled", guidance=1, n_particles=6, max_lag=100,
                                 real_ids=REAL, generator=gen, microbatch_rows=mb, device="cpu", amp="none")
        out.append((logp, X))
    for logp, X in out[1:]:
        assert torch.equal(X, out[0][1]) and torch.allclose(logp, out[0][0], atol=1e-6)
    assert out[0][0].shape == (6, 9, 10)


def test_max_lag_true_bound():
    m = _model()
    ids = _ids()
    c = 2
    res = {}
    for M in (2, 100):
        gen = torch.Generator().manual_seed(9)
        logp, _ = probe_sequence(m, ids, c, "observed", 1, 5, M, REAL, gen, 16, "cpu", "none")
        res[M] = cell_statistics(logp, c, M, 1e-9)["safe"]["kl_mean_df"]
    band2 = ~torch.isnan(res[2])
    assert band2.sum() == n_cells(len(ids), c, 2)
    assert torch.allclose(res[2][band2], res[100][band2], atol=1e-6)
    assert torch.isnan(res[2]).sum() > torch.isnan(res[100]).sum()


def test_probe_reads_the_observed_token_not_the_row_token():
    """Row r's own token at the effect position is noise, but the gather is at the observed token:
    the fully observed row must reproduce the model's plain next-token log-probs."""
    m = _model()
    ids = _ids()
    gen = torch.Generator().manual_seed(1)
    logp, _ = probe_sequence(m, ids, 3, "observed", 1, 2, 100, REAL, gen, 8, "cpu", "none")
    plain = m.log_probs_at(ids[None], ids[None].eq(0), 1)[0]
    assert torch.allclose(logp[:, -1, :], plain[None].expand(2, -1), atol=1e-6)


def test_memory_refusal():
    m = _model()
    est = estimate_peak_bytes(m.n_params, 256, 66, 24, 32, 4, 128, 64)
    assert est > 6 * m.n_params
    info = check_memory(m, 256, 66, 128, 64, cap_gb=1.0)
    assert info["estimated_peak_bytes"] == est
    with pytest.raises(MemoryRefusal) as e:
        check_memory(m, 4096, 66, 128, 64, cap_gb=0.05)
    msg = str(e.value)
    assert "exceeds --memory-cap-gb 0.05" in msg and "largest microbatch that fits is" in msg
    fits = largest_microbatch(int(0.05e9), m.n_params, 66, 24, 32, 4, 128, 64)
    assert str(fits) in msg and estimate_peak_bytes(m.n_params, fits, 66, 24, 32, 4, 128, 64) <= 0.05e9
    assert estimate_peak_bytes(m.n_params, fits + 1, 66, 24, 32, 4, 128, 64) > 0.05e9
