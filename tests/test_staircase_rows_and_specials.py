"""D-CB-2/3/9/10: the staircase's rows, its noise support, the shared noise
across rows, the two history sources and the lag-bound row padding."""
import pytest
import torch

from tracecmibench.constants import BOS, EOS, PAD, UNK
from tracecmibench.model import DecoderLM
from tracecmibench.staircase import build_history, build_staircase, draw_noise, row_pad_mask, sample_history

REAL = range(4, 20)


def _ids(L=8):
    return torch.tensor([BOS, *range(5, 5 + L - 2), EOS])


def test_rows_keep_a_growing_observed_prefix_and_row_zero_is_all_noise():
    ids = _ids()
    c, N = 2, 3
    g = torch.Generator().manual_seed(0)
    noise = draw_noise(REAL, N, len(ids), g, "cpu")
    hist = ids[:c].repeat(N, 1)
    X = build_staircase(ids, c, noise, hist)
    L = len(ids)
    Lc = L - c
    assert X.shape == (N, Lc + 1, L)
    for r in range(Lc + 1):
        obs = X[:, r, c:c + r]
        assert (obs == ids[c:c + r]).all()
        noised = X[:, r, c + r:]
        assert (noised == noise[:, c + r:]).all()
    assert (X[:, 0, c:] == noise[:, c:]).all() and (X[:, Lc] == ids).all()
    assert (X[:, :, :c] == ids[:c]).all()


def test_noise_is_uniform_over_real_tokens_only():
    g = torch.Generator().manual_seed(1)
    noise = draw_noise(REAL, 64, 50, g, "cpu")
    assert noise.min() >= 4 and noise.max() < 20
    assert not ((noise == PAD) | (noise == BOS) | (noise == EOS) | (noise == UNK)).any()
    counts = torch.bincount(noise.flatten(), minlength=20)[4:20].float()
    assert (counts / counts.sum()).min() > 0.03          # roughly uniform over 16 tokens


def test_common_random_numbers_adjacent_rows_differ_at_one_position():
    ids = _ids(10)
    c, N = 3, 4
    noise = draw_noise(REAL, N, len(ids), torch.Generator().manual_seed(2), "cpu")
    # make every noise draw differ from the observed token so the difference count is exact
    noise = torch.where(noise == ids, (noise + 1 - 4) % 16 + 4, noise)
    X = build_staircase(ids, c, noise, ids[:c].repeat(N, 1))
    for r in range(X.shape[1] - 1):
        diff = (X[:, r] != X[:, r + 1])
        assert (diff.sum(-1) == 1).all() and (diff[:, c + r]).all()


def test_row_pad_mask_bounds_each_row_by_the_lag():
    m = row_pad_mask(L=10, c=2, max_lag=2, device="cpu")
    assert m.shape == (9, 10)
    for r in range(9):
        cut = 2 + r + 2
        assert (~m[r, :cut + 1]).all() and m[r, cut + 1:].all()
    assert not row_pad_mask(10, 2, 100, "cpu").any()


def test_history_modes():
    torch.manual_seed(0)
    model = DecoderLM(20, 1, 16, 2, 2.0, 0.0, 10000.0).eval()
    ids = _ids(9)
    N, c, g = 5, 4, 2
    gen = torch.Generator().manual_seed(3)
    obs = build_history("observed", model, ids, c, g, N, REAL, gen, "cpu", "none")
    assert obs.shape == (N, c) and (obs == ids[:c]).all()
    samp = sample_history(model, ids, c, g, N, REAL, gen, "cpu", "none")
    assert samp.shape == (N, c) and (samp[:, :g] == ids[:g]).all()
    assert samp[:, g:].min() >= 4 and samp[:, g:].max() < 20             # sampled over real tokens only
    assert not (samp[:, g:] == ids[g:c]).all()                            # particles differ from the observed prefix
    assert (sample_history(model, ids, c, c, N, REAL, gen, "cpu", "none") == ids[:c]).all()   # g == c: nothing sampled
    with pytest.raises(ValueError):
        sample_history(model, ids, c, 0, N, REAL, gen, "cpu", "none")
    with pytest.raises(ValueError):
        build_staircase(ids, len(ids), obs, obs)
