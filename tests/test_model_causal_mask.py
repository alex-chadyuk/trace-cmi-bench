"""The backbone is causal: a token's log-probability depends only on the tokens
before it; the read-out `log_probs_at` equals a direct log_softmax gather and is
microbatch-invariant; the output projection is float32."""
import torch

from tracecmibench.constants import BOS, EOS
from tracecmibench.model import DecoderLM, attention_mask, estimate_params, lr_at


def _model(V=20):
    torch.manual_seed(0)
    return DecoderLM(V, n_layers=2, d_model=32, n_heads=4, ff_mult=2.0, dropout=0.0, rope_theta=10000.0).eval()


def test_future_tokens_do_not_change_earlier_log_probs():
    m = _model()
    ids = torch.tensor([[BOS, 5, 6, 7, 8, EOS]])
    pad = ids.eq(0)
    a = m.log_probs_at(ids, pad, microbatch=1)
    ids2 = ids.clone()
    ids2[0, 4] = 9                      # change position 4; positions <= 4's log-probs must not move
    b = m.log_probs_at(ids2, pad, microbatch=1)
    assert torch.allclose(a[0, :4], b[0, :4], atol=1e-6)
    assert not torch.allclose(a[0, 4:], b[0, 4:])       # the changed token itself is gathered differently
    assert a[0, 0] == 0.0


def test_log_probs_at_matches_direct_gather_and_microbatching():
    m = _model()
    ids = torch.tensor([[BOS, 5, 6, 7, EOS, 0, 0], [BOS, 8, 9, 10, 11, 12, EOS], [BOS, 4, EOS, 0, 0, 0, 0]])
    pad = ids.eq(0)
    lp = m.logits(ids, pad).log_softmax(-1)
    direct = lp[:, :-1].gather(-1, ids[:, 1:, None]).squeeze(-1)
    got = m.log_probs_at(ids, pad, microbatch=2)
    assert torch.allclose(got[:, 1:], direct, atol=1e-6)
    assert torch.allclose(got, m.log_probs_at(ids, pad, microbatch=3), atol=1e-6)
    assert m.logits(ids, pad).dtype == torch.float32
    assert (got[:, 1:][~pad[:, 1:]] <= 0).all()


def test_attention_mask_is_causal_and_hides_pad_keys():
    pad = torch.tensor([[False, False, True]])
    mk = attention_mask(pad)[0, 0]
    assert mk.tolist() == [[True, False, False], [True, True, False], [True, True, False]]
    assert mk.any(-1).all()             # every query row keeps a key


def test_param_estimate_and_schedule():
    m = _model(V=1000)
    assert m.n_params == estimate_params(1000, 2, 32, 2.0)
    six = DecoderLM(1000, 6, 256, 8, 2.0, 0.0, 10000.0)
    assert 4.0e6 < six.n_params < 4.4e6          # the |X| = 1000 anchor
    assert lr_at(0, 1.0, 10, 100, "cosine") == 0.1 and lr_at(9, 1.0, 10, 100, "cosine") == 1.0
    assert abs(lr_at(100, 1.0, 10, 100, "cosine")) < 1e-9 and lr_at(50, 1.0, 10, 100, "constant") == 1.0


def test_save_load_roundtrip(tmp_path):
    m = _model()
    ids = torch.tensor([[BOS, 5, 6, EOS]])
    sha = m.save(tmp_path / "m.pt", {"step": 3})
    m2, extra = DecoderLM.load(tmp_path / "m.pt")
    assert extra == {"step": 3} and len(sha) == 64 and m2.config == m.config
    assert torch.allclose(m.log_probs_at(ids, ids.eq(0), 1), m2.log_probs_at(ids, ids.eq(0), 1))
