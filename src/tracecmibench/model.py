"""The backbone: a decoder-only Transformer trained by next-token prediction (paper Phase 1).

Pre-norm RMSNorm, rotary position embedding, SwiGLU feed-forward, tied input
and output embedding, causal attention with PAD keys masked. Logits are always
float32 (the final projection runs outside autocast). `log_probs_at` gathers
`log_softmax` at the observed next token on the device per microbatch, so no
`[B, L, V]` tensor ever reaches the host — the read-out the probe needs
(Eq. 2: the Bernoulli parameter is the model's probability of the token that
occurred).

Written from the paper and standard references; no code from any predecessor.
"""
from __future__ import annotations

import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

RMS_EPS = 1e-6      # numerical constant of the norm, not a tunable
INIT_STD = 0.02


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        x32 = x.float()
        y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        return (y * self.weight.float()).to(x.dtype)


def rope_cache(L, head_dim, theta, device):
    half = head_dim // 2
    inv = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(L, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)                          # [L, half]
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """x: [B, H, L, D]; rotates pairs (x[..., :D/2], x[..., D/2:])."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    c = cos[None, None, :, :].to(x.dtype)
    s = sin[None, None, :, :].to(x.dtype)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


def attention_mask(pad_mask):
    """`[B, 1, L, L]` bool, True = may attend: causal, and PAD keys never attended.
    Position 0 is BOS in every sequence, so every query row keeps at least one key."""
    B, L = pad_mask.shape
    causal = torch.ones(L, L, dtype=torch.bool, device=pad_mask.device).tril()
    return causal[None, None] & ~pad_mask[:, None, None, :]


class Attention(nn.Module):
    def __init__(self, d, n_heads, dropout):
        super().__init__()
        if d % n_heads:
            raise ValueError(f"d_model {d} not divisible by n_heads {n_heads}")
        self.h, self.dh = n_heads, d // n_heads
        if self.dh % 2:
            raise ValueError("head dim must be even for rotary embeddings")
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.p = dropout
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask, cos, sin):
        B, L, d = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.p if self.training else 0.0)
        return self.drop(self.out(y.transpose(1, 2).reshape(B, L, d)))


class SwiGLU(nn.Module):
    def __init__(self, d, hidden, dropout):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, d, n_heads, hidden, dropout):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attention(d, n_heads, dropout)
        self.ffn = SwiGLU(d, hidden, dropout)

    def forward(self, x, mask, cos, sin):
        x = x + self.attn(self.n1(x), mask, cos, sin)
        return x + self.ffn(self.n2(x))


class DecoderLM(nn.Module):
    def __init__(self, vocab_size, n_layers, d_model, n_heads, ff_mult, dropout, rope_theta):
        super().__init__()
        self.config = {"vocab_size": int(vocab_size), "n_layers": int(n_layers), "d_model": int(d_model),
                       "n_heads": int(n_heads), "ff_mult": float(ff_mult), "dropout": float(dropout),
                       "rope_theta": float(rope_theta)}
        hidden = int(round(ff_mult * d_model))
        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, hidden, dropout) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.rope_theta = float(rope_theta)
        self.head_dim = d_model // n_heads
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=INIT_STD)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def hidden(self, ids, pad_mask):
        B, L = ids.shape
        cos, sin = rope_cache(L, self.head_dim, self.rope_theta, ids.device)
        mask = attention_mask(pad_mask)
        x = self.drop(self.embed(ids))
        for blk in self.blocks:
            x = blk(x, mask, cos, sin)
        return self.norm(x)

    def logits(self, ids, pad_mask):
        """float32 `[B, L, V]`: the output projection (tied to the embedding) runs outside autocast."""
        h = self.hidden(ids, pad_mask)
        with torch.autocast(device_type=h.device.type, enabled=False):
            return F.linear(h.float(), self.embed.weight.float())

    def loss(self, ids, pad_mask):
        """Mean next-token NLL in nats over non-PAD targets (positions 1..L-1)."""
        logits = self.logits(ids, pad_mask)[:, :-1]
        target = ids[:, 1:]
        keep = ~pad_mask[:, 1:]
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none")
        keep = keep.reshape(-1)
        return (nll * keep).sum() / keep.sum(), int(keep.sum())

    @torch.no_grad()
    def log_probs_at(self, ids, pad_mask, microbatch):
        """`[B, L]` float32: entry `t >= 1` is `log p(ids[t] | ids[<t])`; entry 0 is 0.
        Computed per microbatch of rows; the `[b, L, V]` tensor never leaves the device."""
        B, L = ids.shape
        out = torch.zeros(B, L, dtype=torch.float32, device=ids.device)
        for i in range(0, B, microbatch):
            lg = self.logits(ids[i:i + microbatch], pad_mask[i:i + microbatch])
            lp = lg.log_softmax(-1)
            out[i:i + microbatch, 1:] = lp[:, :-1].gather(-1, ids[i:i + microbatch, 1:, None]).squeeze(-1)
            del lg, lp
        return out

    # --- persistence ---------------------------------------------------------------------
    def save(self, path, extra=None):
        torch.save({"config": self.config, "state_dict": self.state_dict(), "extra": extra or {}}, path)
        return sha256_of(path)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        m = cls(**ckpt["config"])
        m.load_state_dict(ckpt["state_dict"])
        m.to(device).eval()
        return m, ckpt.get("extra", {})


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def estimate_params(vocab_size, n_layers, d_model, ff_mult):
    hidden = int(round(ff_mult * d_model))
    return vocab_size * d_model + n_layers * (4 * d_model * d_model + 3 * d_model * hidden + 2 * d_model) + d_model


def lr_at(step, lr, warmup_steps, total_steps, schedule):
    """Linear warmup, then cosine decay to zero at `total_steps` or a constant."""
    if warmup_steps > 0 and step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    if schedule == "constant":
        return lr
    if schedule == "cosine":
        span = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / span)
        return 0.5 * lr * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"unknown schedule {schedule!r}")
