"""The staircase construction (paper Def. 4.6 and Fig. 3; D-CB-2, D-CB-3, D-CB-9, D-CB-10).

For one sequence `ids[0..L)` with context `c` (BOS at position 0 counts
toward `c`, D-CB-5) the window is `[c, L)`, `Lc = L − c`. The staircase has
`Lc + 1` rows: row `r` keeps the observed tokens at window positions
`0..r−1` (sequence positions `c..c+r−1`) and replaces every later window
position by noise; row 0 is all noise (D-CB-2) and row `Lc` is fully
observed. One uniform noise vector per particle is drawn over the real event
tokens (never a special, never EOS; D-CB-3) and shared across rows (common
random numbers; D-CB-10), so adjacent rows differ at exactly one position.

The history `[0, c)` is either the observed prefix (`observed`, the paper's
teacher-forced figure) or, per particle, the real tokens `[0, g)` followed by
positions `[g, c)` sampled ancestrally from the frozen model over real event
tokens (`sampled`, the author library's reading; D-CB-9).

With a lag bound `M` (D-CB-12) row `r` is only ever read at effect columns
`<= c + r + M`, so its later positions are marked PAD and never attended.
"""
from __future__ import annotations

import torch

from .constants import HISTORIES, HISTORY_OBSERVED, HISTORY_SAMPLED, PAD


def draw_noise(real_ids, n_particles, length, generator, device):
    """`[N, L]` uniform draws over the real event tokens (positions < c are never used)."""
    real = torch.as_tensor(list(real_ids), dtype=torch.long, device=device)
    idx = torch.randint(len(real), (n_particles, length), generator=generator, device=device)
    return real[idx]


def build_staircase(ids, c, noise, history):
    """`ids [L]`, `noise [N, L]`, `history [N, c]` → `X [N, Lc+1, L]`."""
    L = ids.shape[0]
    N = noise.shape[0]
    if not 1 <= c < L:
        raise ValueError(f"context c must satisfy 1 <= c < L = {L}, got {c}")
    Lc = L - c
    t = torch.arange(L, device=ids.device)
    r = torch.arange(Lc + 1, device=ids.device)
    keep = (t[None, :] - c) < r[:, None]                 # [Lc+1, L]: observed window positions of each row
    X = torch.where(keep[None], ids[None, None, :].expand(N, Lc + 1, L), noise[:, None, :].expand(N, Lc + 1, L))
    X = X.clone()
    X[:, :, :c] = history[:, None, :]
    return X


def row_pad_mask(L, c, max_lag, device):
    """`[Lc+1, L]` bool: positions row `r` never needs (`t > c + r + max_lag`), marked PAD."""
    Lc = L - c
    t = torch.arange(L, device=device)
    r = torch.arange(Lc + 1, device=device)
    return t[None, :] > (c + r[:, None] + max_lag)


@torch.no_grad()
def sample_history(model, ids, c, g, n_particles, real_ids, generator, device, amp):
    """`[N, c]`: positions `[0, g)` observed, `[g, c)` drawn ancestrally from the model over real tokens."""
    if not 1 <= g <= c:
        raise ValueError(f"guidance g must satisfy 1 <= g <= c, got g={g}, c={c}")
    H = ids[:c].unsqueeze(0).repeat(n_particles, 1).clone()
    if g == c:
        return H
    real = torch.as_tensor(list(real_ids), dtype=torch.long, device=device)
    for t in range(g, c):
        prefix = H[:, :t]
        pad = torch.zeros_like(prefix, dtype=torch.bool)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(amp == "bf16")):
            logits = model.logits(prefix, pad)[:, -1, :]
        probs = torch.softmax(logits[:, real].float(), dim=-1)
        draw = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        H[:, t] = real[draw]
    return H


def build_history(mode, model, ids, c, g, n_particles, real_ids, generator, device, amp):
    if mode not in HISTORIES:
        raise ValueError(f"history must be one of {HISTORIES}, got {mode!r}")
    if mode == HISTORY_OBSERVED:
        return ids[:c].unsqueeze(0).repeat(n_particles, 1)
    return sample_history(model, ids, c, g, n_particles, real_ids, generator, device, amp)


__all__ = ["draw_noise", "build_staircase", "row_pad_mask", "sample_history", "build_history", "PAD", "HISTORY_SAMPLED"]
