"""Phase 2 read-out: forward the staircase and gather the observed token's
log-probability at every row and position (paper Eq. 2 and Fig. 3; D6).

`probe_sequence` returns `logp [N, Lc+1, L]` on the device, float32. For an
effect at window position `q` and a cause at window position `j < q`:

    p_l = exp(logp[l, j,     c+q])   the cause noised   (row j keeps window 0..j-1)
    q_l = exp(logp[l, j + 1, c+q])   the cause observed (row j+1 keeps window 0..j)

with the mediators `j+1..q−1` noised in both rows. The gather is always at
the observed token (the row's own token at the effect position plays no part:
the model is causal).

A memory estimate runs before the first forward and refuses above the
declared cap naming the estimate and the largest microbatch that fits (PRD
"resource exhaustion"): the peak is set by the microbatch `B`, not by `N`.
"""
from __future__ import annotations

import math

import torch

from .staircase import build_history, build_staircase, draw_noise, row_pad_mask

BYTES_PER_LOGP_CELL = 12    # float32 logp on device plus float64 statistics scratch


class MemoryRefusal(RuntimeError):
    pass


def estimate_peak_bytes(n_params, microbatch_rows, seq_len, vocab_size, d_model, n_heads, n_particles, n_window):
    """`6·P + B·L·(8V + 20d) + 2·B·H·L² + N·(Lc+1)·L·12` bytes."""
    B, L, V, d, H = microbatch_rows, seq_len, vocab_size, d_model, n_heads
    return int(6 * n_params + B * L * (8 * V + 20 * d) + 2 * B * H * L * L + n_particles * (n_window + 1) * L * BYTES_PER_LOGP_CELL)


def largest_microbatch(cap_bytes, n_params, seq_len, vocab_size, d_model, n_heads, n_particles, n_window):
    fixed = 6 * n_params + n_particles * (n_window + 1) * seq_len * BYTES_PER_LOGP_CELL
    per_row = seq_len * (8 * vocab_size + 20 * d_model) + 2 * n_heads * seq_len * seq_len
    return max(0, int((cap_bytes - fixed) // per_row))


def check_memory(model, microbatch_rows, max_seq_len, n_particles, max_window, cap_gb):
    cfg = model.config
    est = estimate_peak_bytes(model.n_params, microbatch_rows, max_seq_len, cfg["vocab_size"], cfg["d_model"],
                              cfg["n_heads"], n_particles, max_window)
    cap = int(cap_gb * 1e9)
    if est > cap:
        fits = largest_microbatch(cap, model.n_params, max_seq_len, cfg["vocab_size"], cfg["d_model"], cfg["n_heads"],
                                  n_particles, max_window)
        raise MemoryRefusal(
            f"estimated peak {est / 1e9:.2f} GB exceeds --memory-cap-gb {cap_gb} at --microbatch-rows {microbatch_rows} "
            f"(L={max_seq_len}, N={n_particles}, Lc={max_window}); the largest microbatch that fits is {fits}")
    return {"estimated_peak_bytes": est, "cap_bytes": cap, "microbatch_rows": microbatch_rows, "max_seq_len": max_seq_len}


@torch.no_grad()
def probe_sequence(model, ids, c, history_mode, guidance, n_particles, max_lag, real_ids, generator, microbatch_rows, device, amp):
    """`(logp [N, Lc+1, L] float32 on device, X [N, Lc+1, L])` for one encoded sequence `ids [L]`."""
    ids = torch.as_tensor(ids, dtype=torch.long, device=device)
    L = int(ids.shape[0])
    Lc = L - c
    noise = draw_noise(real_ids, n_particles, L, generator, device)
    history = build_history(history_mode, model, ids, c, guidance, n_particles, real_ids, generator, device, amp)
    X = build_staircase(ids, c, noise, history)                        # [N, Lc+1, L]
    pad = row_pad_mask(L, c, max_lag, device)[None].expand(n_particles, Lc + 1, L)
    flat = X.reshape(-1, L)
    pad_flat = pad.reshape(-1, L)
    targets = ids[None, :].expand(flat.shape[0], L)
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(amp == "bf16")):
        logp = model.log_probs_at(flat, pad_flat, microbatch_rows, targets=targets)
    return logp.reshape(n_particles, Lc + 1, L), X


def n_rows(L, c):
    return L - c + 1


def n_cells(L, c, max_lag):
    Lc = L - c
    return sum(min(q, max_lag) for q in range(1, Lc))


__all__ = ["probe_sequence", "check_memory", "estimate_peak_bytes", "largest_microbatch", "MemoryRefusal", "n_rows", "n_cells", "math"]
