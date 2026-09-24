# Reproduction gate, attempt 1 (2026-09-24) — FAILED on a defective truth, not on the engine

Report: [`gate/2026-09-24-gate-report.json`](../gate/2026-09-24-gate-report.json)
(`passed: false`, engine `39aae1a5…`, package commit `7444aee`). Job: one
g5.xlarge (A10G, 16.6 GB host RAM, 4 vCPU), 3,946 s wall, exit status 1,
output synced, results push disabled. Replica knobs equal the run record
(`scripts/tfvars_check.py … --args-diff`: OK).

## What the run measured

| quantity | value | registered expectation |
|---|---|---|
| Shannon redundancy R (scenario 1) | **0.5806** ≥ 0.58 | ≥ 0.58 — passed |
| backbone | 4,192,512 params, 12k × 256 steps, 611 s, 3.27·10⁵ tok-pos/s | — |
| val loss / exact H / n-gram H₂ | 2.950 / 2.898 / 2.643 nats | — |
| ε̂ (exact generator entropy) | **0.013**, in regime | note ≈ 0.129 (n-gram floor); paper 0.04–0.05 |
| ε̂ (order-2 n-gram floor) | 0.072, in regime | — |
| truth, val / test (as implemented) | **edge rate 0.674 / 0.677**, lag-1 share **0.257 / 0.256** | note: lag-1 share 0.846 at r = 1 |
| `faithful/library` τ\* (val) | 1.72·10⁻⁵ (F1 0.746) | note: 3·10⁻² |
| `faithful/library` F1 at τ = 3·10⁻⁵ (test) | **0.731** | **[0.40, 0.55]** — failed |
| `faithful/library` F1 at τ\* (test) | **0.747** | **≥ 0.86** — failed |
| `faithful/paper` τ\* / F1 at τ\* / at 3·10⁻⁵ | 3·10⁻⁶ / 0.679 / 0.596 | reported, ungated |
| per-lag recall at τ\*, library (lags 1–6) | 1.00 / 0.95 / 0.74 / 0.67 / 0.67 / 0.62 | note: lag ≥ 2 decades low |
| per-lag recall at τ\*, paper | 0.99 / 0.76 / 0.63 / 0.60 / 0.61 / 0.58 | — |
| probe throughput (both arms, B = 2048) | **1.26·10⁶ tok-pos/s**, 837 s per 1,024-sequence split | plan anchor 5·10⁵ (optimistic) |
| peak device / RSS | 1.66 GB / 2.31 GB (estimator: 1.95 GB) | cap 20 GB |
| published-clamp corruption, library statistic (scenario 15) | val 60,303 cells → 0 and 113 → float-max; test 60,337 / 79, of 1,812,480 cells per split (3.3 %) | reported, never gated |
| published-clamp corruption, paper statistic | val 38,540 / 2,001; test 37,504 / 1,972 | reported |

## Diagnosis

The truth the gate scored against was not the paper's. D-CB-13 as coded
computed the **full categorical KL** `KL(P(X_j | x_<j) ‖ P(X_j | x_<j, x_i := u))`
over the 1,000-way next-token distribution, averaged over the ten draws.
Appendix E.1 defines the truth on the **event process** — "the average KL
divergence … between post-intervention and observational distributions of
`E_t`", with `E_t = 1{X_t = x_t}` the binary indicator of §3.4 — i.e. the
**Bernoulli** KL at the observed token, and both the author's library and the
lab's earlier reimplementation compute it that way (gap report §5.2, `KL_B`).
The categorical KL upper-bounds the Bernoulli one, so nearly every pair
within the lag window crossed δ = 0.05: 68 % of candidate pairs were "edges",
with the lag composition nearly flat. Against such a truth the printed
threshold no longer fails (most predicted pairs are edges, so precision is
high) and the blind-selected threshold cannot reach 0.86 (categorical effects
invisible at the observed token are counted as misses).

**Verified locally on the same world and the same 256 test sequences**
(the world is a deterministic function of the seed; `x` is stored in
`truth/test-truth.npz`; recomputed with fresh counterfactual draws):

| truth statistic | edge rate | edges / sequence | lag share 1 … 6 |
|---|---|---|---|
| categorical KL, mean over draws (as run) | 0.678 | 246 | 0.256 · 0.234 · 0.166 · 0.120 · 0.114 · 0.110 |
| Bernoulli KL, mean of KLs over draws | 0.175 | 64 | 0.743 · 0.155 · 0.039 · 0.022 · 0.019 · 0.021 |
| **Bernoulli KL of the mixture** `KL_B(p ‖ mean_u q_u)` | **0.150** | 55 | **0.848** · 0.084 · 0.026 · 0.015 · 0.013 · 0.014 |

The mixture form reproduces the replication note's registered lag-1 truth
share (0.846) to two decimals and is Eq. 9's placement of the expectation
(inside the do-operator). It is adopted as the amended D-CB-13, dated
2026-09-24, before the second attempt.

## What the run also established (stands regardless of the truth)

- The engine runs end to end on the paper's world at the paper's size, in
  regime (ε̂ 0.013 exact), within an hour on one A10G; the throughput anchor
  is 1.26·10⁶ token-positions/s at B = 2048, 2.5× the plan's optimistic
  figure, so every later replica is re-sized from it.
- The published float32 clamp would have corrupted 3.3 % of the library
  statistic's cells on the paper's own world (to zero: a silent non-edge),
  which is the scenario-15 count the PRD asked for and a finding in itself.
- Redundancy lands 0.0006 above the band at `--w-scale 3.2`; a second world
  seed could fall below. The second attempt keeps 3.2 (same seed).
- The paper-vs-library difference is already large on this truth (−0.07 F1
  at τ\*, −0.14 at the printed τ); its value on the corrected truth is the
  number that matters.
- Two things the report lacked are added for the second attempt: precision
  and recall beside every F1, and the per-sequence statistic matrices
  (`matrices-{val,test}.npz`, S3-only), so a future truth correction can be
  re-scored without re-probing.

## Disposition

The failed report is committed as the record. The truth is corrected
(D-CB-13 amended), the engine modules are untouched (engine hash unchanged:
`39aae1a5…`), and a second replica
(`trace-cmi-bench-26-09-24-gate-r2`) is authored for the owner. Nothing
about the bands, the threshold grid, `c`, `g`, `N` or the world's knobs
changes: a failure at the second attempt would be a finding about the
engine, which this one was not.
