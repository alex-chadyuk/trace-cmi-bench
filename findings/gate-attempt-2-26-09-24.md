# Reproduction gate, attempt 2 (2026-09-24) — FAILED on scenario 2 only; scenario 3 and the lag-graded limit replicate the note

Report: [`gate/2026-09-24-r2-gate-report.json`](../gate/2026-09-24-r2-gate-report.json)
(`passed: false`, engine `39aae1a5…`, package commit `1f2d77d` = attempt 1's
fix branch; the file carries the run name because `selfcheck` names reports
by date alone and attempt 1 holds this date's name — see Disposition). Job:
one g5.xlarge (A10G, 16.6 GB host RAM, 4 vCPU), 3,915 s wall, exit status 1,
output synced, results push disabled. Replica knobs equal the run record
(`scripts/tfvars_check.py … --args-diff`: OK). The only change since
attempt 1 is the truth (D-CB-13 amended): same knobs, seed, bands, grid,
`c`, `g`, `N`; engine hash unchanged.

**The run is attempt 1 re-scored.** The trained model is bit-identical to
attempt 1's (`model_sha256 4aa4ac28…`), and the published-clamp corruption
counts — which depend only on the model and the probe — are identical to the
cell (val 60,303 → 0 / 113 → float-max; test 60,337 / 79). Generation,
training and probing are therefore deterministic across two instances, and
attempt 2 measures exactly what attempt 1's statistic matrices say against
the corrected truth.

## What the run measured

| quantity | value | registered expectation |
|---|---|---|
| Shannon redundancy R (scenario 1) | **0.5806** ≥ 0.58 | passed (same world as attempt 1) |
| truth, val / test (Bernoulli KL of the mixture, δ = 0.05) | edge rate **0.151 / 0.150**, lag-1 share **0.844 / 0.848** | note: lag-1 share 0.846 |
| ε̂ exact / n-gram | 0.013 / 0.072, in regime | note ≈ 0.129 (n-gram floor) |
| `faithful/library` τ\* (val, blind) | **3·10⁻²** (val F1 0.9017) | note: **3·10⁻²** on both seeds |
| `faithful/library` F1 at τ\* (test, scenario 3) | **0.9019** — P 0.958 / R 0.853 | **≥ 0.86** — passed; note 0.9008 / 0.9051, P/R 0.960 / 0.850 |
| `faithful/library` F1 at τ = 3·10⁻⁵ (test, scenario 2) | **0.3515** — P 0.215 / R 0.977 | **[0.40, 0.55]** — **failed** (below the floor by 0.05); note 0.4553 / 0.4647 |
| per-lag recall at τ\*, library (lags 1–6) | **0.992 / 0.038 / 0 / 0 / 0 / 0** | note: lag-1 0.97–0.99, lag ≥ 2 decades low — replicated; recall 0.853 ≈ lag-1 truth share 0.848 |
| per-lag recall at 3·10⁻⁵, library | 1.00 / 0.96 / 0.77 / 0.69 / 0.68 / 0.61 | — |
| `faithful/paper` τ\* / F1 at τ\* / at 3·10⁻⁵ | 10⁻² / **0.897** / **0.529** | reported, ungated; 0.529 sits inside the scenario-2 band |
| paper − library, F1 | **+0.177** at 3·10⁻⁵, −0.005 at each arm's τ\* | the gap the PRD asked to be measured |
| probe throughput / wall | 1.26·10⁶ tok-pos/s, 835 s per split | anchor confirmed |

## Reading, before anything is touched

The plan's instruction on a second failure is to read the per-lag recall,
the precision–recall curves and the paper-vs-library gap first. Read on the
test statistic matrices (`matrices-test.npz`, 1,812,480 cells = 1,024
sequences × 1,770 position pairs; position level, truth defined for the
340,992 cells at lag ≤ 6 whose effect is a real token):

1. **Scenario 3 is a replication of the note to two decimals** — the
   selected threshold, the F1, the precision and the recall all land on the
   note's arm-C numbers, and the recall equals the lag-1 truth share because
   the per-lag recall at τ\* is the note's lag-graded limit. The estimator
   the engine implements is the estimator the note characterised.
2. **Scenario 2 fails in the registered direction but overshoots.** The
   printed threshold does not replicate (0.35 vs the paper's 0.91); it falls
   0.05 below the band's lower edge. Recall matches the note's (0.977 vs
   ≈ 0.96); the whole shortfall is precision, 0.215 against the note's
   ≈ 0.30 at N = 128 (its own N sweep reads 0.446 at N = 2 → 0.271 at
   N = 512).
3. **The false positives at 3·10⁻⁵ sit inside the truth's own lag window,
   on the floor's median.** Of the 183,665 position-level false positives,
   178,041 (97 %) lie at lags 1–6 and 5,624 beyond lag 6 (which holds 81 %
   of the cells). The non-truth statistic by lag:

   | lag | non-truth cells | non-truth median | share > 3·10⁻⁵ | truth cells | truth median | truth share > 3·10⁻⁵ |
   |---|---|---|---|---|---|---|
   | 1 | 16,861 | 2.3·10⁻³ | 0.94 | 43,555 | 1.7 | 1.00 |
   | 2 | 55,046 | 1.2·10⁻⁴ | 0.86 | 4,346 | 5.9·10⁻⁴ | 0.96 |
   | 3 | 56,972 | 4.1·10⁻⁵ | 0.59 | 1,396 | 9.2·10⁻⁵ | 0.77 |
   | 4 | 56,661 | 3.1·10⁻⁵ | 0.51 | 683 | 6.8·10⁻⁵ | 0.69 |
   | 5 | 55,586 | 3.0·10⁻⁵ | 0.50 | 734 | 5.7·10⁻⁵ | 0.68 |
   | 6 | 54,647 | 2.5·10⁻⁵ | 0.45 | 649 | 4.8·10⁻⁵ | 0.61 |
   | 7–12 | 310,272 | 1.6·10⁻⁶ | 0.02 | — | — | — |
   | 13–59 | 1,155,072 | 5·10⁻⁷ | 0.00 | — | — | — |

   At lags 3–6 the non-truth median is 2.5–4.1·10⁻⁵: the printed threshold
   sits on the floor, and half of every non-truth cell there crosses it. The
   lag-1 non-truth cells are the near-δ misses (real effects just under the
   truth cut) and cross at any small τ in any implementation. Pooled over
   lags 2–6 the non-truth median is 4.6·10⁻⁵ against the note's tested
   non-truth ≈ 2·10⁻⁵ at N = 128 (4.1·10⁻⁶ at N = 2 → 2.7·10⁻⁵ at
   N = 512): **this engine's floor is about twice the note's**. Beyond
   lag 6 the floor (median 6·10⁻⁷) matches the note's other-non-truth
   1.1·10⁻⁶. The truth median is 1.2 nats, 4.6 decades above the floor, so
   the observable at τ\* is untouched (1,964 false positives at 0.03).
4. **The paper arm's floor is 6–7× lower** (lags 3–6 median 4–6·10⁻⁶),
   hence 0.529 at the printed threshold and 0.897 at its own τ\* = 10⁻²
   (precision 0.99 at 0.03 costs it recall). The mean-of-particle-KLs
   statistic the library computes is bounded below by the KL of the mean
   particle (Jensen), and the difference on non-truth pairs is Monte-Carlo
   noise; the two arms' gap at the printed threshold is that noise.

**So the scenario-2 observable is a measurement of the library statistic's
non-truth floor at lags 3–6, not of recovered structure.** The note's own
diagnosis says the same: the particle count alone moves this F1 by 0.17,
and the as-printed F1 ranges 0.25–0.48 across vocabulary sizes, while its
cross-seed width within one implementation is 0.01. A factor of two in the
floor between two implementations moves the value out of a band that wide.

### What does not explain the gap

- **Particles:** the note's 0.46 was read at N = 128, the gate's N.
- **Context and layout:** both use `c = 6` BOS-counted on L = 66 (BOS + 64
  + EOS), Lc = 60, 1,770 cells per sequence, every lag probed.
- **The truth:** edge rate 0.150 and lag-1 share 0.848 against the note's
  0.846; the note's position-level density (≈ 59 truth pairs per trace,
  from its particle-diagnosis cell counts) matches this run's 50–55.

### Candidate causes (unresolved; none testable from the records)

- **(a) The note's statistic carried the published float32 clamp** (gap
  report G5: same ε in float32 plus `nan_to_num`), which zeroes any cell
  with one saturated particle. On this model that is 60,337 test cells
  (3.3 %); had they been zeroed here and had they concentrated at lags ≤ 6
  above the threshold, they would account for up to a third of the false
  positives — an upper bound, not a measurement. `selfcheck` scores the
  safe family only and the matrices store only it; testing (a) needs a
  `--divergence` knob on `selfcheck` (not an engine module; the hash would
  not move) and a re-probe (≈ 14 min per split).
- **(b) The model:** ε̂ against the n-gram floor is 0.072 here against the
  note's 0.129 — a differently calibrated model on a clean-room world; the
  floor is estimator noise and moves with calibration.
- **(c) The history sampler** (D-CB-9: ancestral over real tokens, `g = 3`)
  against the author library's proposal.
- **(d) The world instance:** a clean-room generator at the same knobs and
  redundancy; the paper's own world (undisclosed weight scale) is a third.

### On the band itself

PRD scenario 2's stated rationale is one-sided: the gate "fails if the
printed threshold unexpectedly does replicate, because that would mean this
implementation differs from the one the note characterised". The lower edge
(0.40) has no stated rationale. The open question is whether the band
encodes *the printed threshold does not replicate* — which this run
satisfies — or *this implementation's noise floor equals trace-cmi's*, which
it does not, and which the note's own diagnosis shows is a nuisance-sensitive
quantity. That is an owner decision; nothing here changes the band, the
grid, `c`, `g`, `N` or the engine.

## Disposition

The failed report is committed as the record beside attempt 1's. Report
naming: `selfcheck` writes `gate/<date>-gate-report.json`, so two attempts
on one day collide; attempt 2's file carries the run name and the naming
should take the run name in M6 (`selfcheck` is outside the engine hash).
`discover`/`sweep` remain refused on every benchmark corpus until a report
passes at engine `39aae1a5…`. Options, none taken: (A) a diagnostic
re-score under the published clamp to test (a); (B) a dated PRD
Decision-Log amendment making scenario 2 one-sided, recorded before a third
attempt; (C) keep the two-sided band and treat the engine as not reproduced.
