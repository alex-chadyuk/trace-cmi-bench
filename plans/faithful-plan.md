# Faithful-arm plan (pre-registered, 2026-09-24)

Committed before the code that implements it and before the first benchmark
replica. Governs `faithful/paper` and `faithful/library`, each with the
sub-arms `max` and `mean` (D-CB-8), on every trace-bench corpus the ladder
reaches (`xs → s → m → l`, `xl` on the release's entry criteria). The gate
that licenses this plan: `gate/2026-09-24-r3-gate-report.json`, engine
`39aae1a5…`. Nothing below is changed after a corpus's validation sweep has
run on it; a change is a dated addendum at the end of this file, and a test
read never uses a value the addendum introduced after its freeze.

## 1. Arms

| arm | history of the context positions | statistic thresholded | what it is |
|---|---|---|---|
| `faithful/paper` | observed (teacher-forced) | `mean_kl_bf` = KL(p̄ ‖ q̄), mean over particles then KL, base first | the paper's operational figure (Fig. 3, App. C.2, Eq. 9/10) |
| `faithful/library` | sampled ancestrally per particle for positions `[g, c)` | `kl_mean_df` = mean over particles of KL(q_ℓ ‖ p_ℓ), cause-observed first | the author library's core path; the reading the replication note characterised and the gate binds |

Every probing pass emits all four statistics (D-CB-15) so the
paper-vs-library difference decomposes into history source, statistic and
KL order after the fact; only the arm's own statistic is ever thresholded.
Sub-arm `max`: corpus-level score of a type pair = the maximum over its
position-pair occurrences; `mean`: the mean over occurrences. Edge iff
score > τ, strictly.

## 2. Model and sequences

One model per corpus, trained by `pretrain` on the `end-session` view
(D-CB-17) with the pretraining knobs recorded in the replica; both grains are
probed from that frozen model: request sequences from `end-request`, session
sequences from `end-session`, each under the replica's `--max-len`. The
oracle score ε̂ against the order-2 n-gram floor is reported per corpus and
never gated (`in_regime` = ε̂ < 0.1 flagged in `annotate`).

Sequences probed: `--num-sequences` with `--sequence-sample head` (file
order) on both splits, sized per rung in the replica (D-CB-14); `0` = the
whole split. The probe's precision mode is recorded (`--probe-amp`, D-CB-20;
the gate showed it moves cells, not the observable).

## 3. Grids (validation sweep, blind)

- context `c` (BOS-counted): request grid `{1, 2, 3}`, session grid `{2, 4, 8}` (D-CB-4).
- particles `N`: `{2, 8, 32, 128}` (D-CB-19).
- guidance `g`: the replica passes `--guidance 3`; the effective value per
  cell is `g = min(3, c)` so that `1 ≤ g ≤ c` holds (D-CB-9); `g` is recorded
  in every score file and in the freeze.
- threshold τ (score side, never in the sweep): half-decades
  `{1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}` plus the
  anchor `1.72e-5` (C/|X| at |X| = 1000). The grid is a `scoresweep` knob and
  the freeze records it.
- `--max-lag`: the full window (`max-len − c`), as the gate; per-lag files
  are written for lags `1..--per-lag-files`.
- floor: the target's default floor (0.05) for every selection; the sweep
  reports the scorer's floor sensitivity unchanged in the test read.

Cells per corpus: 2 arms × 3 c × 4 N × 2 grains probed; × 2 aggregations
× 12 τ scored = 1,152 scored rows in `val-table.json`.

## 4. Freeze rule

Per `(corpus, arm, sub-arm, grain)`: the full-grid argmax of the scorer's
**directed F1 at the default floor** over `(c, N, τ)`; ties → smaller `N`,
then smaller `c`, then larger τ. `freeze` writes
`freezes/<date>-<rung>-<variant>-s<k>.json` carrying `corpus_id`,
`tool_version`, `config_hash`, `model_sha256`, the package commit, the
val-table's sha256 and the chosen `{tau, c, N, g}` per cell; it refuses to
overwrite. One freeze commit per rung, merged by the owner before any test
read. The test read (`discover --split test --freeze …`) asserts that the
commit which added the freeze file is an ancestor of HEAD, that the run
started after that commit, and that `(τ, c, N, g, model_sha256, corpus_id)`
in the freeze equal the CLI values and the loaded model (PRD scenarios 22,
23). One test read per cell; a second read is a replay into a new folder
with its reason recorded.

## 5. Coverage rule

`scoresweep` and `annotate` report, per `(arm, grain)`, the fraction of the
scorer's universe pairs that co-occur in the probed sample and the fraction
of truth-directed edges at the floor that do (the recall of the full
ranking, since the scorer counts every listed edge as present). If the val
**reachable-recall ceiling** (truth edges co-occurring in the sample /
truth edges) is below **0.90** on either grain, `--num-sequences` is raised
in a new dated val replica before any freeze; the raise and the ceiling are
recorded in RUN.md. Tokens the views never mint (`unreachable_tokens`) cap
recall by construction and are reported per corpus (D-CB-16), never
subtracted.

## 6. Hypotheses (scored in `findings/` after each rung's test read; every clause pass-or-fail, PRD scenario 18)

- **H-agree (paper vs library at the operating point).** At their own
  frozen `(τ, c, N)`, the two faithful arms' directed F1 at the default
  floor differ by **≤ 0.05** on every corpus and grain (five seeds, the
  paired per-seed difference). *Basis:* the gate read −0.005 between them at
  τ\*. *Fail:* any (rung, variant, grain) where the five-seed mean paired
  difference exceeds 0.05 in magnitude.
- **H-floor (paper vs library below the operating point).** Over the τ
  grid on validation, `faithful/library` reads a lower directed precision
  than `faithful/paper` at every τ ≤ 1e-4 on every corpus (the
  mean-of-particle-KLs floor sits above the KL-of-means floor). *Fail:* any
  corpus where library precision ≥ paper precision at some τ ≤ 1e-4 on both
  grains.
- **H-lag (the lag-graded recall limit transfers).** At the frozen τ,
  per-lag recall of the truth edges (from the per-lag prediction files) is
  non-increasing in lag on every corpus, with lag-1 recall ≥ 0.8 of the
  reachable lag-1 ceiling and recall at lags ≥ 3 below 0.2 of their ceiling.
  *Fail:* a corpus where lag-1 recall < 0.8 × ceiling, or where recall at
  lag ≥ 3 exceeds 0.2 × ceiling on both grains.
- **H-sat (particle saturation).** The frozen `N` is ≤ 8 on at least 80 %
  of the `(arm, sub-arm, grain)` cells across a rung, and the val directed F1
  at the frozen `(c, τ)` moves by < 0.01 between the frozen `N` and 128.
  *Fail:* either clause.
- **H-agg (aggregation).** `max` reaches a directed F1 ≥ that of `mean` −
  0.02 on every cell (max-pooling over occurrences is the paper's
  population reading). *Fail:* any cell where `mean` beats `max` by more
  than 0.02 on all five seeds.
- **H-zero (structural limitation, scenario 8).** On the latent variant,
  both faithful arms' bidirected recall is exactly 0 and every annotate
  record carries `structural_limitation.truth_bidirected_edges` > 0 where
  the target has bidirected edges at the floor. Not a performance claim: a
  check that the zero is recorded as structural, never as an achievement.
- **H-validity (scenario 11).** On the twin variant at the request grain
  the causal-validity axis carries numeric SID and AID values on every seed;
  on the latent variant and wherever a graph is cyclic it carries a reason
  string. *Fail:* a numeric value where a reason is expected or vice versa.
- **H-regime.** ε̂ < 0.1 on every corpus through `l`. Reported; a corpus
  out of regime is flagged in its tables, never dropped.

## 7. What is not a hypothesis

The absolute level of any metric. This plan characterises TRACE on the
benchmark; it makes no flagship or Pareto claim and does not trigger the
charter's benchmark commit.

## Addenda

(none)
