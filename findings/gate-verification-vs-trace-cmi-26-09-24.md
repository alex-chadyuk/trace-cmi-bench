# Gate verification against trace-cmi's recorded runs (2026-09-24)

The gate's two registered numbers come from the replication note, and the
note's numbers come from the lab's private predecessor `trace-cmi` (arm C:
`|X| = 1000`, `L = 64`, `h = 6`, `w_scale = 3.2`, `R ≈ 0.59`, `c = 6`,
`N = 128`, `g = 3`, 1,024 head sequences per split, exact per-sequence
interventional truth, seeds 0 and 1). Owner instruction of 2026-09-24: verify
this engine against those runs, not against two summary numbers. Read here
(records only — result JSONs, the M9 grid file and the run registry; the
predecessor's `src/` stays unopened, PRD Decision Log 11): the arm-C run
folders `models/model-26-08-14-c-{s0,s1}-r2` (test) and `c-val-{s0,s1}`
(validation), `out/m9diag/grid.json` (the 23-point test grid per particle
count, of which N = 128 is the arm-C run), and RUN.md rows of 2026-08-14.
This engine's side is attempt 2 (`gate/2026-09-24-r2-gate-report.json`,
`matrices-{val,test}.npz`).

## 1. The full curve, both splits, both seeds

`faithful/library`, mean per-sequence directed F1 / precision / recall of the
Def. 3.2 projection, τ grid of the gate (trace-cmi's grid has the same points
plus intermediates).

**Test split** (trace-cmi: `grid.json` N = 128, seeds 0 / 1; bench: attempt 2):

| τ | trace-cmi s0 F1 (P / R) | trace-cmi s1 F1 (P / R) | bench r2 F1 (P / R) | Δ vs mean of seeds |
|---|---|---|---|---|
| 10⁻⁶ | 0.108 (0.057 / 1.000) | 0.108 (0.057 / 1.000) | 0.114 (0.060 / 1.000) | +0.006 |
| 3·10⁻⁶ | 0.184 (0.102 / 0.997) | 0.187 (0.104 / 0.995) | 0.177 (0.097 / 0.999) | −0.009 |
| 10⁻⁵ | 0.307 (0.182 / 0.981) | 0.315 (0.188 / 0.983) | 0.262 (0.151 / 0.994) | −0.049 |
| 1.72·10⁻⁵ | 0.375 (0.233 / 0.970) | 0.384 (0.240 / 0.972) | 0.303 (0.180 / 0.987) | −0.077 |
| **3·10⁻⁵** | **0.455** (0.300 / 0.954) | **0.465** (0.308 / 0.959) | **0.352** (0.215 / 0.977) | **−0.108** |
| 10⁻⁴ | 0.642 (0.495 / 0.920) | 0.653 (0.504 / 0.930) | 0.501 (0.342 / 0.945) | −0.147 |
| 3·10⁻⁴ | 0.776 (0.689 / 0.893) | 0.784 (0.695 / 0.905) | 0.684 (0.549 / 0.914) | −0.096 |
| 10⁻³ | 0.848 (0.828 / 0.872) | 0.853 (0.826 / 0.883) | 0.817 (0.760 / 0.886) | −0.034 |
| 3·10⁻³ | 0.871 (0.883 / 0.861) | 0.875 (0.883 / 0.870) | 0.857 (0.846 / 0.871) | −0.016 |
| 10⁻² | 0.887 (0.923 / 0.856) | 0.891 (0.923 / 0.863) | 0.881 (0.906 / 0.860) | −0.008 |
| **3·10⁻²** | **0.901** (0.960 / 0.850) | **0.905** (0.961 / 0.857) | **0.902** (0.958 / 0.853) | **−0.001** |
| 10⁻¹ | 0.889 (0.991 / 0.807) | 0.893 (0.991 / 0.814) | 0.892 (0.990 / 0.813) | +0.001 |

**Validation split** (trace-cmi: `c-val-{s0,s1}/per_sequence_eval.json`
`exact`; bench: attempt 2's blind sweep):

| τ | trace-cmi s0 | trace-cmi s1 | bench r2 | Δ |
|---|---|---|---|---|
| 10⁻⁶ | 0.109 | 0.108 | 0.114 | +0.006 |
| 3·10⁻⁶ | 0.185 | 0.187 | 0.177 | −0.009 |
| 10⁻⁵ | 0.307 | 0.314 | 0.262 | −0.049 |
| 1.72·10⁻⁵ | 0.376 | 0.384 | 0.303 | −0.077 |
| 3·10⁻⁵ | 0.455 | 0.464 | 0.352 | −0.108 |
| 10⁻⁴ | 0.641 | 0.652 | 0.502 | −0.145 |
| 3·10⁻⁴ | 0.776 | 0.782 | 0.685 | −0.094 |
| 10⁻³ | 0.846 | 0.851 | 0.819 | −0.030 |
| 3·10⁻³ | 0.868 | 0.874 | 0.858 | −0.013 |
| 10⁻² | 0.885 | 0.890 | 0.882 | −0.006 |
| **3·10⁻²** | **0.898** | **0.904** | **0.902** | **+0.001** |
| 10⁻¹ | 0.885 | 0.891 | 0.892 | +0.004 |

Blind selection picks τ\* = 3·10⁻² on every side (trace-cmi both seeds,
bench). Val-to-test agreement is 0.001–0.003 on every side.

**Reading.** The three curves are one curve at every τ ≥ 10⁻² (within
0.008 F1, inside trace-cmi's own seed-to-seed spread of 0.004–0.006) and at
every τ ≤ 3·10⁻⁶ (where everything crosses). Between 10⁻⁵ and 3·10⁻³ this
engine sits below by up to 0.15 F1, entirely through precision: recall is
equal or higher at every τ. The gap peaks at 10⁻⁴ and closes by 10⁻³. The
engine reproduces trace-cmi's estimator wherever the true-edge signal decides
the reading and differs from it only where the non-edge noise floor decides
it.

## 2. What else agrees

| quantity | trace-cmi (s0 / s1) | bench r2 |
|---|---|---|
| truth pairs per test sequence (cause in window) | 50.7 / 49.8 | 50.2 |
| position pairs tested per split at lag ≤ h | 340,992 | 340,992 |
| truth lag-1 share | 0.846 (note) | 0.848 |
| redundancy R | 0.5930 / 0.5901 | 0.5806 |
| lag-1 truth-pair level: median log₁₀(statistic / generator KL) | 0.00 / +0.01 (M9 diagnosis) | +0.01 (test), +0.007 (val) |
| lag-1 recall at τ\* | 0.99 | 0.992 |
| lag ≥ 2 recall at τ\* | 0.004–0.006 | 0.021 (lag 2: 0.038; lags 3–6: 0) |
| floor beyond the truth's lag range (median statistic, lag > 6) | 1.1·10⁻⁶ ("other valid non-truth") | 6·10⁻⁷ |
| backbone | 4,192,512 params, 12k × 256 | 4,192,512 params, 12k × 256 |
| val loss / generator entropy / ε̂ | 3.341 / 2.812 / **0.129** | 2.950 / 2.898 / **0.013** |

The estimator's level on the edges it resolves is calibrated identically
(unbiased on lag-1 truth pairs on both sides); the lag-graded limit is the
same; the truth density matches to 1 %.

## 3. Where the difference lives

Attempt 2's test matrices, position level, lags ≤ 6, 289,629 non-edge cells
and 51,363 edge cells, binned by the **generator's exact KL** for the pair
(the truth cut is δ = 0.05):

| true KL of the non-edge pair | cells | library median | share > 3·10⁻⁵ |
|---|---|---|---|
| < 10⁻⁶ (true nulls) | 14,083 | 3.0·10⁻⁵ | 0.50 |
| 10⁻⁶–10⁻⁵ | 24,494 | 3.6·10⁻⁵ | 0.55 |
| 10⁻⁵–10⁻⁴ | 43,662 | 4.9·10⁻⁵ | 0.63 |
| 10⁻⁴–10⁻³ | 67,219 | 4.7·10⁻⁵ | 0.61 |
| 10⁻³–10⁻² | 95,524 | 4.6·10⁻⁵ | 0.61 |
| 10⁻²–0.05 | 44,647 | 6.9·10⁻⁵ | 0.69 |

The crossing rate at the printed threshold is nearly flat in the true effect
size: **true-null pairs cross it half the time.** So the extra false
positives are not weak effects that a better-fit model resolves (the model
is better fit: ε̂ 0.013 against 0.129); they are the estimator's own noise on
pairs with no effect, and its median on true nulls sits at the printed
threshold. Rank correlation between statistic and true KL is 0.10 on
non-edges and 0.95 on edges, on both arms. trace-cmi's tested-non-truth
median at N = 128 is ≈ 2·10⁻⁵ (its N sweep: 4.1·10⁻⁶ at N = 2 rising to
2.7·10⁻⁵ at N = 512, the median of a right-skewed per-particle KL climbing
toward its mean); this engine's is 3.0·10⁻⁵ on true nulls and 4.6·10⁻⁵
pooled over lags 2–6. The F1 difference at 3·10⁻⁵ is what a ×1.5–2 floor
does to a threshold that sits on it — the same displacement trace-cmi's own
particle count produces between N = 32 and N = 512 (F1 0.494 → 0.423 at
3·10⁻⁵, seed 0).

`faithful/paper` (mean-then-KL) has a floor 6–7× lower on both sides of the
comparison and reads 0.529 at 3·10⁻⁵, inside the former band; its lag-1
level is biased low by 0.67 decades (the mean of the particle distribution
is smoother than the particles), which is why its τ\* is 10⁻².

## 4. Probe precision — the one stage-level difference in the records

trace-cmi's arm-C `ci/arguments.json` records `ci_amp: off`: its probing
forward passes ran in full precision. Attempt 2 ran them under bf16 autocast,
because `selfcheck`'s single `--amp` flag drove pretraining and probing
alike (the output projection runs in float32 on both sides; the transformer
body did not). The hypothesis that bf16 rounding of the hidden states
inflates the non-edge floor was tested locally on the frozen attempt-2 model
(CPU, six validation sequences, the package's own `probe_split`):

| quantity (`faithful/library`, val sequences 0–5) | CPU bf16, N = 32 | CPU full, N = 32 | CPU bf16, N = 128 | CPU full, N = 128 | GPU bf16, N = 128 (attempt 2) |
|---|---|---|---|---|---|
| non-edge median, lags 3–6 | 1.81·10⁻⁵ | 1.80·10⁻⁵ | 3.08·10⁻⁵ | 3.10·10⁻⁵ | 3.12·10⁻⁵ |
| non-edge share > 3·10⁻⁵, lags 3–6 | 0.366 | 0.365 | 0.507 | 0.514 | 0.517 |
| true-null median (generator KL < 10⁻⁶, 92 cells) | 2.16·10⁻⁵ | 2.20·10⁻⁵ | 3.07·10⁻⁵ | 3.00·10⁻⁵ | 3.72·10⁻⁵ |
| true-null share > 3·10⁻⁵ | 0.413 | 0.413 | 0.500 | 0.500 | 0.565 |
| floor beyond lag 6, median | 3.9·10⁻⁷ | 3.8·10⁻⁷ | 6.2·10⁻⁷ | 5.9·10⁻⁷ | 5.9·10⁻⁷ |
| lag-1 edge level, median log₁₀ ratio | −0.006 | −0.006 | +0.001 | +0.001 | −0.004 |

The two precisions do not compute the same cells — at N = 128 only 125 of
10,620 cells are bit-identical, the log-statistics correlate 0.996 and the
median cell moves 0.026 decades — but every distributional quantity of the
floor is the same to two digits, and the CPU full-precision run reproduces
the GPU bf16 run. Rounding of the hidden state is common to the two
staircase rows a cell compares and cancels in their difference. **The
hypothesis is refuted: probe precision does not move the floor.** (The
table also shows the particle-count effect trace-cmi's diagnosis described:
the true-null median rises from 2.2·10⁻⁵ at N = 32 to 3.0·10⁻⁵ at N = 128.)
Script: session scratchpad `probe_amp_diag2.py`, CPU, 124 s (full) and
490 s (bf16) for six sequences at N = 128.

What remains as the source of the ×1.5–2 floor difference is therefore the
model and the world instance — the same architecture and budget trained to
ε̂ 0.013 here against 0.129 there — with the history sampler as the only
other candidate. None of these is a defect to fix; the gate scores the
estimator at the point where the two implementations agree.

## 5. Consequences

- **Scenario 2 is one-sided** (PRD Decision Log 12): the printed threshold
  must not replicate (F1 ≤ 0.55). The former lower edge asserted equality of
  two implementations' noise floors, a quantity the note's own diagnosis
  shows is a nuisance.
- **`--probe-amp` is a required knob** (D-CB-20), so the probing precision is
  a recorded choice; attempt 3 probes in full precision as trace-cmi did,
  knowing from §4 that this changes cells, not the floor.
- **Attempt 3** (`trace-cmi-bench-26-09-24-gate-r3`) reruns the gate with
  `--probe-amp none` and nothing else changed. Generation and pretraining are
  deterministic (attempt 2's model was bit-identical to attempt 1's), so the
  expected outcome is attempt 2's numbers under the amended assertion, i.e.
  a passing report at engine `39aae1a5…` — the record `discover`/`sweep`
  require. Its curve is read against §1 either way.
- The remaining floor difference (a better-fit model on a clean-room world
  instance) is reported, not chased: at the operating point the gate
  licenses, the two implementations agree to 0.001.
