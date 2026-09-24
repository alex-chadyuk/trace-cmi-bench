# RUN — environment, deviation record, verification, run registry

The reproducibility record for `trace-cmi-bench`: what runs where, every
numbered departure from the paper (each dated **before** the run that depends
on it, PRD non-negotiable 10), how a result is verified, and one registry row
per run. Paper references are to Math & Lienhart 2026 (arXiv:2602.01135);
"the note" is the lab's replication note (*Replicating TRACE: A
Practitioner's Guide to Its Threshold and Particle Budget*).

## Environment

- **Local:** conda env `trace-cmi-bench` from `environment.yml` (python 3.11;
  torch, numpy, pyarrow, huggingface_hub and `trace-bench[sid]` at tag
  `v0.3.0` via `requirements.txt`), created 2026-09-23 on macOS (Apple
  silicon, CPU torch). Local use is the test suite on tiny fixtures and record
  rendering only: **nothing runs locally, the smallest rung included** (PRD
  non-negotiable 11, operating rule of the owner; the package itself runs
  anywhere).
- **Runs:** cloud GPU instances provisioned by the workspace's private
  terraform, one dated replica file per run, no command-line overrides. The
  replica carries the verbatim invocation; the run record
  (`run/arguments.json`) republishes every algorithm value. Provisioning
  values never appear here.
- **Records:** every run writes `run/{arguments,results,run_meta}.json`
  (`run_meta` = git commit, torch/CUDA, GPU model, host RAM, CPU count, wall
  clock, peak device and resident memory, `pip freeze`); binaries ride object
  storage under content hashes with a location-free `artifacts.json`
  manifest.
- **Benchmark:** trace-bench corpora at tool version **0.3.0** only; `report`
  refuses a cell that mixes tool versions.

## Deviations from the paper (D-CB-n)

Each entry states what it **follows** and what it **deviates from**. Dated
2026-09-23 unless noted; all predate the first run.

| id | follows | deviates from |
|---|---|---|
| D-CB-1 | Eq. 21 generator (sparse mixed-sign `W`, sparsity 0.9, fixed embeddings, one-hidden-layer MLP, `e^{-(k-1)}` lag decay) | `W` is multiplied by a required `--w-scale`: the released author library cannot reach the paper's redundancy band `R ≥ 0.58` at any sparsity (unit scale tops out near 0.12), and the author has disclosed that the published table's own run used a different sparsity plus an undisclosed weight scale. The gate asserts the band, not the world (PRD scenario 1). |
| D-CB-2 | Def. 4.6 staircase (row `j` fixes the observed history through window position `j − 1`, randomises the rest) | rows = `Lc + 1` with an all-noise row 0 so the first window position is itself testable as a cause; the reference construction never tests it. |
| D-CB-3 | Def. 4.6 uniform do-operator "over the alphabet" | the proposal draws over real event tokens only — never PAD/BOS/EOS/UNK, never EOS as a counterfactual event. |
| D-CB-4 | §5.1 context truncation `x_<t ≈ x_0:c` | `c` is a required knob swept blind on validation and frozen with τ (request grid {1, 2, 3}, session grid {2, 4, 8}); the paper's `max(0.1L, 20)` exceeds these sequences (request `n_spans` p50 = 5, p99 = 11). Fig. 9's caption value (6) is the gate's value. BOS counts toward `c` (D-CB-5). |
| D-CB-5 | fixed `L = 64` sequences | variable-length sequences: `BOS` anchors position 0 and counts toward `c`, `EOS` models termination, `PAD` is masked from loss and attention, `--max-len` caps the real tokens (tail dropped, truncation count reported), batches pad to the batch maximum. |
| D-CB-6 | Bernoulli KL between the two event probabilities | computed in float64 log space from `log_softmax` with `log(1 − p) = log1mexp(log p)`, `log p` clamped at `−1e-12` and clamped cells counted. The published float32 `clamp(p, ε, 1 − ε)` is a no-op at the upper end and turns a saturated particle into a zero or a float-max cell; it is retained behind `--divergence compat --clamp-eps` and the cells it corrupts are counted per direction (`n_cell_to_zero`, `n_cell_to_fmax`) on the same inputs in the same pass — reported, never gated (PRD scenario 15). |
| D-CB-7 | Eq. 22 oracle score `ε̂ = (L − H) / (log|X| − H)` | `H` is never the minimum validation loss (which degenerates to `ε̂ ≡ 0` on any converging run): on a benchmark corpus it is an independent order-`k` n-gram conditional-entropy floor over the training split; in the gate it is the generator's exact conditional entropy. Reported with `in_regime = ε̂ < 0.1`, never gated. |
| D-CB-8 | Def. 3.2 per-sequence summary graph (type edge iff any position pair carries it) | corpus-level aggregation over every within-sequence position-pair occurrence of `(u, v)` with lag `≤ M`: **max-pool** and **mean** run as separate sub-arms `faithful/{paper,library}/{max,mean}` reading the same emitted matrices. The gate's per-sequence F1 uses the Def. 3.2 union. |
| D-CB-9 | Eq. 7 history particles `x_<t^(l) ~ P_θ` (Prop. 4.2) vs. App. C.2 teacher forcing | the two readings are two arms. `faithful/paper` keeps the observed prefix `[0, c)`; `faithful/library` redraws positions `[g, c)` ancestrally from the frozen model per particle (`--guidance g`, `1 ≤ g ≤ c`, real tokens before `g`; `g = c` makes the two histories coincide and is recorded), the draws restricted to real event tokens (never EOS mid-sequence), with the window rows observed. Gathering is always at the observed token (Eq. 2). |
| D-CB-10 | Def. 4.6 independent mediator draws | one uniform noise vector per particle shared across all staircase rows (common random numbers): adjacent rows differ at exactly one position, so the paired comparison is a paired difference. |
| D-CB-11 | A3.4 temporal precedence (an event may influence later events only) | the sequence ordering is a design decision: the faithful arms use the views' `end` ordering (spans by completion time, callee precedes caller = the direction outcomes propagate in the target); `start` is one flag away (`--ordering`). |
| D-CB-12 | the full `L(L+1)/2` score matrix | within-operation token pairs and self-loops are dropped at projection (the scorer never scores them) and `--max-lag M` is a true per-effect lag bound (rows outside `[q − M, q)` are neither built nor forwarded for column `q`), not the reference's "last `M` rows". |
| D-CB-13 | App. E.1 truth: `i → j` iff `E KL(P_orig ‖ P_do(x_i := u)) > δ = 0.05`, 10 uniform replacements | the threshold symbol is overloaded in the paper (edge threshold τ and truth threshold δ); here the truth is the mean over 10 counterfactual draws of `KL(P(X_j | x_<j) ‖ P(X_j | x_<j, x_i := u))` in the paper's argument order (the draw may equal the observed token, a `1/|X|` no-op), per sequence, positional, direct effect within lag `h`. For the gate's per-sequence F1 both the truth and the prediction are projected by Def. 3.2 with self-loops dropped, and the truth is restricted to pairs whose cause lies in the window (sequence position `≥ c`), the set the staircase can test; the gate probes every lag (`--max-lag 63`), so a predicted edge at a lag beyond `h` counts against precision. |
| D-CB-14 | one sequence per summary graph, evaluated over the test set | `--num-sequences` and `--sequence-sample {head, uniform}` (seeded reservoir) per rung, recorded; the gate probes 1,024 head sequences of each split. A coverage rule (fraction of truth edges co-occurring in the sample) may raise the sample in a new dated replica before a freeze. |
| D-CB-15 | one statistic per cell | four statistics are emitted per cell from every pass — `kl_mean_bf`, `kl_mean_df`, `mean_kl_bf`, `mean_kl_df` (per-particle-KL-then-mean vs mean-then-KL, base-first vs do-first) — so the paper-vs-library difference decomposes into history source, statistic and KL order. Only the arm's own statistic is thresholded. |
| D-CB-16 | abstract event types `X` | tokens are the shipped `views/<view>/model-vocab.json` as-is (specials, base ops = `op:ok`, minted non-OK variants above the correlator's `min_count`); unminted variants fold to their base op (fold count reported by `prepare`) and the number of scoring-universe tokens the method can never emit is reported per corpus. |
| D-CB-17 | one dataset, one model | one model per corpus, trained on the `end-session` view; both grains are probed from that frozen model (request sequences from `end-request`, session sequences from `end-session` under `--max-len`). `results.json.model_sha256` is the hash every arm and comparison record carries. |
| D-CB-18 | one thresholded edge list | two prediction files per read: `prediction-<grain>-<agg>.json` (edges with score > τ, for the structural axes) and `ranking-<grain>-<agg>.json` (every scored pair, for AUROC/AP), plus per-lag rankings — because the benchmark's scorer treats every listed edge as present. |
| D-CB-19 | `N = 128` particles (Table 2) | `N` is swept on {2, 8, 32, 128} on validation and frozen with τ and `c`; the note found the particle average saturated at `N = 2`. The gate runs `N = 128`. |

Deviations for later arms (`prior/topology`, the six shared-model baselines,
`improve/*`, `baseline/causalnet`) are numbered in the pre-registered plan of
each arm before its code exists.

## Verification

- **Every commit:** `pytest tests/` green in the `trace-cmi-bench` env;
  `tests/test_no_defaults.py` (no algorithm knob has a default) and
  `tests/test_repo_hygiene.py` (no private identifier, location or binary in
  any tracked file) fail the build.
- **Gate (M5):** `gate/<date>-gate-report.json` committed with
  `passed: true` and the `engine_sha256` of the probing modules; `discover`
  and `sweep` refuse any benchmark corpus without a passing report whose hash
  equals the current engine hash.
- **Per run:** the executing host's log shows the command exiting with
  status 0 and the output sync completing; `run/*.json` carry every
  scenario-25 field; a registry row lands below.
- **Per rung:** `report` builds every cell from five seeds or records the
  reason; the paper-vs-library difference table exists per axis and per lag;
  every faithful bidirected column carries the structural-limitation note;
  the wall clock lands within twice its estimate before the next rung opens.

## Run registry

One row per run, newest last. Object-store locations are never written here;
the run directory's `artifacts.json` (hashes only) and the private replica
are the pointers.

| date (UTC) | run | command | rung/variant/seed | split | model sha256 (8) | exit | wall clock | gpu | outcome |
|---|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | — | no run yet |
