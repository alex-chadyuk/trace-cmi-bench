# Reproduction gate, attempt 3 (2026-09-24) — PASSED

Report: [`gate/2026-09-24-r3-gate-report.json`](../gate/2026-09-24-r3-gate-report.json)
(`passed: true`, engine `39aae1a5…`, package commit `25fafbf` = `main` after
PR #4, `probe_amp: none`). Job: one g5.xlarge (A10G), 5,327 s wall, exit
status 0, output synced, results push disabled. Replica knobs equal the run
record (`scripts/tfvars_check.py … --args-diff`: OK). `hashes.require_gate`
accepts the report at the current engine hash: `discover` and `sweep` are
licensed for benchmark corpora.

| assertion | value | band | |
|---|---|---|---|
| 1 redundancy | 0.5806 | ≥ 0.58 | ✓ |
| 2 printed τ = 3·10⁻⁵, `faithful/library` | **0.3516** (P 0.215 / R 0.977) | ≤ 0.55 (one-sided, PRD Decision Log 12) | ✓ |
| 3 blind τ\* = 3·10⁻², `faithful/library` | **0.9019** (P 0.958 / R 0.853) | ≥ 0.86 | ✓ |

## What changed since attempt 2 and what did not

Two changes were recorded before the run: the one-sided scenario 2 and the
probe in full precision (`--probe-amp none`, D-CB-20). Nothing else.

- **The model is bit-identical to attempts 1 and 2** (`model_sha256
  4aa4ac28…`): generation and pretraining are deterministic across three
  instances and two package commits.
- **Full-precision probing reproduces the bf16 curve.** The library F1 at
  every grid τ equals attempt 2's within 0.002 (at τ ≥ 10⁻⁵ within 0.0007;
  3·10⁻⁵: 0.3516 vs 0.3515; 3·10⁻²: 0.9019 vs 0.9019); the paper arm
  likewise; the same τ\* on both arms (3·10⁻², 10⁻²); the same per-lag recall
  (0.992 / 0.038 / 0 / 0 / 0 / 0); the published-clamp corruption counts
  identical to the cell for the library arm. This is the local finding of
  `findings/gate-verification-vs-trace-cmi-26-09-24.md` §4 at full scale:
  probe precision moves cells, not the observable.
- **Cost of full precision:** 1,540 s per 1,024-sequence split at
  6.9·10⁵ token-positions/s against 835 s at 1.26·10⁶ under bf16 — 1.84×
  the probe wall, same peak memory (1.66 GB device). Total 89 min against
  65. The benchmark rungs may take either mode; the choice is recorded per
  run.

## Standing

The verification against trace-cmi's recorded curves (§1 of the
verification note) applies unchanged to this run: identical at every
τ ≥ 10⁻² and ≤ 3·10⁻⁶, below in the noise-floor band through precision. The
gate scores the estimator where the implementations agree; the floor
difference (model fit ε̂ 0.013 against 0.129; world instance) is reported,
not chased. Three reports are committed for the date (attempt 1 failed on
the truth defect, attempt 2 failed the former two-sided band, attempt 3
passed); `selfcheck` still names reports by date, so the run name was added
by hand — the naming fix is M6 scope.
