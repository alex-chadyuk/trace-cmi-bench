# trace-cmi-bench

A public, MIT-licensed, clean-room implementation of **TRACE** (Math & Lienhart
2026, *Your Autoregressive Model Already Reveals the Causal Graph*,
arXiv:2602.01135) benchmarked on the
[trace-bench](https://github.com/alex-chadyuk/trace-bench) corpora
(dataset: [`chadyuk/trace-bench`](https://huggingface.co/datasets/chadyuk/trace-bench),
CC BY 4.0) through the benchmark's own scorer.

TRACE trains any autoregressive model by next-token prediction, freezes it,
and reads a causal graph out of it: for every ordered pair of positions in a
sequence it compares the model's probability of the observed effect with the
cause present and with the cause replaced by uniform noise, controlling the
intermediate positions by randomising them (a staircase of intervened inputs
scored in one batch), then thresholds the resulting score matrix and projects
it onto event types. This repository writes that engine from the paper and
runs it, in **both readings the paper admits**, against a benchmark whose
ground truth was emitted by construction and never inferred from the data.

## What this repository claims

A reference characterisation, not a state-of-the-art claim: TRACE's numbers
on every rung, variant, grain and seed of the benchmark, on every axis its
scorer computes, behind a reproduction gate and a validation → freeze → test
protocol, with every departure from the paper numbered and dated in
[`RUN.md`](RUN.md). The results are co-published with the trace-bench dataset
paper.

## Status

Under construction. The milestone table lives in the (private) plan of
record; the public state is what `RUN.md` records:

| milestone | what lands | state |
|---|---|---|
| M0 | scaffold: packaging, records, artifacts, hygiene and no-defaults gates, deviation register | built |
| M1 | corpus adapter, vocabulary, prior door, `pull`, `prepare` | built |
| M2 | decoder backbone, batching, `pretrain`, entropy floor | built |
| M3 | the paper's synthetic generator and its exact interventional truth | built |
| M4 | staircase, probe, statistics, projection, selection, prediction writer | built |
| M5 | `selfcheck`: the reproduction gate, run once on a cloud instance and committed under `gate/` | **passed** — attempt 3 (2026-09-24, `gate/2026-09-24-r3-gate-report.json`, engine `39aae1a5`): R 0.5806, library F1 0.352 at the printed threshold (must not replicate, ≤ 0.55) and 0.902 at the blind-selected threshold (≥ 0.86), the replication note's numbers. Attempts 1–2 are committed beside it (`findings/gate-attempt-{1,2,3}-26-09-24.md`); the engine is verified curve by curve against the predecessor's runs (`findings/gate-verification-vs-trace-cmi-26-09-24.md`) |
| M6 | `discover`, `sweep`, `freeze`, `scoresweep`, `annotate`, `report` | — |
| M7+ | the benchmark ladder xs → s → m → l → xl | — |

## Install

```bash
conda env create -f environment.yml
conda activate trace-cmi-bench
pip install -e ".[dev]"
pytest tests/
```

The benchmark package is pinned to the release tag whose corpora this
repository scores (`trace-bench[sid] @ v0.3.0`); `sid` pulls `gadjid` for
the causal-validity axis on the observable twin.

## Commands

Every command runs as `python -m tracecmibench.<command>`. **Every knob is a
required flag; nothing has a default** (`tests/test_no_defaults.py` fails the
build otherwise), so a run's `run/arguments.json` reconstructs it exactly.

| command | what it does |
|---|---|
| `pull` | fetch one corpus (or its method-readable subset) from the dataset host and verify every file against the corpus manifest |
| `prepare` | report the substrate the method can see: rows per split, vocabulary size, length quantiles, entropy floor |
| `pretrain` | train one decoder on a corpus's training split by next-token prediction; report the oracle score |
| `selfcheck` | the reproduction gate: the paper's synthetic experiment against the registered bands; licenses everything downstream |
| `discover` | run one named arm on one corpus with one frozen model; write the score matrices and the prediction files |
| `sweep` / `freeze` | blind grid on the validation split; dated, committed record of the chosen values |
| `scoresweep` / `annotate` | thin wrappers that call the benchmark's scorer and add provenance and coverage fields; no metric arithmetic |
| `report` | five-seed tables per cell, paired arm-vs-faithful differences |
| `artifacts` | object-store push/pull/verify of a run directory; the manifest it writes carries hashes, never a location |

Scoring is not a command here: `python -m tracebench.score` is invoked on the
prediction files this tool writes, and its report is consumed unchanged.

## What the method may read

A method reads a corpus only through the benchmark's accessor
(`tracebench.allowlist.open_for_method`), which permits the raw feed and the
correlated views and refuses the graphs, labels, oracle and manifest. The
vocabulary is the views' own `model-vocab.json`, never the alphabet record.
The one disclosed exception is the prior-informed arm, which reads the
shipped deployment topology (`topology/prior.json`) through a separately named
door and records every such read in its run record.

## Layout

```
src/tracecmibench/   the package (one build_parser()/main(argv) per command module)
tests/               flat pytest suite; tests/fixture_corpus.py writes a tiny corpus that passes the scorer's self-check
RUN.md               environment, numbered deviations D-CB-n, verification, run registry
gate/                committed reproduction-gate reports
freezes/             dated frozen-threshold records, one per corpus
plans/ findings/     pre-registered plan and scored findings per arm
results/ tables/     per-run score and annotation records; five-seed tables
scripts/             replica checker for the private provisioning layer
```

Trained models, score matrices, prediction and ranking files live in object
storage under content hashes and are never committed; corpora never enter the
repository. Provisioning values (instance types, storage locations, account
identifiers) live outside it; `tests/test_repo_hygiene.py` fails the build if
any reach a tracked file.

## Clean room

The engine is written from the paper, the lab's replication note and the
benchmark's source. No source, configuration or identifier from any private
predecessor enters this repository.

## Licence

MIT (see `LICENSE`). The corpora are CC BY 4.0 under their own release.
