"""Structural constants: names, registries and the values the PRD pins.

Nothing here is a tunable. Every algorithm knob is a required command-line
flag whose value rides the run record (`run/arguments.json`); the values below
are data semantics (token ids, file names, the arm registry) or numbers the
PRD itself fixes (the reproduction-gate bands, the in-regime boundary), which
a run may report against but never choose.
"""
from tracebench.constants import (  # noqa: F401  (re-exported: one definition across the two repositories)
    BOS,
    DEFAULT_FLOOR,
    EOS,
    FLOOR_SWEEP,
    GRAINS,
    METHOD_READABLE_PREFIXES,
    N_SPECIALS,
    ORDERINGS,
    OUTCOME_IDS,
    OUTCOME_NAMES,
    OUTCOME_OK,
    PAD,
    SPLITS,
    UNK,
    VARIANT_LATENT,
    VARIANT_TWIN,
    VARIANTS,
)

TOOL_NAME = "trace-cmi-bench"

# --- the benchmark this repository scores ---------------------------------------------
DATASET_REPO = "chadyuk/trace-bench"          # dataset host id (public)
BENCHMARK_TAG = "v0.3.0"                       # the release tag pinned in pyproject.toml
BENCHMARK_TOOL_VERSION = "0.3.0"               # manifest.tool_version every scored corpus must carry
RUNGS = ("xs", "s", "m", "l", "xl")
SEEDS = (0, 1, 2, 3, 4)
COMPLETE_MARKER = "COMPLETE"
MANIFEST_JSON = "manifest.json"
MODEL_VOCAB_JSON = "model-vocab.json"
EXPORT_STATS_JSON = "export-stats.json"
VIEWS_DIR = "views"
SEQUENCES_DIR = "sequences"
GRAPHS_DIR = "graphs"
TOPOLOGY_DIR = "topology"
PRIOR_JSON = "prior.json"
# the one corpus path outside the method-readable set that the prior arm may read (PRD non-negotiable 2)
PRIOR_RELATIVE_PATH = f"{TOPOLOGY_DIR}/{PRIOR_JSON}"

# --- run record (one directory per run) --------------------------------------------------
RUN_DIR = "run"
ARGUMENTS_JSON = "arguments.json"
RESULTS_JSON = "results.json"
RUN_META_JSON = "run_meta.json"
ARTIFACTS_JSON = "artifacts.json"
ARTIFACTS_SCHEMA = "tracecmibench/artifacts@1"
LOG_JSONL = "log.jsonl"

# --- per-stage artifacts ------------------------------------------------------------------
PREPARE_JSON = "prepare.json"
MODEL_PT = "model.pt"
CHECKPOINT_FMT = "checkpoint-{step:07d}.pt"
MATRICES_NPZ_FMT = "matrices-{grain}.npz"
SCORES_NPZ_FMT = "scores-{grain}.npz"
SWEEP_SCORES_NPZ_FMT = "scores-{arm}-c{c}-N{n}-{grain}.npz"     # arm with '/' -> '-'
PREDICTION_JSON_FMT = "prediction-{grain}-{agg}.json"
RANKING_JSON_FMT = "ranking-{grain}-{agg}.json"
RANKING_LAG_JSON_FMT = "ranking-{grain}-{agg}-lag{lag}.json"
PREDICTION_LAG_JSON_FMT = "prediction-{grain}-{agg}-lag{lag}.json"   # per-lag thresholded, for per-lag recall
VAL_TABLE_JSON = "val-table.json"
GATE_DIR = "gate"
FREEZES_DIR = "freezes"
RESULTS_DIR = "results"
TABLES_DIR = "tables"
PLANS_DIR = "plans"
FINDINGS_DIR = "findings"

# --- arms -----------------------------------------------------------------------------------
# history source of the fixed context positions (D-CB-9)
HISTORY_OBSERVED = "observed"     # teacher-forced real prefix (paper Fig. 3 / App. C.2)
HISTORY_SAMPLED = "sampled"       # positions [g, c) redrawn ancestrally per particle (author library)
HISTORIES = (HISTORY_OBSERVED, HISTORY_SAMPLED)

# the four statistics every probing pass emits per cell (D-CB-15)
STAT_KL_MEAN_BF = "kl_mean_bf"    # mean_l KL(p_l || q_l)   base-first, per particle then mean
STAT_KL_MEAN_DF = "kl_mean_df"    # mean_l KL(q_l || p_l)   do-first (cause observed first), per particle then mean
STAT_MEAN_KL_BF = "mean_kl_bf"    # KL(p_bar || q_bar)      mean over particles then KL, base-first
STAT_MEAN_KL_DF = "mean_kl_df"    # KL(q_bar || p_bar)      mean then KL, do-first
STATISTICS = (STAT_KL_MEAN_BF, STAT_KL_MEAN_DF, STAT_MEAN_KL_BF, STAT_MEAN_KL_DF)

ARM_FAITHFUL_PAPER = "faithful/paper"
ARM_FAITHFUL_LIBRARY = "faithful/library"
ARMS = {
    # the paper's operational figure: observed history, mediator draws as particles, KL(base || do) after the mean (Eq. 9/10)
    ARM_FAITHFUL_PAPER: {"class": "reference", "history": HISTORY_OBSERVED, "statistic": STAT_MEAN_KL_BF},
    # the author library's core path: ancestral history particles, per-particle KL(cause observed || cause noised), then mean
    ARM_FAITHFUL_LIBRARY: {"class": "reference", "history": HISTORY_SAMPLED, "statistic": STAT_KL_MEAN_DF},
}
FAITHFUL_ARMS = (ARM_FAITHFUL_PAPER, ARM_FAITHFUL_LIBRARY)
AGGREGATIONS = ("max", "mean")                 # corpus-level sub-arms over position-pair occurrences (D-CB-8)
SEQUENCE_SAMPLES = ("head", "uniform")         # --sequence-sample (D-CB-14)
DIVERGENCES = ("log1mexp", "compat")           # --divergence (D-CB-6)
LR_SCHEDULES = ("cosine", "constant")
DEVICES = ("cpu", "cuda")

# a faithful arm cannot emit a bidirected edge: the assumption it rests on (PRD scenario 8)
STRUCTURAL_LIMITATION = "causal_sufficiency"

# the source files whose sha256 is the engine hash a gate report binds (D11)
ENGINE_MODULES = ("staircase", "probe", "statistics", "project", "select", "model", "data", "vocab")

# --- PRD-pinned values (reported against, never chosen) -------------------------------------
ORACLE_IN_REGIME = 0.1                         # eps_hat below this = in regime (paper's phase transition)
GATE_ARM = ARM_FAITHFUL_LIBRARY                # the reading the replication note characterised
GATE_TAU_PRINTED = 3e-5                        # the paper's Table 2 threshold
GATE_F1_PRINTED_MAX = 0.55                     # scenario 2: the printed threshold must NOT replicate (one-sided since the
                                               # 2026-09-24 amendment; the band's lower edge 0.40 measured the noise floor)
GATE_F1_SELECTED_MIN = 0.86                    # scenario 3: blind val-selected threshold must
GATE_REDUNDANCY_MIN = 0.58                     # scenario 1: the paper's redundancy band
GATE_CONTEXT = 6                               # BOS-counted, as the note characterised
GATE_GUIDANCE = 3
GATE_PARTICLES = 128
GATE_TAU_GRID = tuple(sorted({
    1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1,   # half-decades
    1.72e-5,                                                             # C / |X| anchor at |X| = 1000
}))
GATE_TRUTH_DELTA = 0.05                        # paper E.1 counterfactual-KL threshold (D-CB-13)
GATE_TRUTH_COUNTERFACTUALS = 10
