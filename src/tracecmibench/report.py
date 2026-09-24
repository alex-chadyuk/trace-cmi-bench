"""Five-seed tables per cell and paired arm differences (D14; PRD scenarios 6, 8, 9, 12, 14).

    python -m tracecmibench.report --results-dir results --rung xs --variant latent \
        --pairs faithful/paper:faithful/library --reasons tables/reasons.json \
        --output-folder tables/<run>

Reads `results/<rung>/<variant>/seed=<k>/<arm>/<agg>/<grain>/{score.json,annotate.json}`.
Per `(arm, agg, grain)` every metric is `tracebench.score.headline` over the
five seeds (the cell is refused below five, scenario 9); a cell absent on
every seed must carry a reason in `--reasons` (scenario 12); a cell whose
seeds mix benchmark tool versions is refused. Bidirected columns of an arm
with a structural limitation carry the note and the count of truth
bidirected edges it could never find (scenario 8). For every `a:b` pair the
per-seed difference `a − b` is taken only after asserting `(corpus_id, seed,
model_sha256)` equal in that cell (scenario 14), then headlined. Writes
`<rung>-<variant>.json` and `.md`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tracebench.score import headline

from .arms import cell_key, parse_cell_key
from .constants import AGGREGATIONS, FAITHFUL_ARMS, GRAINS, RUNGS, SEEDS, VARIANTS
from .log import log
from .record import RunRecord, read_json, write_json

SCORE_JSON = "score.json"
ANNOTATE_JSON = "annotate.json"
REPORT_SCHEMA = "tracecmibench/report@1"

# dotted keys into the flattened per-seed record; `None` values become a recorded reason
METRICS = (
    "directed.f1", "directed.precision", "directed.recall", "directed.shd",
    "skeleton.f1", "skeleton.shd", "orientation.accuracy", "orientation.accuracy_compelled",
    "bidirected.recall", "bidirected.precision", "shd_mixed",
    "auroc.directed", "average_precision.directed", "auroc.skeleton",
    "causal_validity.sid", "causal_validity.parent_aid", "causal_validity.ancestor_aid",
    "coverage.reachable_recall_ceiling",
)
MIN_SEEDS = 5


class ReportRefusal(RuntimeError):
    pass


def flatten(score, annotate):
    """One flat dict per seed: the scorer's structural axes, the ranking's threshold-free axes,
    the causal-validity values (or their reason) and the per-lag recall."""
    flat = {}
    for axis in ("directed", "skeleton", "bidirected", "orientation"):
        for k, v in score[axis].items():
            flat[f"{axis}.{k}"] = v
    flat["shd_mixed"] = score["shd_mixed"]
    for axis in ("auroc", "average_precision"):
        for k, v in annotate["score_ranking"][axis].items():
            flat[f"{axis}.{k}"] = v
    cv = annotate["causal_validity"]
    if cv.get("value"):
        for k, v in cv["value"].items():
            flat[f"causal_validity.{k}"] = v
        flat["causal_validity.reason"] = None
    else:
        flat["causal_validity.reason"] = cv.get("reason")
    for k, v in annotate["per_lag_recall"].items():
        flat[f"per_lag_recall.{k}"] = v
    flat["coverage.reachable_recall_ceiling"] = annotate["coverage"]["reachable_recall_ceiling"]
    return flat


def load_cells(results_dir, rung, variant):
    """`{cell_key: {seed: {"flat", "annotate"}}}` over every seed directory present."""
    root = Path(results_dir) / rung / variant
    cells = {}
    for seed_dir in sorted(root.glob("seed=*")):
        seed = int(seed_dir.name.split("=")[1])
        for ann_path in sorted(seed_dir.rglob(ANNOTATE_JSON)):
            cell_dir = ann_path.parent
            score_path = cell_dir / SCORE_JSON
            if not score_path.exists():
                raise ReportRefusal(f"{cell_dir} has {ANNOTATE_JSON} but no {SCORE_JSON}")
            annotate = read_json(ann_path)
            score = read_json(score_path)
            key = annotate["cell"]
            parse_cell_key(key)
            if annotate.get("seed") is not None and int(annotate["seed"]) != seed:
                raise ReportRefusal(f"{cell_dir}: annotate says seed {annotate['seed']} but sits under seed={seed}")
            cells.setdefault(key, {})[seed] = {"flat": flatten(score, annotate), "annotate": annotate}
    return cells


def _headline(per_seed, key):
    vals = [s["flat"].get(key) for s in per_seed]
    if any(v is None for v in vals):
        reasons = sorted({str(s["flat"].get("causal_validity.reason") or "value absent on at least one seed") for s in per_seed
                          if s["flat"].get(key) is None})
        return {"metric": key, "n_seeds": len(vals), "reason": "; ".join(reasons)}
    return headline([{"v": v} for v in vals], "v") | {"metric": key}


def cell_table(per_seed, key):
    seeds = sorted(per_seed)
    if len(seeds) < MIN_SEEDS:
        raise ReportRefusal(f"cell {key} has {len(seeds)} seed(s) {seeds}; a headline needs at least {MIN_SEEDS} (PRD scenario 9)")
    versions = {s["annotate"].get("tool_version") for s in per_seed.values()}
    if len(versions) != 1:
        raise ReportRefusal(f"cell {key} mixes benchmark tool versions {sorted(map(str, versions))}")
    rows = [per_seed[s] for s in seeds]
    out = {"cell": key, "seeds": seeds, "tool_version": versions.pop(), "n_seeds": len(seeds),
           "corpus_ids": [per_seed[s]["annotate"]["corpus_id"] for s in seeds],
           "model_sha256": [per_seed[s]["annotate"]["model_sha256"] for s in seeds],
           "metrics": {m: _headline(rows, m) for m in METRICS}}
    lags = sorted({int(k.split(".")[1]) for r in rows for k in r["flat"] if k.startswith("per_lag_recall.")})
    out["per_lag_recall"] = {str(k): _headline(rows, f"per_lag_recall.{k}") for k in lags
                             if all(f"per_lag_recall.{k}" in r["flat"] for r in rows)}
    lim = rows[0]["annotate"].get("structural_limitation")
    if lim and lim.get("assumption"):
        out["structural_limitation"] = {"assumption": lim["assumption"],
                                        "truth_bidirected_edges": [r["annotate"]["structural_limitation"]["truth_bidirected_edges"] for r in rows],
                                        "note": lim.get("note")}
    out["in_regime"] = [bool(r["annotate"]["oracle"]["in_regime"]) for r in rows]
    return out


def paired_table(cells, arm_a, arm_b, agg, grain):
    ka, kb = cell_key(arm_a, agg, grain), cell_key(arm_b, agg, grain)
    if ka not in cells or kb not in cells:
        return None
    a, b = cells[ka], cells[kb]
    seeds = sorted(set(a) & set(b))
    if len(seeds) < MIN_SEEDS:
        raise ReportRefusal(f"pair {ka} vs {kb}: {len(seeds)} shared seed(s); a paired headline needs {MIN_SEEDS} (PRD scenario 9)")
    triples = []
    for s in seeds:
        ta = (a[s]["annotate"]["corpus_id"], int(a[s]["annotate"]["seed"]), a[s]["annotate"]["model_sha256"])
        tb = (b[s]["annotate"]["corpus_id"], int(b[s]["annotate"]["seed"]), b[s]["annotate"]["model_sha256"])
        if ta != tb:
            raise ReportRefusal(f"pair {ka} vs {kb} at seed {s}: (corpus_id, seed, model_sha256) differ: {ta} vs {tb} (PRD scenario 14)")
        triples.append(list(ta))
    diffs = []
    for s in seeds:
        fa, fb = a[s]["flat"], b[s]["flat"]
        diffs.append({"flat": {k: (fa[k] - fb[k]) if (fa.get(k) is not None and fb.get(k) is not None and not isinstance(fa[k], str)) else None
                               for k in set(fa) | set(fb)}})
    lags = sorted({int(k.split(".")[1]) for d in diffs for k in d["flat"] if k.startswith("per_lag_recall.") and d["flat"][k] is not None})
    return {"pair": f"{arm_a} - {arm_b}", "agg": agg, "grain": grain, "seeds": seeds, "triples": triples,
            "metrics": {m: _headline(diffs, m) for m in METRICS},
            "per_lag_recall": {str(k): _headline(diffs, f"per_lag_recall.{k}") for k in lags}}


def expected_cells(cells, arms):
    keys = set(cells)
    for arm in arms:
        for agg in AGGREGATIONS:
            for grain in GRAINS:
                keys.add(cell_key(arm, agg, grain))
    return sorted(keys)


def run_report(args, rec):
    cells = load_cells(args.results_dir, args.rung, args.variant)
    reasons = read_json(args.reasons) if args.reasons else {}
    arms = set(FAITHFUL_ARMS) | {parse_cell_key(k)[0] for k in cells}
    tables, absent = {}, {}
    for key in expected_cells(cells, arms):
        if key in cells:
            tables[key] = cell_table(cells[key], key)
        elif key in reasons:
            absent[key] = reasons[key]
        else:
            raise ReportRefusal(f"cell {key} is absent on every seed and --reasons gives no reason for it (PRD scenario 12)")
    pairs = {}
    for spec in args.pairs:
        arm_a, arm_b = spec.split(":")
        for agg in AGGREGATIONS:
            for grain in GRAINS:
                t = paired_table(cells, arm_a, arm_b, agg, grain)
                if t is not None:
                    pairs[f"{arm_a}:{arm_b}/{agg}/{grain}"] = t
    report = {"schema": REPORT_SCHEMA, "rung": args.rung, "variant": args.variant, "n_cells": len(tables), "cells": tables,
              "absent": absent, "pairs": pairs, "metrics": list(METRICS)}
    stem = f"{args.rung}-{args.variant}"
    write_json(rec.out_dir / f"{stem}.json", report)
    (rec.out_dir / f"{stem}.md").write_text(render_markdown(report), encoding="utf-8")
    log({"event": "report_done", "rung": args.rung, "variant": args.variant, "n_cells": len(tables), "n_absent": len(absent), "n_pairs": len(pairs)})
    return {"report": f"{stem}.json", "markdown": f"{stem}.md", "n_cells": len(tables), "n_absent": len(absent), "n_pairs": len(pairs)}


# --- rendering -------------------------------------------------------------------------------------
SHOW = ("directed.f1", "directed.precision", "directed.recall", "directed.shd", "skeleton.f1", "orientation.accuracy",
        "auroc.directed", "average_precision.directed", "bidirected.recall", "shd_mixed", "causal_validity.sid",
        "causal_validity.parent_aid", "coverage.reachable_recall_ceiling")


def _fmt(h):
    if "reason" in h:
        return f"n/a ({h['reason']})"
    return f"{h['mean']:.3f} ± {h['std']:.3f}"


def render_markdown(report):
    lines = [f"# {report['rung']} / {report['variant']} — five-seed tables", ""]
    for grain in GRAINS:
        rows = [(k, t) for k, t in report["cells"].items() if k.endswith(f"/{grain}")]
        if not rows:
            continue
        lines += [f"## {grain} grain", "", "| cell | " + " | ".join(SHOW) + " |", "|---|" + "---|" * len(SHOW)]
        for key, t in rows:
            vals = []
            for m in SHOW:
                v = _fmt(t["metrics"][m])
                if m.startswith("bidirected") and "structural_limitation" in t:
                    n = t["structural_limitation"]["truth_bidirected_edges"]
                    v += f" [structural: {t['structural_limitation']['assumption']}; truth bidirected edges {min(n)}–{max(n)}]"
                vals.append(v)
            lines.append(f"| {key} | " + " | ".join(vals) + " |")
        lines.append("")
        lag_rows = [(k, t) for k, t in rows if t["per_lag_recall"]]
        if lag_rows:
            lags = sorted({int(l) for _, t in lag_rows for l in t["per_lag_recall"]})
            lines += ["### per-lag recall at the frozen τ", "", "| cell | " + " | ".join(f"lag {l}" for l in lags) + " |", "|---|" + "---|" * len(lags)]
            for key, t in lag_rows:
                lines.append(f"| {key} | " + " | ".join(_fmt(t["per_lag_recall"][str(l)]) if str(l) in t["per_lag_recall"] else "—" for l in lags) + " |")
            lines.append("")
    if report["absent"]:
        lines += ["## absent cells", ""] + [f"- `{k}`: {v}" for k, v in report["absent"].items()] + [""]
    if report["pairs"]:
        lines += ["## paired differences (a − b, per seed, then five-seed headline; (corpus_id, seed, model_sha256) equal per cell)", ""]
        lines += ["| pair | " + " | ".join(SHOW) + " |", "|---|" + "---|" * len(SHOW)]
        for key, t in report["pairs"].items():
            lines.append(f"| {key} | " + " | ".join(_fmt(t["metrics"][m]) for m in SHOW) + " |")
        lines.append("")
    return "\n".join(lines)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--rung", required=True, choices=RUNGS)
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--pairs", required=True, nargs="+", help="arm pairs a:b to difference, e.g. faithful/paper:faithful/library")
    p.add_argument("--reasons", required=True, help="JSON {cell_key: reason} for cells absent on every seed, or '' when none")
    p.add_argument("--output-folder", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.reasons = args.reasons or None
    with RunRecord(args.output_folder, "report", vars(args)) as rec:
        res = run_report(args, rec)
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
