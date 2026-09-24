"""Freeze the validation-selected values of one corpus (D12; plan §4; PRD scenario 22).

    python -m tracecmibench.freeze --val-tables out/<scoresweep-request>/val-table.json out/<scoresweep-session>/val-table.json \
        --rung xs --variant latent --seed 0 --freezes-dir freezes --output-folder out/<run>

Several val tables (one per grain, as Job V writes them) are merged into one
freeze; they must name the same corpus, model, tool version and config hash.

Per `(arm, aggregation, grain)` the full-grid argmax of the scorer's directed
F1 at the default floor over `(c, N, τ)`; ties → smaller `N`, then smaller
`c`, then larger τ. Writes `freezes/<date>-<rung>-<variant>-s<k>.json` with
the corpus identity, the benchmark tool version and config hash, the model
hash, the package commit at freeze time, the val table's sha256, the τ grid
and the chosen `{tau, c, N, g}` per cell; refuses to overwrite. The record
is then committed (one freeze commit per rung) and `discover --split test`
asserts on that commit.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from .arms import cell_key
from .constants import RUNGS, SEEDS, VARIANTS
from .log import log, now_iso
from .record import RunRecord, git_commit, read_json, sha256_file, write_json

FREEZE_SCHEMA = "tracecmibench/freeze@1"


class FreezeExists(FileExistsError):
    pass


def select_cells(table):
    """The argmax rows per cell with the plan's tie-breaks."""
    by_cell = {}
    for r in table["cells"]:
        if r["floor"] != table["cells"][0]["floor"]:
            raise ValueError("val table mixes floors; the freeze selects at the default floor only")
        by_cell.setdefault(cell_key(r["arm"], r["agg"], r["grain"]), []).append(r)
    chosen = {}
    for key, rows in by_cell.items():
        best = max(rows, key=lambda r: (r["directed"]["f1"], -r["N"], -r["c"], r["tau"]))
        chosen[key] = {"tau": float(best["tau"]), "c": int(best["c"]), "N": int(best["N"]), "g": int(best["g"]),
                       "val_directed_f1": float(best["directed"]["f1"]), "val_directed_precision": float(best["directed"]["precision"]),
                       "val_directed_recall": float(best["directed"]["recall"]), "n_edges": int(best["n_edges"]), "floor": float(best["floor"]),
                       "n_rows_considered": len(rows)}
    return chosen


def freeze_name(date, rung, variant, seed):
    return f"{date}-{rung}-{variant}-s{seed}.json"


def merge_tables(tables):
    """One table from the per-grain tables of one corpus; refuses mixed corpora, models or grids."""
    first = tables[0]
    for t in tables[1:]:
        for k in ("corpus_id", "model_sha256", "tool_version", "config_hash", "taus"):
            if t.get(k) != first.get(k):
                raise ValueError(f"val tables disagree on {k}: {first.get(k)!r} vs {t.get(k)!r}")
    grains = []
    for t in tables:
        for g in t.get("grains", []):
            if g in grains:
                raise ValueError(f"grain {g} appears in more than one val table")
            grains.append(g)
    return {**first, "grains": grains, "cells": [c for t in tables for c in t["cells"]],
            "coverage": {k: v for t in tables for k, v in t.get("coverage", {}).items()}}


def build_freeze(table, val_table_paths, rung, variant, seed):
    paths = [Path(p) for p in val_table_paths]
    return {
        "schema": FREEZE_SCHEMA, "rung": rung, "variant": variant, "seed": int(seed),
        "corpus_id": table.get("corpus_id"), "tool_version": table.get("tool_version"), "config_hash": table.get("config_hash"),
        "model_sha256": table.get("model_sha256"), "commit": git_commit(), "frozen_at": now_iso(),
        "val_tables": [{"name": p.name, "sha256": sha256_file(p)} for p in paths],
        "taus": table.get("taus"), "grains": table.get("grains"), "cells": select_cells(table),
    }


def run_freeze(args, rec):
    table = merge_tables([read_json(p) for p in args.val_tables])
    freeze = build_freeze(table, args.val_tables, args.rung, args.variant, args.seed)
    date = dt.datetime.now(dt.timezone.utc).date().isoformat()
    path = Path(args.freezes_dir) / freeze_name(date, args.rung, args.variant, args.seed)
    if path.exists():
        raise FreezeExists(f"{path} exists; a freeze is never overwritten (a re-freeze is a new dated file with its reason recorded)")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, freeze)
    log({"event": "freeze_written", "path": str(path), "n_cells": len(freeze["cells"])})
    return {"freeze": str(path), "n_cells": len(freeze["cells"]), "cells": freeze["cells"], "corpus_id": freeze["corpus_id"],
            "model_sha256": freeze["model_sha256"]}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val-tables", required=True, nargs="+", help="one val-table.json per grain of one corpus")
    p.add_argument("--rung", required=True, choices=RUNGS)
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--seed", required=True, type=int, choices=SEEDS)
    p.add_argument("--freezes-dir", required=True, help="the repository's freezes/ directory")
    p.add_argument("--output-folder", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, "freeze", vars(args)) as rec:
        res = run_freeze(args, rec)
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
