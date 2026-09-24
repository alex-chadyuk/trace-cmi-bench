"""Freeze the validation-selected values of one corpus (D12; plan §4; PRD scenario 22).

    python -m tracecmibench.freeze --val-table out/<scoresweep-run>/val-table.json \
        --rung xs --variant latent --seed 0 --freezes-dir freezes --output-folder out/<run>

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


def build_freeze(table, val_table_path, rung, variant, seed):
    return {
        "schema": FREEZE_SCHEMA, "rung": rung, "variant": variant, "seed": int(seed),
        "corpus_id": table.get("corpus_id"), "tool_version": table.get("tool_version"), "config_hash": table.get("config_hash"),
        "model_sha256": table.get("model_sha256"), "commit": git_commit(), "frozen_at": now_iso(),
        "val_table_sha256": sha256_file(val_table_path), "val_table_name": Path(val_table_path).name,
        "taus": table.get("taus"), "grains": table.get("grains"), "cells": select_cells(table),
    }


def run_freeze(args, rec):
    table = read_json(args.val_table)
    freeze = build_freeze(table, args.val_table, args.rung, args.variant, args.seed)
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
    p.add_argument("--val-table", required=True)
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
