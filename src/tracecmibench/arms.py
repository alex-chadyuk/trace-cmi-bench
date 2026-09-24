"""The arm registry (D12): an arm name → the history source of its context
positions and the statistic it thresholds; the sub-arm is the corpus-level
aggregation (`max` or `mean`, D-CB-8). Later arms register here.

A cell of the freeze, the val table and the report is keyed
`<arm>/<agg>/<grain>`.
"""
from __future__ import annotations

from .constants import AGGREGATIONS, ARMS, GRAINS


def arm_spec(name):
    if name not in ARMS:
        raise ValueError(f"unknown arm {name!r}; registered: {sorted(ARMS)}")
    return ARMS[name]


def arm_slug(name):
    """File-name form of an arm name (`faithful/paper` → `faithful-paper`)."""
    return arm_spec(name) and name.replace("/", "-")


def cell_key(arm, agg, grain):
    if agg not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {agg!r}")
    if grain not in GRAINS:
        raise ValueError(f"grain must be one of {GRAINS}, got {grain!r}")
    arm_spec(arm)
    return f"{arm}/{agg}/{grain}"


def parse_cell_key(key):
    arm, agg, grain = key.rsplit("/", 2)
    cell_key(arm, agg, grain)
    return arm, agg, grain


__all__ = ["arm_spec", "arm_slug", "cell_key", "parse_cell_key"]
