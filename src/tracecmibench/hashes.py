"""The engine hash and the gate requirement (PRD scenario 4, D11).

`engine_sha256()` hashes the sources of the probing modules — the code that
turns a frozen model into a score matrix and a prediction. A committed gate
report binds one such hash; `require_gate` refuses a benchmark run whose
report did not pass or whose hash is not the current engine's, naming the
unmet value, so nothing scores a benchmark corpus with an engine the
reproduction gate never saw.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .constants import ENGINE_MODULES
from .record import read_json

PACKAGE_DIR = Path(__file__).resolve().parent


class GateRefusal(RuntimeError):
    pass


def engine_sha256(package_dir=PACKAGE_DIR):
    h = hashlib.sha256()
    for name in ENGINE_MODULES:
        p = Path(package_dir) / f"{name}.py"
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def require_gate(report_path, package_dir=PACKAGE_DIR):
    """Load a gate report and refuse unless it passed at the current engine hash."""
    path = Path(report_path)
    if not path.exists():
        raise GateRefusal(f"gate report {path} does not exist; run selfcheck first (PRD scenario 4)")
    report = read_json(path)
    if report.get("passed") is not True:
        failed = [a["name"] for a in report.get("assertions", []) if not a.get("passed")]
        raise GateRefusal(f"gate report {path.name} did not pass (failed assertions: {failed or 'unknown'}); "
                          f"no benchmark corpus may be scored (PRD scenario 4)")
    current = engine_sha256(package_dir)
    if report.get("engine_sha256") != current:
        raise GateRefusal(f"gate report {path.name} binds engine {report.get('engine_sha256')} but the current engine "
                          f"hashes to {current}; re-run selfcheck for this engine (PRD scenario 4)")
    return report
