"""PRD scenario 4: the gate report binds the engine's source hash; a changed
engine module, a failed report or a missing report refuses a benchmark run
naming the unmet value."""
import shutil
from pathlib import Path

import pytest

from tracecmibench import hashes
from tracecmibench.constants import ENGINE_MODULES
from tracecmibench.record import write_json


def test_engine_hash_changes_when_an_engine_module_changes(tmp_path):
    src = hashes.PACKAGE_DIR
    copy = tmp_path / "pkg"
    shutil.copytree(src, copy, ignore=shutil.ignore_patterns("__pycache__"))
    assert hashes.engine_sha256(copy) == hashes.engine_sha256()
    (copy / "staircase.py").write_text((copy / "staircase.py").read_text() + "\n# touched\n")
    assert hashes.engine_sha256(copy) != hashes.engine_sha256()
    # a non-engine module does not move the hash
    copy2 = tmp_path / "pkg2"
    shutil.copytree(src, copy2, ignore=shutil.ignore_patterns("__pycache__"))
    (copy2 / "report_or_other.py").write_text("x = 1\n")
    (copy2 / "pull.py").write_text((copy2 / "pull.py").read_text() + "\n# touched\n")
    assert hashes.engine_sha256(copy2) == hashes.engine_sha256()
    assert all((src / f"{m}.py").exists() for m in ENGINE_MODULES)


def test_require_gate_refusals(tmp_path):
    good = tmp_path / "good.json"
    write_json(good, {"passed": True, "engine_sha256": hashes.engine_sha256(), "assertions": []})
    assert hashes.require_gate(good)["passed"] is True
    with pytest.raises(hashes.GateRefusal, match="does not exist"):
        hashes.require_gate(tmp_path / "missing.json")
    failed = tmp_path / "failed.json"
    write_json(failed, {"passed": False, "engine_sha256": hashes.engine_sha256(),
                        "assertions": [{"name": "selected_threshold_replicates", "passed": False}]})
    with pytest.raises(hashes.GateRefusal, match="selected_threshold_replicates"):
        hashes.require_gate(failed)
    stale = tmp_path / "stale.json"
    write_json(stale, {"passed": True, "engine_sha256": "0" * 64, "assertions": []})
    with pytest.raises(hashes.GateRefusal, match="binds engine 0000"):
        hashes.require_gate(stale)
    # the current engine differs from the report once an engine module changes
    copy = tmp_path / "pkg"
    shutil.copytree(hashes.PACKAGE_DIR, copy, ignore=shutil.ignore_patterns("__pycache__"))
    (copy / "probe.py").write_text((copy / "probe.py").read_text() + "\n# edited\n")
    with pytest.raises(hashes.GateRefusal, match="current engine hashes to"):
        hashes.require_gate(good, package_dir=copy)
