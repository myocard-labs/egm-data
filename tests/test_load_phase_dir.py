"""Tests for phase-manifest I/O — load_phase_dir, plus the provenance-optional
round trip phase_manifest gained in egm-contracts v0.6.0 (B19)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from myocard_egm_contracts.validators import validate_phase_manifest

from myocard_egm_data.phases import (
    MANIFEST_FILENAME,
    PhaseManifest,
    load_phase_dir,
    load_phase_manifest,
    write_phase_manifest,
)

_MANIFEST = {
    "schema_version": "1",
    "phase": 1.5,
    "status": "in_progress",
    "egm_banks": [
        {
            "id": "tbank_synthetic_v1_5_2026-06-25",
            "path": "banks/tbank.h5",
            "produced_by_package": "synthetic-egm-pipeline",
            "produced_by_version": "v0.3.0",
        }
    ],
}


def _write_phase(phase_dir: Path) -> None:
    phase_dir.mkdir(parents=True, exist_ok=True)
    write_phase_manifest(phase_dir / MANIFEST_FILENAME, PhaseManifest.model_validate(_MANIFEST))


def test_load_phase_dir_reads_manifest(tmp_path: Path) -> None:
    phase_dir = tmp_path / "phase_1_5"
    _write_phase(phase_dir)
    manifest = load_phase_dir(phase_dir)
    assert manifest.phase == 1.5
    assert manifest.egm_banks is not None
    assert len(manifest.egm_banks) == 1


def test_load_phase_dir_matches_direct_manifest_load(tmp_path: Path) -> None:
    phase_dir = tmp_path / "phase_1_5"
    _write_phase(phase_dir)
    from_dir = load_phase_dir(phase_dir)
    from_file = load_phase_manifest(phase_dir / MANIFEST_FILENAME)
    assert from_dir.model_dump() == from_file.model_dump()


def test_load_phase_dir_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_phase_dir(tmp_path / "phase_does_not_exist")


# ---------------------------------------------------------------------------
# B19 — produced_by_* optional (phase_manifest, egm-contracts v0.6.0)
# ---------------------------------------------------------------------------

_MANIFEST_NO_PROVENANCE = {
    "schema_version": "1",
    "phase": 1.5,
    "status": "in_progress",
    "egm_banks": [
        # A hand-added / externally-produced artifact: the curator knows
        # the id and the path but genuinely does not know what produced it.
        {"id": "tbank_handmade_v1_5_2026-07-31", "path": "banks/tbank.h5"}
    ],
}


def test_manifest_entry_without_provenance_round_trips(tmp_path: Path) -> None:
    """An entry may omit produced_by_package / produced_by_version entirely.

    Before v0.6.0 both were required, so egm-studio's curator stamped
    ``"unknown"`` / ``"0"`` sentinels on artifacts it had not produced —
    strings that read like real provenance to anything downstream. B19
    made the pair optional so absence can be recorded as absence.
    """
    path = tmp_path / MANIFEST_FILENAME
    write_phase_manifest(path, PhaseManifest.model_validate(_MANIFEST_NO_PROVENANCE))

    reloaded = load_phase_manifest(path)
    assert reloaded.egm_banks is not None
    entry = reloaded.egm_banks[0]
    assert entry.id == "tbank_handmade_v1_5_2026-07-31"
    assert entry.produced_by_package is None
    assert entry.produced_by_version is None
    # The file the curator wrote is still schema-valid.
    result = validate_phase_manifest(path)
    assert result.ok, result.issues


def test_absent_provenance_is_omitted_not_null(tmp_path: Path) -> None:
    """Absent optional fields are omitted from the JSON, never written as null.

    The schema types these as plain strings, not ``["string", "null"]``,
    so an explicit ``null`` would fail validation on the next read. The
    writer's ``exclude_defaults=True`` is what keeps them out; this test
    pins that behavior, because it is a property of the serializer rather
    than of the schema and would break silently if the writer ever
    switched to ``exclude_none`` or dropped the flag.
    """
    path = tmp_path / MANIFEST_FILENAME
    write_phase_manifest(path, PhaseManifest.model_validate(_MANIFEST_NO_PROVENANCE))

    raw = json.loads(path.read_text(encoding="utf-8"))
    entry = raw["egm_banks"][0]
    assert "produced_by_package" not in entry
    assert "produced_by_version" not in entry
    assert set(entry) == {"id", "path"}


def test_partial_provenance_is_preserved(tmp_path: Path) -> None:
    """The two fields are independently optional.

    A curator that knows the producing package but not its version must
    be able to record exactly that, rather than being forced into a
    sentinel for the half it does not know.
    """
    doc = json.loads(json.dumps(_MANIFEST_NO_PROVENANCE))
    doc["egm_banks"][0]["produced_by_package"] = "synthetic-egm-pipeline"

    path = tmp_path / MANIFEST_FILENAME
    write_phase_manifest(path, PhaseManifest.model_validate(doc))

    banks = load_phase_manifest(path).egm_banks
    assert banks is not None
    entry = banks[0]
    assert entry.produced_by_package == "synthetic-egm-pipeline"
    assert entry.produced_by_version is None
    assert validate_phase_manifest(path).ok
