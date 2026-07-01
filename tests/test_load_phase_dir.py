"""Tests for load_phase_dir — read a phase's manifest.json from its directory."""

from __future__ import annotations

from pathlib import Path

import pytest

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
