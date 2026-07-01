"""Read + write the per-phase ``manifest.json`` (egm-contracts phase_manifest).

The phase manifest is the shallow index of every artifact in one project
phase. egm-studio is its canonical curator; this module is the typed I/O
the curator (and ``intracardiac-platform/scripts/validate_manifest.py``)
read and write it through. Banks are split into ``egm_banks``
(training / pretraining / prediction) and ``noise_banks`` per the
cross-artifact-linkage design.

Like the ``records/`` modules, every ``load_*`` returns a typed
:class:`PhaseManifest` (the Pydantic model IS the validator) and every
``write_*`` takes one and emits strict, deterministic JSON.
"""

from __future__ import annotations

from pathlib import Path

from myocard_egm_contracts._generated.python.phase_manifest import (
    EgmBankEntry,
    FigureEntry,
    ModelEntry,
    NoiseBankEntry,
    ObservationEntry,
    PaperEntry,
    PhaseManifest,
    TrainingRunEntry,
    UsageTag,
)

# Generic Pydantic <-> strict-JSON plumbing, shared with the records/ modules.
from .._serialization import _load_pydantic_json, _write_pydantic_json

#: Conventional filename of a phase's manifest inside its phase folder
#: (``intracardiac-platform/project/phases/phase_X/manifest.json``).
MANIFEST_FILENAME = "manifest.json"

__all__ = [
    "MANIFEST_FILENAME",
    "EgmBankEntry",
    "FigureEntry",
    "ModelEntry",
    "NoiseBankEntry",
    "ObservationEntry",
    "PaperEntry",
    "PhaseManifest",
    "TrainingRunEntry",
    "UsageTag",
    "load_phase_dir",
    "load_phase_manifest",
    "write_phase_manifest",
]


def load_phase_manifest(path: Path | str) -> PhaseManifest:
    """Load a ``manifest.json`` document into a typed :class:`PhaseManifest`.

    Raises ``pydantic.ValidationError`` on shape mismatch (bad id pattern,
    unknown key, missing required entry field, ...).
    """
    return _load_pydantic_json(path, PhaseManifest)


def load_phase_dir(phase_dir: Path | str) -> PhaseManifest:
    """Load a phase's manifest from its phase DIRECTORY.

    A phase lives at ``intracardiac-platform/project/phases/phase_X/`` with its
    shallow index at ``manifest.json``; pass the phase directory and this reads
    ``<phase_dir>/manifest.json`` through :func:`load_phase_manifest`. When you
    already hold the manifest file path, call :func:`load_phase_manifest`
    directly. Propagates that function's errors: a ``pydantic.ValidationError``
    on a malformed manifest, ``FileNotFoundError`` when the file is absent.
    """
    return load_phase_manifest(Path(phase_dir) / MANIFEST_FILENAME)


def write_phase_manifest(path: Path | str, manifest: PhaseManifest) -> Path:
    """Write a :class:`PhaseManifest` to ``path`` as strict, deterministic JSON.

    Optional fields left at their default (empty artifact sections,
    unset ``usage_tag`` / ``download_url`` / ``usage_notes``) are omitted
    rather than written as ``null``, matching the JSON Schema.
    """
    return _write_pydantic_json(path, manifest)
