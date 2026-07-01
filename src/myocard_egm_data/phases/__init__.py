"""Phase-artifact I/O — the per-phase manifest, observations, and figure specs.

These are the cross-artifact-linkage formats (egm-contracts v0.5.0): JSON
documents that organize one project phase's artifacts. egm-studio is the
canonical curator; ``intracardiac-platform/scripts/validate_manifest.py`` is
the scan-and-validate safety net. This subpackage is the typed I/O both read
and write them through.

Layout (mirrors ``records/``): one per-schema module owning load + write for
that schema, returning / accepting the typed egm-contracts Pydantic model.

- :mod:`.phase_manifest` — ``load_phase_manifest`` / ``write_phase_manifest``;
  re-exports :class:`PhaseManifest` + the entry models (``EgmBankEntry``,
  ``NoiseBankEntry``, ``TrainingRunEntry``, ``ModelEntry``,
  ``ObservationEntry``, ``FigureEntry``, ``PaperEntry``) and the ``UsageTag``
  vocabulary.
- :mod:`.observation` — ``load_observation`` / ``write_observation``;
  re-exports :class:`Observation` + ``TraceRef`` / ``ViewState`` /
  ``References``.
- :mod:`.figure_spec` — ``load_figure_spec`` / ``write_figure_spec``;
  re-exports :class:`FigureSpec` + ``Inputs`` / ``Group`` / ``Output``.

Every ``load_*`` returns a typed model (the Pydantic model IS the validator);
every ``write_*`` emits strict, deterministic JSON with optional-defaulted
fields omitted.
"""

from __future__ import annotations

from .figure_spec import (
    FigureSpec,
    Group,
    Inputs,
    Output,
    load_figure_spec,
    write_figure_spec,
)
from .observation import (
    Observation,
    References,
    TraceRef,
    ViewState,
    load_observation,
    write_observation,
)
from .phase_manifest import (
    MANIFEST_FILENAME,
    EgmBankEntry,
    FigureEntry,
    ModelEntry,
    NoiseBankEntry,
    ObservationEntry,
    PaperEntry,
    PhaseManifest,
    TrainingRunEntry,
    UsageTag,
    load_phase_dir,
    load_phase_manifest,
    write_phase_manifest,
)

__all__ = [
    "MANIFEST_FILENAME",
    "EgmBankEntry",
    "FigureEntry",
    "FigureSpec",
    "Group",
    "Inputs",
    "ModelEntry",
    "NoiseBankEntry",
    "Observation",
    "ObservationEntry",
    "Output",
    "PaperEntry",
    "PhaseManifest",
    "References",
    "TraceRef",
    "TrainingRunEntry",
    "UsageTag",
    "ViewState",
    "load_figure_spec",
    "load_observation",
    "load_phase_dir",
    "load_phase_manifest",
    "write_figure_spec",
    "write_observation",
    "write_phase_manifest",
]
