"""myocard-egm-data — pure I/O for the intracardiac-EGM stack.

This package owns:

- **Bank readers/writers** (``banks.readers`` / ``banks.writers``) for the
  two HDF5 bank schemas defined by ``myocard-egm-contracts``: the synthetic
  bank (clean or noise-mixed traces) and the IAFDB healthy-segment
  bank.
- **Record readers/writers** (``records.readers`` / ``records.writers``) for
  the JSON / CSV training and evaluation artifacts: ``run.json``,
  ``metrics.csv``, ``predictions_<split>.{json,csv}``, and
  ``model_metadata.json``.
- **Phase-artifact readers/writers** (``phases``) for the cross-artifact-
  linkage JSON formats: the per-phase ``manifest.json``, observations, and
  figure specs (egm-contracts v0.5.0).

Schema versioning lives in ``myocard-egm-contracts``. This package is
deliberately a thin I/O layer over those schemas — the formats are the
truth, the readers/writers just decode/encode them, and every writer is
covered by a round-trip test that runs the contracts' file-level
validators.

This is a pure I/O package: it does not import torch. The torch-based
training-data layer (``Dataset`` wrappers, patient-aware split, per-trace
augmentation) lives in its sole consumer, ``myocard-egm-classifier``.
"""

from __future__ import annotations

from importlib import metadata

try:
    __version__ = metadata.version("myocard-egm-data")
except metadata.PackageNotFoundError:  # pragma: no cover — editable install w/o metadata
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
