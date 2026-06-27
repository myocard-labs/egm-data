"""myocard-egm-data — I/O and dataset ergonomics for the intracardiac-EGM stack.

This package owns:

- **Bank readers/writers** (``banks.readers`` / ``banks.writers``) for the
  two HDF5 bank schemas defined by ``myocard-egm-contracts``: the synthetic
  bank (clean or hybrid clean+noise traces) and the IAFDB healthy-segment
  bank.
- **Record readers/writers** (``records.readers`` / ``records.writers``) for
  the JSON / CSV training and evaluation artifacts: ``run.json``,
  ``metrics.csv``, ``predictions_<split>.{json,csv}``, and
  ``model_metadata.json``.
- **PyTorch ``Dataset`` wrappers** (``datasets``), including the per-trace
  normalize + pad + augment transform and the patient-aware split.
- **Phase-artifact readers/writers** (``phases``) for the cross-artifact-
  linkage JSON formats: the per-phase ``manifest.json``, observations, and
  figure specs (egm-contracts v0.5.0).

Schema versioning lives in ``myocard-egm-contracts``. This package is
deliberately a thin I/O layer over those schemas — the formats are the
truth, the readers/writers just decode/encode them, and every writer is
covered by a round-trip test that runs the contracts' file-level
validators.

Torch is an optional dependency. Bank and record I/O does not import
torch; the ``datasets`` subpackage does. Install ``myocard-egm-data[torch]``
for the dataset wrappers.
"""

from __future__ import annotations

from importlib import metadata

try:
    __version__ = metadata.version("myocard-egm-data")
except metadata.PackageNotFoundError:  # pragma: no cover — editable install w/o metadata
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
