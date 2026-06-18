"""Record I/O — the JSON / CSV training and evaluation artifacts.

The package owns readers and writers for everything that is *not* a bank:

- ``run.json`` (``run_record`` schema): the full-fidelity, versioned
  per-run record (run metadata + config + epoch-by-epoch records + best
  epoch summary + optional held-out test block).
- ``metrics.csv`` (``metrics`` schema describes one row): the flat
  per-epoch metrics table, trivially loadable into pandas / a plotting
  lib / a spreadsheet.
- ``hybrid_eval_metrics.json`` (``hybrid_eval_metrics`` schema): the
  hybrid (synthetic + IAFDB) eval summary. Per-trace prediction outputs
  live inside the producing ClassifierBank, not in a separate record.
- ``model_metadata.json`` (``model_metadata`` schema): the inference
  contract for a deployed model artifact.

Schema versions live in ``myocard-egm-contracts``; this package writes
them into the right field on every artifact via ``schema_info``. The
writers do not enforce the schema — that is what the contracts'
file-level validators are for — but the round-trip tests in this repo
hold the line.
"""

from __future__ import annotations

# Re-export the run_record schema's EpochRecord Pydantic model from
# myocard-egm-contracts so consumers can `from myocard_egm_data.records
# import EpochRecord` without learning the egm-contracts subpackage
# layout. There is no separate egm-data dataclass here — the Pydantic
# model is the single source of truth.
from myocard_egm_contracts._generated.python.run_record import EpochRecord

from .noise_bank_run_record import (
    build_noise_bank_run_record,
    load_noise_bank_run_record,
    write_noise_bank_run_record,
)
from .readers import (
    load_metrics_csv,
    load_run_record,
)
from .writers import (
    write_hybrid_eval_metrics,
    write_metrics_csv,
    write_model_metadata,
    write_run_record,
)

__all__ = [
    "EpochRecord",
    "build_noise_bank_run_record",
    "load_metrics_csv",
    "load_noise_bank_run_record",
    "load_run_record",
    "write_hybrid_eval_metrics",
    "write_metrics_csv",
    "write_model_metadata",
    "write_noise_bank_run_record",
    "write_run_record",
]
