"""Record I/O — typed JSON / CSV training and evaluation artifacts.

The package owns the on-disk format for everything that is *not* a
bank: the per-run ML training record (``run.json``), the per-epoch
metrics table (``metrics.csv``), the hybrid eval summary
(``hybrid_eval_metrics.json``), the EGM-classifier inference-side
model metadata (``model_metadata.json``), and the noise-bank
provenance sidecar (``noise_bank_run_record.json``).

Layout
------
Each schema gets its own per-file module that owns build + write +
load for that schema only, mirroring the per-schema layout in
``myocard-egm-contracts._generated.python.*``:

- :mod:`.training_run_record` — ``build_training_run_record``,
  ``write_training_run_record``, ``load_training_run_record``,
  ``best_epoch``; re-exports :class:`TrainingRunRecord` and its
  nested models from contracts.
- :mod:`.training_metrics` — ``write_training_metrics``,
  ``load_training_metrics``; re-exports
  :class:`TrainingMetricsRow`. No ``build_*`` helper: the schema
  describes one row, the file is N rows, no per-document
  ``schema_version`` field.
- :mod:`.hybrid_eval_metrics` — ``build_hybrid_eval_metrics``,
  ``write_hybrid_eval_metrics``, ``load_hybrid_eval_metrics``;
  re-exports :class:`HybridEvalMetrics` and its nested models.
- :mod:`.egm_class_model_metadata` —
  ``build_egm_class_model_metadata``,
  ``write_egm_class_model_metadata``,
  ``load_egm_class_model_metadata``; re-exports
  :class:`EgmClassModelMetadata` and its nested models. Specific to
  the 1-D EGM-classifier family (future 2-D / 3-D electrode model
  topologies will get their own per-schema modules).
- :mod:`.noise_bank_run_record` — ``build_noise_bank_run_record``,
  ``write_noise_bank_run_record``, ``load_noise_bank_run_record``;
  re-exports :class:`NoiseBankRunRecord` and its nested models.

Shared low-level plumbing lives in :mod:`._helpers` (NaN
sanitization, strict-JSON writer, typed reader). It is strictly
internal — not re-exported here.

Typing
------
Every ``build_*`` returns a Pydantic model from
``myocard-egm-contracts``; every ``write_*`` takes a model;
every ``load_*`` returns a model. Dict-shaped inputs to the build
helpers are validated into nested sub-models by Pydantic at
construction time, so producers can hand-build a block as a mapping
and let the helper enforce the shape.

Schema validation is the Pydantic model — call
``myocard_egm_contracts.validators.*`` if you want the additional
JSON-Schema-level cross-cutting checks (the typed read path already
enforces the data shape).
"""

from __future__ import annotations

from .egm_class_model_metadata import (
    Decision,
    EgmClassModelMetadata,
    Input,
    ModelArtifact,
    Output,
    Preprocessing,
    build_egm_class_model_metadata,
    load_egm_class_model_metadata,
    write_egm_class_model_metadata,
)
from .hybrid_eval_metrics import (
    HybridEvalMetrics,
    IafdbOnly,
    Mixed,
    Producer,
    build_hybrid_eval_metrics,
    load_hybrid_eval_metrics,
    write_hybrid_eval_metrics,
)
from .noise_bank_run_record import (
    Calibration,
    NoiseBankRunRecord,
    PerTraceProvenance,
    Selection,
    ThresholdMode,
    Windowing,
    build_noise_bank_run_record,
    load_noise_bank_run_record,
    write_noise_bank_run_record,
)
from .training_metrics import (
    TrainingMetricsRow,
    load_training_metrics,
    write_training_metrics,
)
from .training_run_record import (
    BestEpoch,
    EpochRecord,
    HeldOutTest,
    ReliabilityBin,
    TrainingRunRecord,
    best_epoch,
    build_training_run_record,
    load_training_run_record,
    write_training_run_record,
)

__all__ = [
    "BestEpoch",
    "Calibration",
    "Decision",
    "EgmClassModelMetadata",
    "EpochRecord",
    "HeldOutTest",
    "HybridEvalMetrics",
    "IafdbOnly",
    "Input",
    "Mixed",
    "ModelArtifact",
    "NoiseBankRunRecord",
    "Output",
    "PerTraceProvenance",
    "Preprocessing",
    "Producer",
    "ReliabilityBin",
    "Selection",
    "ThresholdMode",
    "TrainingMetricsRow",
    "TrainingRunRecord",
    "Windowing",
    "best_epoch",
    "build_egm_class_model_metadata",
    "build_hybrid_eval_metrics",
    "build_noise_bank_run_record",
    "build_training_run_record",
    "load_egm_class_model_metadata",
    "load_hybrid_eval_metrics",
    "load_noise_bank_run_record",
    "load_training_metrics",
    "load_training_run_record",
    "write_egm_class_model_metadata",
    "write_hybrid_eval_metrics",
    "write_noise_bank_run_record",
    "write_training_metrics",
    "write_training_run_record",
]
