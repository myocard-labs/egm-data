"""Build, write, and read ``hybrid_eval_metrics.json``.

The schema (``myocard_egm_contracts.hybrid_eval_metrics``) describes
the aggregate metrics for a hybrid (mixed synthetic + IAFDB)
evaluation run. ``mixed`` carries scalar metrics over the combined
set; ``iafdb_only`` carries the IAFDB-side diagnostic that exposes
calibration saturation (a model that calls every IAFDB row "fibrotic"
will look fine on AUROC across the mixed set but show 1.0 FPR here).

Per-trace prediction outputs are NOT written here — they live inside
the producing :class:`ClassifierBank` (one ``ClassifierPrediction``
per trace). The standalone predictions document was retired in
egm-contracts v0.1.2.

"Hybrid" here follows the cardiac-ML convention established by
Sánchez et al. 2021 — mixed in silico + in vivo data in a single
labeled dataset. The legacy "noise-mixed synthetic" usage of the
word is being phased out across the polyrepo.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from myocard_egm_contracts._generated.python.hybrid_eval_metrics import (
    HybridEvalMetrics,
    IafdbOnly,
    Mixed,
    Producer,
)
from myocard_egm_contracts._generated.python.hybrid_eval_metrics import (
    SchemaVersion as HybridEvalMetricsSchemaVersion,
)
from myocard_egm_contracts.schema_info import current_version

from ._helpers import _load_pydantic_json, _utc_now, _write_pydantic_json

__all__ = [
    "HybridEvalMetrics",
    "IafdbOnly",
    "Mixed",
    "Producer",
    "build_hybrid_eval_metrics",
    "load_hybrid_eval_metrics",
    "write_hybrid_eval_metrics",
]


def build_hybrid_eval_metrics(
    *,
    producer: Producer | Mapping[str, Any],
    mixed: Mixed | Mapping[str, Any],
    iafdb_only: IafdbOnly | Mapping[str, Any],
) -> HybridEvalMetrics:
    """Assemble a :class:`HybridEvalMetrics` with version + timestamp stamped.

    Each of ``producer`` / ``mixed`` / ``iafdb_only`` may be passed as
    a typed Pydantic instance OR as a plain mapping — Pydantic
    validates dicts into the nested sub-models at construction time.
    Dicts are preferred at the call site only when the producer is
    assembling the values inline; library code that already has typed
    instances should pass them through.
    """
    return HybridEvalMetrics(
        schema_version=HybridEvalMetricsSchemaVersion(
            current_version("hybrid_eval_metrics"),
        ),
        created_utc=_utc_now(),
        # ``model_validate`` accepts either a dict or an existing
        # instance and returns the typed sub-model — gives mypy the
        # concrete type it wants while preserving the dict-ergonomic
        # API at the call site.
        producer=Producer.model_validate(producer),
        mixed=Mixed.model_validate(mixed),
        iafdb_only=IafdbOnly.model_validate(iafdb_only),
    )


def write_hybrid_eval_metrics(path: Path | str, record: HybridEvalMetrics) -> Path:
    """Write a :class:`HybridEvalMetrics` to ``path`` as strict JSON."""
    return _write_pydantic_json(path, record)


def load_hybrid_eval_metrics(path: Path | str) -> HybridEvalMetrics:
    """Load a ``hybrid_eval_metrics.json`` document into a typed model.

    Raises ``pydantic.ValidationError`` on shape mismatch.
    """
    return _load_pydantic_json(path, HybridEvalMetrics)
