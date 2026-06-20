"""Build, write, and read ``run.json`` — the per-run ML training record.

The schema (``myocard_egm_contracts.training_run_record``) describes
the full per-run artifact: schema version, run metadata, the resolved
training configuration, a per-epoch :class:`EpochRecord` list (each
carrying its reliability bins), the ``best`` epoch
(:class:`BestEpoch`) by the configured selection metric, and an
optional held-out test block (:class:`HeldOutTest`).

Producers (currently only ``myocard-egm-classifier``'s trainer) build
a :class:`TrainingRunRecord` via :func:`build_training_run_record` and
write it with :func:`write_training_run_record`. Consumers (the
viewer, the paper-figure notebooks) load it with
:func:`load_training_run_record`, which returns a typed
:class:`TrainingRunRecord` instance — no dict-keyed access, no
schema-version unwrapping at the call site.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from myocard_egm_contracts._generated.python.training_run_record import (
    BestEpoch,
    EpochRecord,
    HeldOutTest,
    ReliabilityBin,
    TrainingRunRecord,
)
from myocard_egm_contracts._generated.python.training_run_record import (
    SchemaVersion as TrainingRunRecordSchemaVersion,
)
from myocard_egm_contracts.schema_info import current_version

from ._helpers import _load_pydantic_json, _sanitize_floats, _utc_now, _write_pydantic_json

__all__ = [
    "BestEpoch",
    "EpochRecord",
    "HeldOutTest",
    "ReliabilityBin",
    "TrainingRunRecord",
    "best_epoch",
    "build_training_run_record",
    "load_training_run_record",
    "write_training_run_record",
]


def build_training_run_record(
    *,
    config: Mapping[str, Any],
    run_meta: Mapping[str, Any],
    epoch_records: Sequence[EpochRecord],
    select_metric: str,
    test_loss: float | None = None,
    test_metrics: Mapping[str, Any] | None = None,
) -> TrainingRunRecord:
    """Assemble a :class:`TrainingRunRecord` with version + timestamp stamped.

    Computes the ``best`` epoch by the configured selection metric and
    folds in the optional held-out test block when provided. Non-finite
    floats in any metric dict are sanitized to ``None`` so the result
    is JSON-serializable downstream.

    Parameters
    ----------
    config
        The fully-resolved training configuration — typically nested
        ``model`` / ``data`` / ``training`` / ``eval`` blocks. Stored
        verbatim under ``TrainingRunRecord.config``.
    run_meta
        Run-level metadata: ``run_id``, ``git_sha``, ``host``,
        ``model_version``, ``training_started_utc`` /
        ``training_ended_utc``. Stored under ``TrainingRunRecord.run``.
    epoch_records
        Per-epoch :class:`EpochRecord` instances, in epoch order.
    select_metric
        Which scalar in ``val_metrics`` to maximize for the ``best``
        block (typically ``"auroc"``). See :func:`best_epoch` for the
        non-finite fallback rule.
    test_loss, test_metrics
        Held-out test results (only present when a test split was
        evaluated). When supplied, the reliability bins inside
        ``test_metrics`` are split out of the scalar dict and placed
        into the dedicated ``HeldOutTest.reliability`` slot.
    """
    test_block: HeldOutTest | None = None
    if test_metrics is not None:
        scalar = {k: v for k, v in test_metrics.items() if k != "reliability"}
        test_block = HeldOutTest(
            loss=_sanitize_floats(test_loss),
            metrics=_sanitize_floats(scalar),
            reliability=list(test_metrics.get("reliability", [])),
        )
    return TrainingRunRecord(
        schema_version=TrainingRunRecordSchemaVersion(current_version("training_run_record")),
        created_utc=_utc_now(),
        run=dict(run_meta),
        config=dict(config),
        epochs=list(epoch_records),
        best=best_epoch(epoch_records, select_metric),
        test=test_block,
    )


def write_training_run_record(path: Path | str, record: TrainingRunRecord) -> Path:
    """Write a :class:`TrainingRunRecord` to ``path`` as strict, deterministic JSON."""
    return _write_pydantic_json(path, record)


def load_training_run_record(path: Path | str) -> TrainingRunRecord:
    """Load a ``run.json`` document and return a typed :class:`TrainingRunRecord`.

    Raises ``pydantic.ValidationError`` on shape mismatch — callers
    don't need a separate ``validate_training_run_record`` round-trip;
    the Pydantic model IS the validator (the JSON-Schema-level
    validators in ``myocard-egm-contracts.validators`` add cross-cutting
    checks on top, but the typed read path already enforces the data
    shape).
    """
    return _load_pydantic_json(path, TrainingRunRecord)


def best_epoch(records: Sequence[EpochRecord], metric: str) -> BestEpoch:
    """Return the best epoch by a maximized validation metric.

    Walks ``records`` in order; for each epoch, looks up ``metric`` in
    ``EpochRecord.val_metrics``. Falls back to ``-val_loss`` (with the
    ``"neg_val_loss"`` tag in ``BestEpoch.metric``) when the metric is
    missing or non-finite on a given epoch. The maximum across all
    epochs wins.

    Returns an empty :class:`BestEpoch` (all fields ``None``) when
    ``records`` is empty.
    """
    best: dict[str, Any] | None = None
    for r in records:
        value = r.val_metrics.get(metric)
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            value = -(r.val_loss if r.val_loss is not None else float("inf"))
            used = "neg_val_loss"
        else:
            used = metric
        if best is None or value > best["_value"]:
            best = {"epoch": r.epoch, "metric": used, "value": value, "_value": value}
    if best is None:
        # All BestEpoch fields default to None; mypy doesn't see the
        # Field() default so we pass them explicitly.
        return BestEpoch(epoch=None, metric=None, value=None)
    return BestEpoch(epoch=best["epoch"], metric=best["metric"], value=best["value"])
