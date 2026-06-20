"""Write and read ``metrics.csv`` — flat per-epoch training metrics.

The schema (``myocard_egm_contracts.training_metrics``) describes ONE
row; the file is the concatenation of N such rows with a header.

Column order is the contract: consumers read by header name, but
spreadsheet defaults and plotting libraries assume the declared order.
Non-finite floats (e.g. AUROC undefined on a single-class val split)
are written as empty CSV cells and round-trip back to ``None`` on
read, which is distinct from "this metric was zero".

Unlike the JSON records, ``metrics.csv`` has no per-document
``schema_version`` field — the schema describes a row, not a file —
so this module has no ``build_*`` helper. Producers build
:class:`EpochRecord` instances (the run-time row source) and the
writer projects them into the CSV column order.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from myocard_egm_contracts._generated.python.training_metrics import TrainingMetricsRow
from myocard_egm_contracts._generated.python.training_run_record import EpochRecord
from myocard_egm_contracts.schema_info import csv_column_order

from ._helpers import _sanitize_floats

__all__ = [
    "TrainingMetricsRow",
    "load_training_metrics",
    "write_training_metrics",
]

# Column order pulled from the contracts package via schema_info so a
# future schema bump that reorders columns picks up automatically.
TRAINING_METRICS_CSV_COLUMNS: tuple[str, ...] = csv_column_order("training_metrics")


def write_training_metrics(path: Path | str, epoch_records: Sequence[EpochRecord]) -> Path:
    """Write per-epoch training metrics to a CSV in the schema's column order.

    Projects each :class:`EpochRecord` down to the
    :class:`TrainingMetricsRow` shape (the well-known scalar keys from
    ``val_metrics``), serializes via the schema's declared column
    order, and writes empty cells for non-finite or absent values.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(TRAINING_METRICS_CSV_COLUMNS))
        writer.writeheader()
        for r in epoch_records:
            row = _epoch_to_row(r)
            writer.writerow({k: _csv_cell(v) for k, v in row.model_dump().items()})
    return p


def load_training_metrics(path: Path | str) -> list[TrainingMetricsRow]:
    """Load ``metrics.csv`` and return a list of typed :class:`TrainingMetricsRow`.

    Numeric columns are coerced from string to int / float; empty cells
    (the writer's encoding of non-finite or absent values) round-trip
    to ``None`` so callers can distinguish "this metric was not
    measured" from "this metric was zero". Raises
    ``pydantic.ValidationError`` on rows that don't match the schema
    (e.g., missing required column).
    """
    p = Path(path)
    rows: list[TrainingMetricsRow] = []
    with p.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append(TrainingMetricsRow.model_validate(_coerce_row(raw)))
    return rows


# ---------------------------------------------------------------------------
# Helpers (schema-specific; not in _helpers.py because no other schema needs them)
# ---------------------------------------------------------------------------


def _epoch_to_row(r: EpochRecord) -> TrainingMetricsRow:
    """Project an :class:`EpochRecord` to a :class:`TrainingMetricsRow`.

    Pulls the well-known scalar keys from ``val_metrics`` by name;
    anything else in ``val_metrics`` is dropped here because the CSV
    columns are fixed by the schema (extra keys live in
    :class:`TrainingRunRecord` instead, which carries the full
    ``val_metrics`` dict per epoch).

    Non-finite floats (e.g. NaN AUROC on a single-class val split) are
    sanitized to ``None`` before constructing the
    :class:`TrainingMetricsRow` — the schema constrains the val_*
    fields to ``[0, 1]`` and nullable, so NaN must round-trip through
    ``None`` rather than be rejected at validation time.
    """
    # The training_metrics schema requires train_loss and val_loss to
    # be present and finite — "if a producer writes them as null/empty,
    # that's a producer bug worth surfacing" (schema docstring).
    # EpochRecord allows them to be None for pre-training-step or
    # otherwise-blank rows; here we promote that to an error rather
    # than silently writing an empty cell.
    if r.train_loss is None or r.val_loss is None:
        raise ValueError(
            f"EpochRecord(epoch={r.epoch}) has null train_loss / val_loss; "
            "training_metrics.csv requires both to be finite."
        )
    m = r.val_metrics
    return TrainingMetricsRow(
        epoch=r.epoch,
        lr=r.lr,
        train_loss=r.train_loss,
        val_loss=r.val_loss,
        val_auroc=_sanitize_floats(m.get("auroc")),
        val_accuracy=_sanitize_floats(m.get("accuracy")),
        val_precision=_sanitize_floats(m.get("precision")),
        val_recall=_sanitize_floats(m.get("recall")),
        val_f1=_sanitize_floats(m.get("f1")),
        val_ece=_sanitize_floats(m.get("ece")),
        epoch_seconds=r.epoch_seconds,
    )


def _csv_cell(value: Any) -> Any:
    """Empty cell for non-finite numbers or ``None``; pass through otherwise."""
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


# Columns the schema declares as nullable float; ``epoch`` is integer
# (handled separately); the always-finite cols (lr/train_loss/val_loss/
# epoch_seconds) are also float but Pydantic re-validates them so we
# treat them uniformly here.
_FLOAT_COLUMNS = frozenset(TRAINING_METRICS_CSV_COLUMNS) - {"epoch"}


def _coerce_row(raw: dict[str, str]) -> dict[str, Any]:
    """Convert a CSV string row to numeric types (empty cell -> ``None``)."""
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k == "epoch":
            out[k] = int(v) if v != "" else None
        elif k in _FLOAT_COLUMNS:
            out[k] = float(v) if v != "" else None
        else:
            out[k] = v
    return out
