"""Readers for run records and metrics CSV documents.

These are thin loaders that parse the on-disk file and return a Python
data structure shaped to make downstream code (viewer / classifier eval
/ notebook analysis) ergonomic. Schema validation is the contracts
package's job — call ``myocard_egm_contracts.validators.validate_*`` if
you want to assert conformance. The readers here happily return what
they got and trust that you ran the validator if it mattered.

Per-trace prediction outputs are NOT loaded here — they live inside the
ClassifierBank produced by the eval step. Use
:func:`~myocard_egm_data.banks.load_classifier_bank` for those.

Round-trip with the writers in this same package is covered by the
package's tests against the contracts' validators.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from myocard_egm_contracts.schema_info import field_type_map


def load_run_record(path: Path | str) -> dict[str, Any]:
    """Load and parse a run.json document.

    Returns the raw dict (matching the ``run_record`` schema). Use the
    contracts' Pydantic models if you want typed access; the dict shape
    is intentionally permissive for forward-compatible additions.
    """
    return _load_json(Path(path))


def load_metrics_csv(path: Path | str) -> list[dict[str, Any]]:
    """Load metrics.csv into a list of dicts, one per epoch.

    Numeric columns are converted from string to int/float; empty cells
    (the writer's encoding of non-finite or absent values) round-trip
    to ``None``. The ``epoch`` column is always int, ``lr`` /
    ``train_loss`` / ``val_loss`` / ``epoch_seconds`` always float, and
    the validation metrics nullable float.
    """
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(_coerce_metrics_row(row))
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metrics_float_columns() -> frozenset[str]:
    """Columns in metrics.schema.json declared as JSON "number" (nullable or not).

    Derived at import time from the schema rather than hardcoded — if the
    contracts add or remove a numeric metric column, this picks it up
    automatically. ``epoch`` is intentionally excluded: it's "integer",
    not "number", and is coerced separately to int.
    """
    types = field_type_map("metrics")
    return frozenset(prop for prop, t in types.items() if "number" in t and prop != "epoch")


_METRICS_FLOAT_COLUMNS = _metrics_float_columns()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, Mapping):
        raise ValueError(f"{path}: expected a JSON object at the top level.")
    return dict(doc)


def _coerce_metrics_row(row: dict[str, str]) -> dict[str, Any]:
    """Convert a CSV string row to the right numeric types.

    Empty cells become ``None`` (the schema's null) so downstream callers
    can distinguish "this column was not measured" from "this column was
    zero".
    """
    out: dict[str, Any] = {}
    for k, v in row.items():
        if k == "epoch":
            out[k] = int(v) if v != "" else None
        elif k in _METRICS_FLOAT_COLUMNS:
            out[k] = float(v) if v != "" else None
        else:
            out[k] = v
    return out
