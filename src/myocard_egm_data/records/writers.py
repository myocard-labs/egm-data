"""Writers for run records, metrics rows, hybrid eval, and model metadata.

Per-trace prediction outputs are NOT written here — they live inside
the producing ClassifierBank (see ``banks/classifier_bank.py``). The
standalone predictions schema was retired in egm-contracts v0.1.2.

Every writer stamps the schema version into the document it produces,
so a freshly-written file always carries a usable ``schema_version``
field. Non-finite floats are serialized as JSON ``null``
(``allow_nan=False``) and as empty CSV cells; consumers can rely on
strict-JSON readers.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from myocard_egm_contracts._generated.python.run_record import EpochRecord
from myocard_egm_contracts.schema_info import csv_column_order, current_version

# Schema versions and CSV column orders pulled from the contracts
# package via schema_info. We do NOT redeclare them here.
METRICS_CSV_COLUMNS: tuple[str, ...] = csv_column_order("metrics")


# ---------------------------------------------------------------------------
# EpochRecord helpers
# ---------------------------------------------------------------------------


def _epoch_csv_row(r: EpochRecord) -> dict[str, Any]:
    """Project a Pydantic EpochRecord into the metrics.csv row shape.

    Pulls val_metrics' well-known keys (auroc, accuracy, precision,
    recall, f1, ece) by name; anything else in val_metrics is dropped
    here since the CSV columns are fixed by the metrics schema.
    """
    m = r.val_metrics
    return {
        "epoch": r.epoch,
        "lr": r.lr,
        "train_loss": r.train_loss,
        "val_loss": r.val_loss,
        "val_auroc": m.get("auroc"),
        "val_accuracy": m.get("accuracy"),
        "val_precision": m.get("precision"),
        "val_recall": m.get("recall"),
        "val_f1": m.get("f1"),
        "val_ece": m.get("ece"),
        "epoch_seconds": r.epoch_seconds,
    }


def _epoch_to_json(r: EpochRecord) -> dict[str, Any]:
    """Serialize an EpochRecord to the run.json shape for one epoch."""
    return {
        "epoch": r.epoch,
        "lr": r.lr,
        "train_loss": r.train_loss,
        "val_loss": r.val_loss,
        "epoch_seconds": r.epoch_seconds,
        "val_metrics": dict(r.val_metrics),
        "val_reliability": [b.model_dump() for b in r.val_reliability],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _csv_cell(value: Any) -> Any:
    """Empty cell for non-finite numbers / None; pass through otherwise."""
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _json_safe(obj: Any) -> Any:
    """Recursively make a value strictly-JSON-safe.

    Converts Path -> str, tuple -> list, non-finite float -> None. Numpy
    scalars are converted via ``.item()`` if the input ever comes from
    numpy land (the trainer typically converts itself, but this is cheap
    insurance).
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return _json_safe(obj.item())
        except (ValueError, TypeError):
            return obj
    return obj


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _write_json(path: Path, doc: Mapping[str, Any]) -> Path:
    """Serialize ``doc`` to ``path`` as strict UTF-8 JSON.

    ``allow_nan=False`` ensures a downstream JSON parser cannot encounter
    NaN/Infinity tokens (which are not part of the spec). ``sort_keys=True``
    makes the byte stream deterministic across runs — the same logical
    record produces the same on-disk bytes regardless of dict insertion
    order, so content hashes / diffs / cache keys stay stable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(dict(doc)), f, indent=2, allow_nan=False, sort_keys=True)
    return path


# ---------------------------------------------------------------------------
# run.json
# ---------------------------------------------------------------------------


def best_epoch(records: Sequence[EpochRecord], metric: str) -> dict[str, Any]:
    """Best epoch by a maximized val metric, falling back to minimum val_loss."""
    best: dict[str, Any] | None = None
    for r in records:
        value = r.val_metrics.get(metric)
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            # Pydantic codegen types val_loss as nullable; treat missing
            # val_loss as +inf so it ranks last in the maximization.
            value = -(r.val_loss if r.val_loss is not None else float("inf"))
            used = "neg_val_loss"
        else:
            used = metric
        if best is None or value > best["_value"]:
            best = {"epoch": r.epoch, "metric": used, "value": value, "_value": value}
    if best is None:
        return {}
    best.pop("_value")
    return best


def build_run_record(
    *,
    config: Mapping[str, Any],
    run_meta: Mapping[str, Any],
    epoch_records: Sequence[EpochRecord],
    select_metric: str,
    test_loss: float | None = None,
    test_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the run.json document (not yet written)."""
    doc: dict[str, Any] = {
        "schema_version": current_version("run_record"),
        "created_utc": _utc_now(),
        "run": dict(run_meta),
        "config": dict(config),
        "epochs": [_epoch_to_json(r) for r in epoch_records],
        "best": best_epoch(epoch_records, select_metric),
    }
    if test_metrics is not None:
        doc["test"] = {
            "loss": test_loss,
            "metrics": {k: v for k, v in test_metrics.items() if k != "reliability"},
            "reliability": [dict(b) for b in test_metrics.get("reliability", [])],
        }
    return doc


def write_run_record(path: Path | str, doc: Mapping[str, Any]) -> Path:
    """Write a pre-built run.json document."""
    return _write_json(Path(path), doc)


# ---------------------------------------------------------------------------
# metrics.csv
# ---------------------------------------------------------------------------


def write_metrics_csv(path: Path | str, epoch_records: Sequence[EpochRecord]) -> Path:
    """Write the flat per-epoch metrics.csv with the schema's column order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(METRICS_CSV_COLUMNS))
        writer.writeheader()
        for r in epoch_records:
            writer.writerow({k: _csv_cell(v) for k, v in _epoch_csv_row(r).items()})
    return path


# ---------------------------------------------------------------------------
# hybrid_eval_metrics.json
# ---------------------------------------------------------------------------


def write_hybrid_eval_metrics(
    path: Path | str,
    *,
    producer: Mapping[str, Any],
    mixed: Mapping[str, Any],
    iafdb_only: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write a hybrid (synthetic+IAFDB) eval metrics document.

    The two metric blocks are required; ``extra`` is merged into the
    top-level document for any additional keys (e.g. tuned threshold
    sweeps) a future schema minor might add.

    Per-trace prediction outputs are NOT written here — they live inside
    the ClassifierBank produced by the eval step.
    """
    doc: dict[str, Any] = {
        "schema_version": current_version("hybrid_eval_metrics"),
        "created_utc": _utc_now(),
        "producer": dict(producer),
        "mixed": dict(mixed),
        "iafdb_only": dict(iafdb_only),
    }
    if extra:
        doc.update(dict(extra))
    return _write_json(Path(path), doc)


# ---------------------------------------------------------------------------
# model_metadata.json
# ---------------------------------------------------------------------------


def write_model_metadata(
    path: Path | str,
    *,
    model_artifact: Mapping[str, Any],
    input_spec: Mapping[str, Any],
    output_spec: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    decision: Mapping[str, Any],
    training_provenance: Mapping[str, Any],
) -> Path:
    """Write a model_metadata.json document (the inference contract).

    The six required blocks map 1:1 to the ``model_metadata`` schema. The
    caller assembles each block; this writer stamps the schema version
    and timestamps and serializes.
    """
    doc: dict[str, Any] = {
        "schema_version": current_version("model_metadata"),
        "created_utc": _utc_now(),
        "model_artifact": dict(model_artifact),
        "input": dict(input_spec),
        "output": dict(output_spec),
        "preprocessing": dict(preprocessing),
        "decision": dict(decision),
        "training_provenance": dict(training_provenance),
    }
    return _write_json(Path(path), doc)
