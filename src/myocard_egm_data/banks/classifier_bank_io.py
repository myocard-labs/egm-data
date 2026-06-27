"""HDF5 reader and writer for :class:`ClassifierBank`.

On-disk layout (informally; the canonical doc is
``project/classifier_bank_format.md``):

::

    /                                ← root
    ├── attrs:
    │   ├── schema_version: "0.2"
    │   ├── id: stable artifact id (optional; absent on 0.1 banks)
    │   ├── created_utc: ISO timestamp
    │   └── labels_json: '{"0": "healthy", "1": "fibrotic"}'
    ├── banks/                       ← group, one row per source bank
    │   └── datasets:
    │       ├── bank_id            vlen-utf8[B]  (stable ArtifactId)
    │       ├── bank_type          vlen-utf8[B]
    │       ├── bank_path          vlen-utf8[B]
    │       └── bank_metadata_json vlen-utf8[B]
    └── traces/                      ← group, one row per trace
        └── datasets:
            ├── bank_id              vlen-utf8[N]  (stable ArtifactId ref)
            ├── signal               float32[N, T]
            ├── freq_hz              float64[N]
            ├── amp_type             vlen-utf8[N]
            ├── split                vlen-utf8[N] ('' for None)
            ├── has_label_truth      uint8[N]  (sentinel mask)
            ├── label_truth          int64[N]
            ├── has_prediction       uint8[N]  (sentinel mask)
            ├── pred_label_pred      int64[N]
            ├── pred_label_prob      float64[N]
            ├── pred_logits_json     vlen-utf8[N]
            └── trace_metadata_json  vlen-utf8[N]

Phase-1 constraint: all traces in one ClassifierBank share the signal
length T. The writer enforces this; if a future version supports
variable-length signals it will introduce an HDF5 vlen dtype.

Nullable fields use a sentinel-mask convention rather than an HDF5
NaN encoding so we don't conflate "missing" with "zero" — e.g. a
``label_truth`` of 0 is a legitimate "healthy" label and must be
distinguishable from "no ground truth".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .classifier_bank import (
    SUPPORTED_CLASSIFIER_BANK_VERSIONS,
    ClassifierBank,
    ClassifierBankMetaData,
    ClassifierPrediction,
    ClassifierTrace,
)

__all__ = [
    "load_classifier_bank",
    "write_classifier_bank",
]

_STR_DTYPE = h5py.string_dtype(encoding="utf-8")


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_classifier_bank(
    bank: ClassifierBank,
    path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """Serialize a ClassifierBank to HDF5 at ``path``."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True.")
    path.parent.mkdir(parents=True, exist_ok=True)

    # Validate uniform trace length up front.
    if bank.traces:
        T = bank.traces[0].signal.shape[0]
        for i, t in enumerate(bank.traces):
            if t.signal.shape[0] != T:
                raise ValueError(
                    f"Trace {i} length {t.signal.shape[0]} != bank "
                    f"length {T}. All traces in one ClassifierBank must "
                    "share the signal length."
                )
    else:
        T = 0

    with h5py.File(path, "w") as f:
        _write_root_attrs(f, bank)
        _write_banks_group(f, bank.banks)
        _write_traces_group(f, bank.traces, T)
    return path


def _write_root_attrs(f: h5py.File, bank: ClassifierBank) -> None:
    f.attrs["schema_version"] = bank.schema_version
    f.attrs["created_utc"] = bank.created_utc
    # Stable artifact id is optional: written only when this bank is a
    # tracked artifact (e.g. a saved predictions bank). Intermediate banks
    # leave it None, and the attr is simply absent.
    if bank.id is not None:
        f.attrs["id"] = bank.id
    # labels keys are ints but JSON requires string keys; stringify.
    f.attrs["labels_json"] = json.dumps(
        {str(k): str(v) for k, v in bank.labels.items()}, sort_keys=True
    )


def _write_banks_group(f: h5py.File, banks: list[ClassifierBankMetaData]) -> None:
    g = f.create_group("banks")
    b = len(banks)
    g.create_dataset(
        "bank_id",
        data=np.asarray([m.bank_id for m in banks], dtype=object),
        dtype=_STR_DTYPE,
    )
    g.create_dataset(
        "bank_type",
        data=np.asarray([m.bank_type for m in banks], dtype=object),
        dtype=_STR_DTYPE,
    )
    g.create_dataset(
        "bank_path",
        data=np.asarray([m.bank_path for m in banks], dtype=object),
        dtype=_STR_DTYPE,
    )
    g.create_dataset(
        "bank_metadata_json",
        data=np.asarray(
            [json.dumps(m.bank_metadata, sort_keys=True, default=str) for m in banks],
            dtype=object,
        ),
        dtype=_STR_DTYPE,
    )
    g.attrs["n_banks"] = b


def _write_traces_group(f: h5py.File, traces: list[ClassifierTrace], T: int) -> None:
    g = f.create_group("traces")
    n = len(traces)

    # Signal block.
    if n > 0:
        signal_arr = np.empty((n, T), dtype=np.float32)
        for i, t in enumerate(traces):
            signal_arr[i] = t.signal.astype(np.float32, copy=False)
    else:
        signal_arr = np.empty((0, 0), dtype=np.float32)
    g.create_dataset("signal", data=signal_arr, dtype=np.float32)

    # Scalar / string per-trace columns.
    g.create_dataset(
        "bank_id",
        data=np.asarray([t.bank_id for t in traces], dtype=object),
        dtype=_STR_DTYPE,
    )
    g.create_dataset("freq_hz", data=np.asarray([t.freq_hz for t in traces], dtype=np.float64))
    g.create_dataset(
        "amp_type",
        data=np.asarray([t.amp_type for t in traces], dtype=object),
        dtype=_STR_DTYPE,
    )
    g.create_dataset(
        "split",
        data=np.asarray([t.split if t.split is not None else "" for t in traces], dtype=object),
        dtype=_STR_DTYPE,
    )

    # label_truth, with a sentinel mask so 0 stays distinct from missing.
    has_label = np.asarray([1 if t.label_truth is not None else 0 for t in traces], dtype=np.uint8)
    label_vals = np.asarray(
        [int(t.label_truth) if t.label_truth is not None else 0 for t in traces],
        dtype=np.int64,
    )
    g.create_dataset("has_label_truth", data=has_label)
    g.create_dataset("label_truth", data=label_vals)

    # Predictions, same sentinel pattern. We pack label_pred / label_prob /
    # logits as parallel columns; logits are stored as a per-row JSON
    # string so that a future multi-class extension doesn't need a schema
    # change here.
    has_pred = np.asarray([1 if t.prediction is not None else 0 for t in traces], dtype=np.uint8)
    pred_label = np.asarray(
        [t.prediction.label_pred if t.prediction is not None else 0 for t in traces],
        dtype=np.int64,
    )
    pred_prob = np.asarray(
        [t.prediction.label_prob if t.prediction is not None else 0.0 for t in traces],
        dtype=np.float64,
    )
    pred_logits_json = np.asarray(
        [
            json.dumps(
                {str(k): float(v) for k, v in (t.prediction.pred_logits or {}).items()},
                sort_keys=True,
            )
            if t.prediction is not None
            else ""
            for t in traces
        ],
        dtype=object,
    )
    g.create_dataset("has_prediction", data=has_pred)
    g.create_dataset("pred_label_pred", data=pred_label)
    g.create_dataset("pred_label_prob", data=pred_prob)
    g.create_dataset("pred_logits_json", data=pred_logits_json, dtype=_STR_DTYPE)

    # Trace metadata (generic dict) — JSON-encoded per row.
    metadata_json = np.asarray(
        [json.dumps(t.trace_metadata, sort_keys=True, default=str) for t in traces],
        dtype=object,
    )
    g.create_dataset("trace_metadata_json", data=metadata_json, dtype=_STR_DTYPE)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def load_classifier_bank(path: Path | str) -> ClassifierBank:
    """Load a ClassifierBank from an HDF5 file at ``path``.

    The reverse of :func:`write_classifier_bank`. Sentinel masks
    (``has_label_truth``, ``has_prediction``) round-trip ``None`` values
    correctly so a 0 ground-truth label stays distinct from "missing".
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ClassifierBank file not found: {path}")
    with h5py.File(path, "r") as f:
        schema_version = _decode(f.attrs.get("schema_version", ""))
        if schema_version not in SUPPORTED_CLASSIFIER_BANK_VERSIONS:
            raise ValueError(
                f"ClassifierBank {path} has schema_version "
                f"{schema_version!r}; this build of myocard-egm-data "
                f"supports {SUPPORTED_CLASSIFIER_BANK_VERSIONS!r}."
            )
        created_utc = _decode(f.attrs.get("created_utc", ""))
        # Optional stable artifact id (added in ClassifierBank 0.2); absent
        # on 0.1 banks, which load with id=None.
        artifact_id = _decode(f.attrs["id"]) if "id" in f.attrs else None
        labels_raw = json.loads(_decode(f.attrs.get("labels_json", "{}")))
        labels = {int(k): str(v) for k, v in labels_raw.items()}

        banks = _read_banks_group(f["banks"]) if "banks" in f else []
        traces = _read_traces_group(f["traces"]) if "traces" in f else []

    return ClassifierBank(
        schema_version=schema_version,
        created_utc=created_utc,
        id=artifact_id,
        banks=banks,
        traces=traces,
        labels=labels,
    )


def _read_banks_group(g: h5py.Group) -> list[ClassifierBankMetaData]:
    bank_ids = [_decode(x) for x in g["bank_id"].asstr()[...]]
    bank_types = [_decode(x) for x in g["bank_type"].asstr()[...]]
    bank_paths = [_decode(x) for x in g["bank_path"].asstr()[...]]
    bank_metadata_jsons = [_decode(x) for x in g["bank_metadata_json"].asstr()[...]]
    out: list[ClassifierBankMetaData] = []
    for i, bid in enumerate(bank_ids):
        meta_dict = json.loads(bank_metadata_jsons[i]) if bank_metadata_jsons[i] else {}
        out.append(
            ClassifierBankMetaData(
                bank_id=bid,
                bank_type=bank_types[i],
                bank_path=bank_paths[i],
                bank_metadata=meta_dict,
            )
        )
    return out


def _read_traces_group(g: h5py.Group) -> list[ClassifierTrace]:
    signal = np.asarray(g["signal"][...], dtype=np.float32)
    bank_id = [_decode(x) for x in g["bank_id"].asstr()[...]]
    freq_hz = np.asarray(g["freq_hz"][...], dtype=np.float64)
    amp_type = [_decode(x) for x in g["amp_type"].asstr()[...]]
    split_raw = [_decode(x) for x in g["split"].asstr()[...]]
    has_label = np.asarray(g["has_label_truth"][...], dtype=np.uint8)
    label_truth_vals = np.asarray(g["label_truth"][...], dtype=np.int64)
    has_pred = np.asarray(g["has_prediction"][...], dtype=np.uint8)
    pred_label = np.asarray(g["pred_label_pred"][...], dtype=np.int64)
    pred_prob = np.asarray(g["pred_label_prob"][...], dtype=np.float64)
    pred_logits_jsons = [_decode(x) for x in g["pred_logits_json"].asstr()[...]]
    metadata_jsons = [_decode(x) for x in g["trace_metadata_json"].asstr()[...]]

    out: list[ClassifierTrace] = []
    for i in range(signal.shape[0]):
        label_truth = int(label_truth_vals[i]) if has_label[i] else None
        prediction: ClassifierPrediction | None = None
        if has_pred[i]:
            logits_dict_raw = json.loads(pred_logits_jsons[i]) if pred_logits_jsons[i] else {}
            logits_dict = {int(k): float(v) for k, v in logits_dict_raw.items()}
            prediction = ClassifierPrediction(
                label_pred=int(pred_label[i]),
                label_prob=float(pred_prob[i]),
                pred_logits=logits_dict,
            )
        metadata_dict: dict[str, Any] = json.loads(metadata_jsons[i]) if metadata_jsons[i] else {}
        out.append(
            ClassifierTrace(
                bank_id=bank_id[i],
                signal=signal[i].astype(np.float32, copy=False),
                freq_hz=float(freq_hz[i]),
                amp_type=amp_type[i],
                split=split_raw[i] if split_raw[i] else None,
                label_truth=label_truth,
                prediction=prediction,
                trace_metadata=metadata_dict,
            )
        )
    return out


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
