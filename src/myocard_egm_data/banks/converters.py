"""Converters from source-bank Pydantic models to :class:`ClassifierBank`.

Two conversion paths today:

- :func:`synthetic_bank_to_classifier` — takes a Pydantic
  ``SyntheticBank`` (from ``myocard-egm-contracts``) and a caller-
  supplied ``label_fn`` + ``labels`` dict, produces a ClassifierBank
  with one source-bank entry.
- :func:`iafdb_bank_to_classifier` — same for the IAFDB bank.

Each converter takes a ``label_fn``: a callable from the source-bank
Pydantic model to a length-N integer label array (or to ``None`` for
"no ground-truth labels for this trace"). Labeling policy stays on the
consumer side; egm-data does not pre-decide what counts as "fibrotic"
or "healthy".

Convenience wrappers (:func:`load_synthetic_bank_as_classifier`,
:func:`load_iafdb_bank_as_classifier`) chain the HDF5 read step with
the conversion step in one call. This is the recommended entry point
for most consumers; reach for the two-step variants only when you need
to inspect or manipulate the Pydantic model between read and convert.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from myocard_egm_contracts import iafdb_bank as _iafdb_bank_models
from myocard_egm_contracts import synthetic_bank as _synthetic_bank_models

from .classifier_bank import (
    ClassifierBank,
    ClassifierBankMetaData,
    ClassifierTrace,
)
from .readers import read_iafdb_bank_hdf5, read_synthetic_bank_hdf5

__all__ = [
    "iafdb_bank_to_classifier",
    "load_iafdb_bank_as_classifier",
    "load_synthetic_bank_as_classifier",
    "synthetic_bank_to_classifier",
]


LabelResult = tuple[np.ndarray, dict[int, str]]
"""What a label_fn returns: an ``[N]`` int64 label array plus the
integer-to-human-string translation dict. Both come from the same place
because they encode the same policy decision; bundling them prevents a
caller from accidentally using a labels dict that doesn't match the
emitted label ints.
"""

SyntheticLabelFn = Callable[[_synthetic_bank_models.SyntheticBank], LabelResult | None]
"""Label function for synthetic banks.

Receives the Pydantic SyntheticBank, returns either:

- ``(labels_array, labels_dict)`` — one integer label per trace plus the
  int-to-human-string translation.
- ``None`` — every ``label_truth`` set to ``None`` and ``labels`` dict
  left empty. Useful for converting an unlabeled bank for pretraining /
  inference-time use.

Example: binary fibrotic-vs-healthy from fibrosis_density::

    def label_fn(bank: SyntheticBank) -> tuple[np.ndarray, dict[int, str]]:
        labels = (
            np.asarray(bank.traces.fibrosis_density) > 0.0
        ).astype(np.int64)
        return labels, {0: "healthy", 1: "fibrotic"}
"""

IAFDBLabelFn = Callable[[_iafdb_bank_models.IafdbBank], LabelResult | None]
"""Label function for IAFDB banks. See :data:`SyntheticLabelFn`."""


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------


def synthetic_bank_to_classifier(
    pyd_bank: _synthetic_bank_models.SyntheticBank,
    *,
    label_fn: SyntheticLabelFn | None = None,
    bank_path: Path | str,
) -> ClassifierBank:
    """Convert a Pydantic SyntheticBank into a ClassifierBank.

    Parameters
    ----------
    pyd_bank
        The Pydantic-model view of the on-disk synthetic bank.
    label_fn
        Optional. Either returns ``(labels_array, labels_dict)`` or
        ``None``. If the whole ``label_fn`` is ``None`` (or it returns
        ``None``) the bank is built with every ``label_truth`` set to
        ``None`` and an empty labels dict — useful for pretraining /
        inference-time use on unlabeled data.
    bank_path
        Provenance only; recorded in
        :attr:`ClassifierBankMetaData.bank_path`.

    Returns
    -------
    ClassifierBank with one entry in ``banks`` and one ClassifierTrace
    per source-bank row.
    """
    n = len(pyd_bank.traces.signal)
    labels_arr, labels_dict = _run_label_fn(label_fn, pyd_bank, n)

    # The source bank's stable id becomes the ClassifierBankMetaData /
    # ClassifierTrace bank_id (egm-contracts v0.5.0). Required — a bank
    # without one can't be referenced from a ClassifierBank.
    if pyd_bank.bank_id is None:
        raise ValueError(
            "synthetic_bank_to_classifier requires the source bank to carry a "
            "stable bank_id (egm-contracts v0.5.0). Re-export the bank with a "
            "producer that stamps it."
        )
    source_id = pyd_bank.bank_id

    bank_metadata: dict[str, Any] = {
        "schema_version": _enum_or_str(pyd_bank.schema_version),
        "created_utc": _datetime_to_str(pyd_bank.created_utc),
        "description": pyd_bank.description,
        "trace_duration_ms": pyd_bank.trace_duration_ms,
        "simulator": pyd_bank.simulator,
        "cell_model": pyd_bank.cell_model,
        "patch_size_mm": pyd_bank.patch_size_mm,
        "patch_dr_mm": pyd_bank.patch_dr_mm,
        "ap_time_unit_ms": pyd_bank.ap_time_unit_ms,
        "fibrosis_strategy_name": pyd_bank.fibrosis_strategy_name,
        "fibrosis_params": pyd_bank.fibrosis_params,
        "electrode_config": pyd_bank.electrode_config,
        "mixer_config": pyd_bank.mixer_config,
        "experiment_config": pyd_bank.experiment_config,
        "noise_bank_source": pyd_bank.noise_bank_source,
    }
    source_meta = ClassifierBankMetaData(
        bank_id=source_id,
        bank_type="synthetic",
        bank_path=str(bank_path),
        bank_metadata=bank_metadata,
    )

    traces: list[ClassifierTrace] = []
    t = pyd_bank.traces
    fs_hz = _enum_or_float(pyd_bank.fs_hz)
    for i in range(n):
        trace_metadata: dict[str, Any] = {
            "patient_id": str(t.simulation_id[i]),
            "simulation_id": int(t.simulation_id[i]),
            "pair_index": int(t.pair_index[i]),
            "electrode_row": int(t.electrode_row[i]),
            "fibrosis_density": float(t.fibrosis_density[i]),
            "fibrosis_density_realized": float(t.fibrosis_density_realized[i]),
            "electrode_height_mm": float(t.electrode_height_mm[i]),
            "seed": int(t.seed[i]),
            "snr_db": float(t.snr_db[i]),
            "stim_edge": str(t.stim_edge[i]),
            "noise_record": str(t.noise_record[i]),
            "noise_channel": str(t.noise_channel[i]),
        }
        traces.append(
            ClassifierTrace(
                bank_id=source_id,
                signal=np.asarray(t.signal[i], dtype=np.float32),
                freq_hz=fs_hz,
                amp_type="mv",
                split=None,
                label_truth=int(labels_arr[i]) if labels_arr is not None else None,
                prediction=None,
                trace_metadata=trace_metadata,
            )
        )

    return ClassifierBank(
        banks=[source_meta],
        traces=traces,
        labels=labels_dict,
    )


def load_synthetic_bank_as_classifier(
    path: Path | str,
    *,
    label_fn: SyntheticLabelFn | None = None,
) -> ClassifierBank:
    """Read a synthetic-bank HDF5 file and convert to ClassifierBank in one call.

    The recommended entry point for most consumers. If you need to
    inspect or manipulate the source bank before conversion, use
    :func:`~myocard_egm_data.banks.readers.read_synthetic_bank_hdf5`
    followed by :func:`synthetic_bank_to_classifier`.
    """
    pyd_bank = read_synthetic_bank_hdf5(path)
    return synthetic_bank_to_classifier(pyd_bank, label_fn=label_fn, bank_path=path)


# ---------------------------------------------------------------------------
# IAFDB
# ---------------------------------------------------------------------------


def iafdb_bank_to_classifier(
    pyd_bank: _iafdb_bank_models.IafdbBank,
    *,
    label_fn: IAFDBLabelFn | None = None,
    bank_path: Path | str,
) -> ClassifierBank:
    """Convert a Pydantic IafdbBank into a ClassifierBank.

    See :func:`synthetic_bank_to_classifier` for the parameter semantics.
    The IAFDB schema (v0.1.2) carries no labels; the consumer-supplied
    ``label_fn`` is the only way to produce ``label_truth`` values.
    Passing ``label_fn=None`` (or having it return ``None``) yields a
    ClassifierBank with every ``label_truth`` set to ``None`` — useful
    for pretraining workflows on the full IAFDB.
    """
    n = len(pyd_bank.traces.signal)
    labels_arr, labels_dict = _run_label_fn(label_fn, pyd_bank, n)

    # The source bank's stable id becomes the ClassifierBankMetaData /
    # ClassifierTrace bank_id (egm-contracts v0.5.0). Required.
    if pyd_bank.bank_id is None:
        raise ValueError(
            "iafdb_bank_to_classifier requires the source bank to carry a "
            "stable bank_id (egm-contracts v0.5.0). Re-export the bank with a "
            "producer that stamps it."
        )
    source_id = pyd_bank.bank_id

    bank_metadata: dict[str, Any] = {
        "schema_version": _enum_or_str(pyd_bank.schema_version),
        "created_utc": _datetime_to_str(pyd_bank.created_utc),
        "source": _enum_or_str(pyd_bank.source),
        "trace_duration_ms": pyd_bank.trace_duration_ms,
        "calibration_method": _enum_or_str(pyd_bank.calibration_method),
        "calibration_target_qrs_pp_mv": pyd_bank.calibration_target_qrs_pp_mv,
        "threshold_mode": _enum_or_str(pyd_bank.threshold_mode),
        "threshold_value": pyd_bank.threshold_value,
        "band_hz": list(pyd_bank.band_hz),
        "window_ms": pyd_bank.window_ms,
        "window_samples": pyd_bank.window_samples,
        "hop_ms": pyd_bank.hop_ms,
        "source_records": list(pyd_bank.source_records),
    }
    source_meta = ClassifierBankMetaData(
        bank_id=source_id,
        bank_type="iafdb",
        bank_path=str(bank_path),
        bank_metadata=bank_metadata,
    )

    traces: list[ClassifierTrace] = []
    t = pyd_bank.traces
    fs_hz = _enum_or_float(pyd_bank.fs_hz)
    for i in range(n):
        trace_metadata: dict[str, Any] = {
            "patient_id": str(t.patient_id[i]),
            "source_record": str(t.source_record[i]),
            "source_channel": str(t.source_channel[i]),
            # Codegen wraps numeric items in a constraint type with a
            # ``.root`` accessor; unwrap before casting.
            "start_sample": int(_unwrap(t.start_sample[i])),
            "peak_to_peak_mv": float(_unwrap(t.peak_to_peak_mv[i])),
            "calibration_scalar": float(t.calibration_scalar[i]),
        }
        traces.append(
            ClassifierTrace(
                bank_id=source_id,
                signal=np.asarray(t.signal[i], dtype=np.float32),
                freq_hz=fs_hz,
                amp_type="mv",
                split=None,
                label_truth=int(labels_arr[i]) if labels_arr is not None else None,
                prediction=None,
                trace_metadata=trace_metadata,
            )
        )

    return ClassifierBank(
        banks=[source_meta],
        traces=traces,
        labels=labels_dict,
    )


def load_iafdb_bank_as_classifier(
    path: Path | str,
    *,
    label_fn: IAFDBLabelFn | None = None,
) -> ClassifierBank:
    """Read an IAFDB-bank HDF5 file and convert to ClassifierBank in one call.

    The recommended entry point for most consumers. If you need to
    inspect or manipulate the source bank before conversion, use
    :func:`~myocard_egm_data.banks.readers.read_iafdb_bank_hdf5`
    followed by :func:`iafdb_bank_to_classifier`.
    """
    pyd_bank = read_iafdb_bank_hdf5(path)
    return iafdb_bank_to_classifier(pyd_bank, label_fn=label_fn, bank_path=path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_label_fn(
    label_fn: Any,
    pyd_bank: Any,
    n: int,
) -> tuple[np.ndarray | None, dict[int, str]]:
    """Resolve a possibly-``None`` label_fn into ``(labels_array, labels_dict)``.

    - If ``label_fn`` is ``None`` (the converter caller did not supply
      one), returns ``(None, {})``.
    - If ``label_fn`` returns ``None``, returns ``(None, {})`` — same
      meaning: no ground-truth labels.
    - If ``label_fn`` returns ``(array, dict)``, validates the array
      shape and returns both.
    """
    if label_fn is None:
        return None, {}
    result = label_fn(pyd_bank)
    if result is None:
        return None, {}
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError(
            "label_fn must return either None or a 2-tuple "
            "(labels_array, labels_dict). See LabelResult."
        )
    arr_raw, labels_dict = result
    arr = np.asarray(arr_raw, dtype=np.int64)
    if arr.shape != (n,):
        raise ValueError(
            f"label_fn returned label array of shape {arr.shape}; "
            f"expected ({n},) — one integer label per trace."
        )
    if not isinstance(labels_dict, dict):
        raise ValueError("label_fn must return a dict[int, str] as the second element.")
    return arr, {int(k): str(v) for k, v in labels_dict.items()}


def _enum_or_str(value: Any) -> str:
    """Unwrap a Pydantic codegen Enum to its underlying string value."""
    inner = getattr(value, "value", value)
    return str(inner)


def _enum_or_float(value: Any) -> float:
    """Unwrap a Pydantic codegen Enum to its underlying float value."""
    inner = getattr(value, "value", value)
    return float(inner)


def _unwrap(value: Any) -> Any:
    """Unwrap a Pydantic codegen constraint-root container.

    The codegen wraps constrained scalar fields (e.g. ``start_sample``
    items declared as ``integer, minimum: 0``) in a one-field root model
    with a ``.root`` attribute. Pass through unwrapped scalars.
    """
    return getattr(value, "root", value)


def _datetime_to_str(value: Any) -> str:
    """Normalize a possibly-datetime value to an ISO 8601 string."""
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)
