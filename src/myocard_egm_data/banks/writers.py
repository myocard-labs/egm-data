"""Writers for the source-bank HDF5 formats owned by ``myocard-egm-contracts``.

Producers (the synthetic-EGM pipeline mixer; the IAFDB pipeline bank
exporter) construct the Pydantic ``SyntheticBank`` / ``IafdbBank`` model
codegen'd from the JSON Schemas and hand it to the matching writer.
We do NOT keep a second set of dataclasses here that mirror those
schemas — the Pydantic model is the only in-memory representation.

The writers do not enforce the schema — that is what the contracts'
file-level validators are for — but they produce HDF5 layouts that the
matching validator should accept on a clean round-trip. The tests in
this package hold that line.

JSON-encoded sub-configs (``fibrosis_params`` etc. on the synthetic
bank) are stored as flat HDF5 string attrs with a ``_json`` suffix on
write; the corresponding readers decode them.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from myocard_egm_contracts import iafdb_bank as _iafdb_bank_models
from myocard_egm_contracts import synthetic_bank as _synthetic_bank_models
from myocard_egm_contracts.schema_info import current_version

__all__ = [
    "write_iafdb_bank",
    "write_synthetic_bank",
]

_STR_DTYPE = h5py.string_dtype(encoding="utf-8")


def _enum_or_str(value: Any) -> str:
    inner = getattr(value, "value", value)
    return str(inner)


def _enum_or_float(value: Any) -> float:
    inner = getattr(value, "value", value)
    return float(inner)


def _unwrap(value: Any) -> Any:
    """Unwrap a codegen constraint-root container (``.root`` accessor)."""
    return getattr(value, "root", value)


def _datetime_str(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


# ---------------------------------------------------------------------------
# Synthetic bank
# ---------------------------------------------------------------------------


def write_synthetic_bank(
    bank: _synthetic_bank_models.SyntheticBank,
    path: Path | str,
    *,
    dtype: np.dtype | type = np.float32,
    overwrite: bool = False,
) -> Path:
    """Write a Pydantic SyntheticBank to ``path`` in the schema-1.0 HDF5 layout.

    The Pydantic model is the in-memory representation. The producer
    (synthetic-egm-pipeline) constructs the model and calls this; the
    writer's only job is to serialize.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True.")
    path.parent.mkdir(parents=True, exist_ok=True)

    t = bank.traces
    n = len(t.signal)
    if n > 0:
        n_samples = len(t.signal[0])
        for i, row in enumerate(t.signal):
            if len(row) != n_samples:
                raise ValueError(
                    f"Synthetic bank trace {i} has length {len(row)}; "
                    f"expected {n_samples}. All traces must share T."
                )
    else:
        n_samples = round(bank.trace_duration_ms * 1e-3 * _enum_or_float(bank.fs_hz))

    np_dtype = np.dtype(dtype)
    with h5py.File(path, "w") as f:
        _write_synthetic_root_attrs(f, bank)
        _create_synthetic_traces_group(f, n_samples=n_samples, dtype=np_dtype)
        _append_synthetic_rows(f, bank.traces, n=n, dtype=np_dtype)
    return path


def _write_synthetic_root_attrs(f: h5py.File, bank: _synthetic_bank_models.SyntheticBank) -> None:
    f.attrs["schema_version"] = current_version("synthetic_bank")
    f.attrs["created_utc"] = _datetime_str(bank.created_utc)
    f.attrs["description"] = bank.description
    f.attrs["fs_hz"] = _enum_or_float(bank.fs_hz)
    f.attrs["trace_duration_ms"] = float(bank.trace_duration_ms)
    f.attrs["simulator"] = bank.simulator
    f.attrs["cell_model"] = bank.cell_model
    # Pydantic types these as nullable per the schema (the field isn't
    # in `required`); on disk we promise a concrete float, defaulting
    # to 0.0 if a producer didn't supply one.
    f.attrs["patch_size_mm"] = float(bank.patch_size_mm if bank.patch_size_mm is not None else 0.0)
    f.attrs["patch_dr_mm"] = float(bank.patch_dr_mm if bank.patch_dr_mm is not None else 0.0)
    f.attrs["ap_time_unit_ms"] = float(
        bank.ap_time_unit_ms if bank.ap_time_unit_ms is not None else 0.0
    )
    f.attrs["fibrosis_strategy_name"] = bank.fibrosis_strategy_name
    f.attrs["fibrosis_params_json"] = json.dumps(bank.fibrosis_params, sort_keys=True)
    f.attrs["electrode_config_json"] = json.dumps(bank.electrode_config, sort_keys=True)
    f.attrs["mixer_config_json"] = json.dumps(bank.mixer_config, sort_keys=True)
    f.attrs["experiment_config_json"] = json.dumps(bank.experiment_config, sort_keys=True)
    f.attrs["noise_bank_source"] = bank.noise_bank_source


def _create_synthetic_traces_group(f: h5py.File, n_samples: int, dtype: np.dtype) -> None:
    g = f.create_group("traces")
    g.create_dataset(
        "signal",
        shape=(0, n_samples),
        maxshape=(None, n_samples),
        dtype=dtype,
        chunks=(1, n_samples),
        compression="gzip",
        compression_opts=4,
    )
    for name in ("simulation_id", "pair_index", "electrode_row", "seed"):
        g.create_dataset(name, shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(256,))
    for name in (
        "fibrosis_density",
        "fibrosis_density_realized",
        "electrode_height_mm",
        "snr_db",
    ):
        g.create_dataset(name, shape=(0,), maxshape=(None,), dtype=np.float64, chunks=(256,))
    for name in ("stim_edge", "noise_record", "noise_channel"):
        g.create_dataset(name, shape=(0,), maxshape=(None,), dtype=_STR_DTYPE, chunks=(256,))


def _append_synthetic_rows(
    f: h5py.File,
    traces: Any,
    n: int,
    dtype: np.dtype,
) -> None:
    if n == 0:
        return
    g = f["traces"]
    sig = g["signal"]
    n_samples = sig.shape[1]

    new_signal = np.empty((n, n_samples), dtype=dtype)
    for i, row in enumerate(traces.signal):
        new_signal[i, :] = np.asarray(row, dtype=dtype)

    sig.resize((n, n_samples))
    sig[:, :] = new_signal

    def _col(name: str, np_dtype: Any, raw: Any) -> None:
        ds = g[name]
        ds.resize((n,))
        ds[:] = np.array(raw, dtype=np_dtype)

    _col("simulation_id", np.int64, [int(_unwrap(x)) for x in traces.simulation_id])
    _col("pair_index", np.int64, [int(_unwrap(x)) for x in traces.pair_index])
    _col("electrode_row", np.int64, [int(_unwrap(x)) for x in traces.electrode_row])
    _col("seed", np.int64, [int(_unwrap(x)) for x in traces.seed])
    _col("fibrosis_density", np.float64, [float(_unwrap(x)) for x in traces.fibrosis_density])
    _col(
        "fibrosis_density_realized",
        np.float64,
        [float(_unwrap(x)) for x in traces.fibrosis_density_realized],
    )
    _col(
        "electrode_height_mm",
        np.float64,
        [float(_unwrap(x)) for x in traces.electrode_height_mm],
    )
    _col("snr_db", np.float64, [float(_unwrap(x)) for x in traces.snr_db])
    _col("stim_edge", object, [_enum_or_str(x) for x in traces.stim_edge])
    _col("noise_record", object, [str(_unwrap(x)) for x in traces.noise_record])
    _col("noise_channel", object, [str(_unwrap(x)) for x in traces.noise_channel])


# ---------------------------------------------------------------------------
# IAFDB bank
# ---------------------------------------------------------------------------


def write_iafdb_bank(
    bank: _iafdb_bank_models.IafdbBank,
    path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a Pydantic IafdbBank to ``path`` in the schema-1.0 HDF5 layout.

    The bank itself carries no label column (per egm-contracts v0.1.2);
    labeling is consumer-side policy applied at ClassifierBank
    conversion time.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True.")
    path.parent.mkdir(parents=True, exist_ok=True)

    t = bank.traces
    n = len(t.signal)
    window_samples = int(bank.window_samples)

    signal_arr = np.empty((n, window_samples), dtype=np.float32)
    for i, row in enumerate(t.signal):
        if len(row) != window_samples:
            raise ValueError(f"IAFDB segment {i} has length {len(row)}; expected {window_samples}.")
        signal_arr[i] = np.asarray(row, dtype=np.float32)

    patient_arr = np.asarray([str(_unwrap(x)) for x in t.patient_id], dtype=object)
    source_record_arr = np.asarray([str(_unwrap(x)) for x in t.source_record], dtype=object)
    source_channel_arr = np.asarray([str(_unwrap(x)) for x in t.source_channel], dtype=object)
    start_sample_arr = np.asarray([int(_unwrap(x)) for x in t.start_sample], dtype=np.int64)
    peak_to_peak_arr = np.asarray([float(_unwrap(x)) for x in t.peak_to_peak_mv], dtype=np.float32)
    calibration_scalar_arr = np.asarray(
        [float(_unwrap(x)) for x in t.calibration_scalar], dtype=np.float32
    )

    created_utc = _dt.datetime.now(_dt.timezone.utc).isoformat()

    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = current_version("iafdb_bank")
        f.attrs["created_utc"] = created_utc
        f.attrs["source"] = _enum_or_str(bank.source)
        f.attrs["fs_hz"] = _enum_or_float(bank.fs_hz)
        f.attrs["trace_duration_ms"] = float(bank.trace_duration_ms)
        f.attrs["calibration_method"] = _enum_or_str(bank.calibration_method)
        f.attrs["calibration_target_qrs_pp_mv"] = float(bank.calibration_target_qrs_pp_mv)
        f.attrs["threshold_mode"] = _enum_or_str(bank.threshold_mode)
        # iafdb_bank schema 1.1 allows threshold_value to be null when
        # threshold_mode is "none" (unfiltered export). HDF5 has no
        # native null, so we stamp NaN — JSON Schema accepts NaN as a
        # number and the validator round-trips it via json.loads.
        f.attrs["threshold_value"] = (
            float(bank.threshold_value) if bank.threshold_value is not None else float("nan")
        )
        f.attrs["band_hz"] = np.asarray([float(_unwrap(x)) for x in bank.band_hz], dtype=np.float64)
        f.attrs["window_ms"] = float(bank.window_ms)
        f.attrs["window_samples"] = window_samples
        f.attrs["hop_ms"] = float(bank.hop_ms if bank.hop_ms is not None else 0.0)
        f.attrs.create(
            "source_records",
            np.asarray([str(s) for s in bank.source_records], dtype=object),
            dtype=_STR_DTYPE,
        )

        traces = f.create_group("traces")
        traces.create_dataset("signal", data=signal_arr, dtype=np.float32)
        traces.create_dataset("patient_id", data=patient_arr, dtype=_STR_DTYPE)
        traces.create_dataset("source_record", data=source_record_arr, dtype=_STR_DTYPE)
        traces.create_dataset("source_channel", data=source_channel_arr, dtype=_STR_DTYPE)
        traces.create_dataset("start_sample", data=start_sample_arr, dtype=np.int64)
        traces.create_dataset("peak_to_peak_mv", data=peak_to_peak_arr, dtype=np.float32)
        traces.create_dataset("calibration_scalar", data=calibration_scalar_arr, dtype=np.float32)
    return path
