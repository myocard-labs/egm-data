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
from pydantic import BaseModel

__all__ = [
    "write_iafdb_bank",
    "write_synthetic_bank",
]

_STR_DTYPE = h5py.string_dtype(encoding="utf-8")

#: ``simulations/`` columns, mirroring the reader's constants. The nine
#: JSON columns are written as ``<name>_json``; the schema describes the
#: decoded form under the unsuffixed name.
_SIMULATION_INT_COLUMNS = ("simulation_id", "seed")
_SIMULATION_JSON_COLUMNS = (
    "geometry",
    "cell_model",
    "substrate",
    "substrate_summary",
    "activation",
    "electrodes",
    "backend",
    "label_policy",
    "label_names",
)


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
    """Write a Pydantic SyntheticBank to ``path`` in the schema-2.0 HDF5 layout.

    The Pydantic model is the in-memory representation. The producer
    (synthetic-egm-pipeline) constructs the model and calls this; the
    writer's only job is to serialize.

    Two things this refuses to write, both because the schema cannot
    express them and a bank that violates either is silently useless:

    - a bank with no ``bank_id`` (optional-in-schema, required-on-write,
      the convention shared by all three producer banks);
    - a trace whose ``simulation_id`` does not resolve into
      ``simulations/`` — an orphan trace has no generation config, and
      catching it here rather than at validation time means the producer
      learns about it at the moment it can still fix it.
    """
    if bank.bank_id is None:
        raise ValueError(
            "write_synthetic_bank requires bank.bank_id (the stable "
            "cross-artifact id) on new banks — the producer stamps it at "
            "write time (egm-contracts v0.5.0). Legacy banks without it can "
            "still be read, just not re-written."
        )
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

    _check_simulation_fk(bank)

    np_dtype = np.dtype(dtype)
    with h5py.File(path, "w") as f:
        _write_synthetic_root_attrs(f, bank)
        _write_simulations_group(f, bank.simulations)
        _write_synthetic_traces_group(f, bank.traces, n=n, n_samples=n_samples, dtype=np_dtype)
    return path


def _check_simulation_fk(bank: _synthetic_bank_models.SyntheticBank) -> None:
    """Refuse to write a trace whose ``simulation_id`` has no simulation.

    The cross-group foreign key ``traces/simulation_id`` ->
    ``simulations/simulation_id`` is checked by the contracts validator,
    but JSON Schema cannot express a cross-group reference, so the
    schema hands this to egm-data (``synthetic_bank.schema.json``:
    "egm-data checks it since JSON Schema cannot express a cross-group
    reference"). Doing it on the way *out* rather than relying on
    validation afterwards means a producer bug surfaces at the write
    call that caused it, not later against a file already on disk.
    """
    known = {int(_unwrap(s)) for s in bank.simulations.simulation_id}
    orphans = sorted(
        {
            int(_unwrap(sim_id))
            for sim_id in bank.traces.simulation_id
            if int(_unwrap(sim_id)) not in known
        }
    )
    if orphans:
        raise ValueError(
            f"write_synthetic_bank: traces reference simulation_id(s) {orphans} "
            f"that are absent from the simulations/ group (known ids: "
            f"{sorted(known)}). Every trace must resolve to the simulation "
            "that generated it; an orphan trace carries no generation config."
        )


def _write_synthetic_root_attrs(f: h5py.File, bank: _synthetic_bank_models.SyntheticBank) -> None:
    """Write the 2.0 bank-scoped root attrs.

    Only ``generation_params`` (the sweep-scoped θ-spec) is JSON-encoded
    — everything a 1.1 bank flattened into root attrs now lives
    per-simulation in ``simulations/``.
    """
    f.attrs["schema_version"] = current_version("synthetic_bank")
    f.attrs["created_utc"] = _datetime_str(bank.created_utc)
    f.attrs["bank_id"] = bank.bank_id
    f.attrs["description"] = bank.description or ""
    f.attrs["fs_hz"] = _enum_or_float(bank.fs_hz)
    f.attrs["trace_duration_ms"] = float(bank.trace_duration_ms)
    f.attrs["noise_bank_source"] = bank.noise_bank_source or ""
    f.attrs["generation_params_json"] = _dump_json(bank.generation_params)


def _write_simulations_group(f: h5py.File, simulations: Any) -> None:
    """Write ``simulations/`` — two int columns plus nine JSON columns.

    The polymorphic config objects are serialized one JSON document per
    row under a ``<name>_json`` dataset, matching what the schema's
    ``x-hdf5-mapping`` declares and what the contracts validator decodes.
    ``sort_keys=True`` keeps the bytes deterministic so the same logical
    bank always hashes the same.
    """
    g = f.create_group("simulations")
    for name in _SIMULATION_INT_COLUMNS:
        values = [int(_unwrap(x)) for x in getattr(simulations, name)]
        g.create_dataset(name, data=np.asarray(values, dtype=np.int64))

    for name in _SIMULATION_JSON_COLUMNS:
        encoded = [_dump_json(obj) for obj in getattr(simulations, name)]
        g.create_dataset(
            f"{name}_json",
            data=np.asarray(encoded, dtype=object),
            dtype=_STR_DTYPE,
        )


def _write_synthetic_traces_group(
    f: h5py.File,
    traces: Any,
    *,
    n: int,
    n_samples: int,
    dtype: np.dtype,
) -> None:
    """Write the collapsed 2.0 ``traces/`` group."""
    g = f.create_group("traces")

    signal = np.empty((n, n_samples), dtype=dtype)
    for i, row in enumerate(traces.signal):
        signal[i, :] = np.asarray(row, dtype=dtype)
    g.create_dataset(
        "signal",
        data=signal,
        dtype=dtype,
        chunks=(1, n_samples) if n > 0 else None,
        compression="gzip",
        compression_opts=4,
    )

    g.create_dataset(
        "simulation_id",
        data=np.asarray([int(_unwrap(x)) for x in traces.simulation_id], dtype=np.int64),
    )
    g.create_dataset(
        "pair_index",
        data=np.asarray([int(_unwrap(x)) for x in traces.pair_index], dtype=np.int64),
    )
    # Plain int label; the {int: name} map lives per simulation.
    g.create_dataset(
        "label", data=np.asarray([int(_unwrap(x)) for x in traces.label], dtype=np.int64)
    )
    g.create_dataset(
        "snr_db", data=np.asarray([float(_unwrap(x)) for x in traces.snr_db], dtype=np.float64)
    )
    for name in ("noise_record", "noise_channel"):
        values = [str(_unwrap(x)) for x in getattr(traces, name)]
        g.create_dataset(name, data=np.asarray(values, dtype=object), dtype=_STR_DTYPE)

    # Optional and permanently so: written only once the splitter has
    # anchored the crop (SEP2, Wave 2). Omitted rather than zero-filled —
    # absence means "unknown position", a zero would mean "activation at
    # the very start of the trace".
    if traces.activation_position is not None:
        g.create_dataset(
            "activation_position",
            data=np.asarray(
                [float(_unwrap(x)) for x in traces.activation_position], dtype=np.float32
            ),
        )


def _dump_json(obj: Any) -> str:
    """Serialize a Pydantic config object (or plain dict) to deterministic JSON.

    ``exclude_none=True`` keeps unset optionals out of the encoded row
    rather than writing them as ``null``: the per-function variants type
    their optional fields as plain values, not nullable ones, so a
    ``null`` would fail validation when the row is read back.
    """
    if isinstance(obj, BaseModel):
        return json.dumps(obj.model_dump(mode="json", exclude_none=True), sort_keys=True)
    return json.dumps(obj, sort_keys=True)


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
    if bank.bank_id is None:
        raise ValueError(
            "write_iafdb_bank requires bank.bank_id (the stable "
            "cross-artifact id) on new banks — the producer stamps it at "
            "write time (egm-contracts v0.5.0). Legacy banks without it can "
            "still be read, just not re-written."
        )
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
    # iafdb_bank 1.3, optional and permanently so: only an activation-aware
    # split produces a position. Sliding-window banks have no anchor and
    # multi-beat traces have no single position, so the column is omitted
    # rather than filled with a sentinel — absence means "unknown", and a
    # zero would read as "activation at the very start of the window".
    activation_position_arr = (
        np.asarray([float(_unwrap(x)) for x in t.activation_position], dtype=np.float32)
        if t.activation_position is not None
        else None
    )

    created_utc = _dt.datetime.now(_dt.timezone.utc).isoformat()

    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = current_version("iafdb_bank")
        f.attrs["created_utc"] = created_utc
        f.attrs["bank_id"] = bank.bank_id
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
        # iafdb_bank 1.3 (B11): optional pointer to the sibling audit-report
        # JSON, relative to the bank's own directory. A bank written without
        # the report simply omits the attr — the reader treats absence as
        # "no sidecar", never as an error.
        if bank.run_record_path is not None:
            f.attrs["run_record_path"] = bank.run_record_path
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
        if activation_position_arr is not None:
            traces.create_dataset(
                "activation_position", data=activation_position_arr, dtype=np.float32
            )
    return path
