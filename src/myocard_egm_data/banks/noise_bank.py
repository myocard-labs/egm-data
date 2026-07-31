"""HDF5 reader + writer for the (slim) noise_bank schema.

The noise_bank format owned by myocard-egm-contracts (>= 0.2.0) is
intentionally minimal: it carries only what the synthetic-EGM mixer
consumes — the signal waveform plus per-trace source identifiers,
with sample-rate at the root for the mixer's sanity check.

Extraction provenance (calibration scheme, threshold strategy,
windowing parameters, per-trace audit) lives in a sibling JSON file
described by the ``noise_bank_run_record`` schema and handled by
:mod:`myocard_egm_data.records.noise_bank_run_record`. The two files
are paired by convention (matching name stem, same directory) — no
back-pointer is recorded inside the bank, mirroring the
``metrics.csv`` ↔ ``run.json`` pattern used by egm-classifier.

Since noise_bank 1.1 (egm-contracts v0.6.0) the bank carries its own
stable ``bank_id`` root attr, so consumers read the id from the bank
rather than having to locate the sidecar. Both files carry it and the
schema requires them to agree when both are present; because JSON
Schema cannot compare across two files, that cross-file check is
egm-data's to make — see the id-consistency checks in :mod:`.writers`.

This module provides:

- :func:`read_noise_bank_hdf5` — HDF5 → Pydantic NoiseBank model.
- :func:`write_noise_bank` — Pydantic NoiseBank → HDF5 file.

Producers (the iafdb-pipeline noise-extraction CLI) construct the
Pydantic model and hand it to the writer. The writer's only job is to
serialize; the contracts' validator is what asserts schema conformance.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from myocard_egm_contracts import noise_bank as _noise_bank_models
from myocard_egm_contracts.schema_info import current_version, supported_versions

__all__ = [
    "read_noise_bank_hdf5",
    "write_noise_bank",
]

_STR_DTYPE = h5py.string_dtype(encoding="utf-8")

# The bank carries a single traces/ group; the constants mirror those used
# by the iafdb_bank reader so cross-bank refactoring stays trivial.
_TRACES_GROUP = "traces"
_SIGNAL_DATASET = "signal"


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_noise_bank(
    bank: _noise_bank_models.NoiseBank,
    path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a Pydantic NoiseBank to ``path`` in the schema-1.1 HDF5 layout.

    The Pydantic model is the in-memory representation. The producer
    (iafdb-pipeline today) constructs it; the writer serializes. The
    sibling run record JSON is the caller's responsibility — see
    :func:`myocard_egm_data.records.write_noise_bank_run_record`.

    ``bank_id`` is *optional in the schema but required on write* (the
    cross-artifact-linkage convention shared with ``write_synthetic_bank``
    and ``write_iafdb_bank``): legacy banks written before egm-contracts
    v0.6.0 carry no id and still read, but every new bank gets one, so
    downstream artifacts can reference it. The schema cannot enforce this
    — JSON Schema has no "required only when writing" — so the writer does.

    Parameters
    ----------
    bank
        The :class:`myocard_egm_contracts.noise_bank.NoiseBank` to write.
    path
        Destination HDF5 file. Parent directory is created if missing.
    overwrite
        If ``False`` (default) and ``path`` exists, raise
        :class:`FileExistsError`.
    """
    if bank.bank_id is None:
        raise ValueError(
            "write_noise_bank requires bank.bank_id (the stable "
            "cross-artifact id) on new banks — the producer stamps it at "
            "write time (egm-contracts v0.6.0, noise_bank 1.1). Legacy "
            "banks without it can still be read, just not re-written."
        )
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True.")
    path.parent.mkdir(parents=True, exist_ok=True)

    t = bank.traces
    n = len(t.signal)
    if n > 0:
        T = len(t.signal[0])
        for i, row in enumerate(t.signal):
            if len(row) != T:
                raise ValueError(
                    f"NoiseBank segment {i} has length {len(row)}; expected {T}. "
                    "All segments must share the trace length."
                )
    else:
        T = 0

    signal_arr = np.empty((n, T), dtype=np.float32)
    for i, row in enumerate(t.signal):
        signal_arr[i] = np.asarray(row, dtype=np.float32)

    source_record_arr = np.asarray([str(x) for x in t.source_record], dtype=object)
    source_channel_arr = np.asarray([str(x) for x in t.source_channel], dtype=object)

    created_utc = _dt.datetime.now(_dt.timezone.utc).isoformat()

    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = current_version("noise_bank")
        f.attrs["created_utc"] = created_utc
        f.attrs["bank_id"] = bank.bank_id
        f.attrs["source"] = str(bank.source)
        f.attrs["fs_hz"] = float(bank.fs_hz)

        traces = f.create_group(_TRACES_GROUP)
        traces.create_dataset(_SIGNAL_DATASET, data=signal_arr, dtype=np.float32)
        traces.create_dataset("source_record", data=source_record_arr, dtype=_STR_DTYPE)
        traces.create_dataset("source_channel", data=source_channel_arr, dtype=_STR_DTYPE)
    return path


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def read_noise_bank_hdf5(path: Path | str) -> _noise_bank_models.NoiseBank:
    """Read a noise-bank HDF5 file into a Pydantic NoiseBank.

    Performs the version-enum check against the installed
    myocard-egm-contracts (so a file produced by a newer major version
    of the schema raises clearly rather than silently mis-deserializing).
    The bulk per-trace arrays round-trip through Python lists because
    the JSON-Schema codegen models declare ``signal`` as
    ``list[list[float]]`` — fine for Phase-1 bank sizes; a faster
    HDF5→numpy fast path can be added later without changing this API.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"NoiseBank file not found: {path}")
    with h5py.File(path, "r") as f:
        _check_version(f, path)
        if _TRACES_GROUP not in f:
            raise ValueError(f"NoiseBank {path} has no '{_TRACES_GROUP}' group.")

        doc: dict[str, Any] = {
            "schema_version": _str_attr(f, "schema_version", required=True, path=path),
            "created_utc": _str_attr(f, "created_utc", required=True, path=path),
            "bank_id": _opt_str_attr(f, "bank_id"),
            "source": _str_attr(f, "source", required=True, path=path),
            "fs_hz": _float_attr(f, "fs_hz", required=True, path=path),
        }

        g = f[_TRACES_GROUP]
        if _SIGNAL_DATASET not in g:
            raise ValueError(f"NoiseBank {path} missing '{_TRACES_GROUP}/{_SIGNAL_DATASET}'.")
        signal = np.asarray(g[_SIGNAL_DATASET][...], dtype=np.float32)
        doc["traces"] = {
            "signal": signal.tolist(),
            "source_record": [_decode(x) for x in g["source_record"][...]],
            "source_channel": [_decode(x) for x in g["source_channel"][...]],
        }

    return _noise_bank_models.NoiseBank.model_validate(doc)


# ---------------------------------------------------------------------------
# Helpers (mirroring banks/readers.py for symmetry)
# ---------------------------------------------------------------------------


def _check_version(f: h5py.File, path: Path) -> None:
    raw = f.attrs.get("schema_version", "")
    version = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    supported = supported_versions("noise_bank")
    if version not in supported:
        raise ValueError(
            f"NoiseBank {path} has schema_version {version!r}; the installed "
            f"myocard-egm-contracts supports noise_bank versions {supported}."
        )


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _str_attr(
    f: h5py.File,
    name: str,
    *,
    required: bool,
    path: Path,
    default: str = "",
) -> str:
    if name not in f.attrs:
        if required:
            raise ValueError(f"NoiseBank {path} is missing required root attr {name!r}.")
        return default
    raw = f.attrs[name]
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def _opt_str_attr(f: h5py.File, name: str) -> str | None:
    """Read an optional string root attr, returning ``None`` when absent.

    Distinct from ``_str_attr(..., required=False)`` which returns ``""``:
    an absent optional id must be ``None`` (so the Pydantic model leaves it
    unset), never ``""`` (which would fail the stable-id pattern).
    """
    if name not in f.attrs:
        return None
    raw = f.attrs[name]
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def _float_attr(f: h5py.File, name: str, *, required: bool, path: Path) -> float:
    if name not in f.attrs:
        if required:
            raise ValueError(f"NoiseBank {path} is missing required root attr {name!r}.")
        return 0.0
    return float(f.attrs[name])
