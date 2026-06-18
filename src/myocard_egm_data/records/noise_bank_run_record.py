"""Reader, writer, and builder for the noise_bank_run_record JSON sidecar.

The noise_bank_run_record schema (myocard-egm-contracts >= 0.2.0)
captures the extraction provenance for a NoiseBank HDF5 file:
calibration scheme, threshold strategy, filter band, sliding-window
parameters, and optional per-trace audit arrays. It is written as a
sibling JSON file next to the bank with a matching name stem
(``foo_noise.h5`` ↔ ``foo_noise_run_record.json``), following the
existing ``metrics.csv`` ↔ ``run.json`` pattern in egm-classifier.

The mixer never opens this file; debugging, reproducibility audits,
and the white paper's methods section do. Producers (iafdb-pipeline's
noise-extraction CLI) write both the bank and the sidecar in the same
run; the schema does not enforce the pair — the convention does.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from myocard_egm_contracts.schema_info import current_version

__all__ = [
    "build_noise_bank_run_record",
    "load_noise_bank_run_record",
    "write_noise_bank_run_record",
]


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def load_noise_bank_run_record(path: Path | str) -> dict[str, Any]:
    """Load and parse a noise_bank_run_record.json document.

    Returns the raw dict (matching the schema). Schema validation is the
    contracts package's job — call
    ``myocard_egm_contracts.validators.validate_noise_bank_run_record``
    if you want to assert conformance before consuming.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, Mapping):
        raise ValueError(f"{path}: expected a JSON object at the top level.")
    return dict(doc)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_noise_bank_run_record(
    *,
    source: str,
    fs_hz: float,
    window_ms: float,
    window_samples: int,
    hop_ms: float,
    band_hz: Sequence[float],
    calibration_method: str,
    calibration_target_qrs_pp_mv: float | None,
    threshold_mode: str,
    threshold_value: float,
    source_records: Sequence[str],
    description: str = "",
    per_trace_provenance: Mapping[str, Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """Assemble a noise_bank_run_record document.

    Builds the nested structure required by the schema's
    ``windowing`` / ``calibration`` / ``selection`` sub-objects. The
    ``per_trace_provenance`` argument is forwarded through as-is when
    supplied; producers building paper-citable banks should include it.

    The ``created_utc`` timestamp and the ``schema_version`` are
    stamped automatically (mirroring :func:`build_run_record`).

    Parameters
    ----------
    source
        Free-form provenance tag identifying the upstream dataset. Must
        match the value stamped on the sibling NoiseBank.
    fs_hz, window_ms, window_samples, hop_ms
        Sampling rate and sliding-window parameters used during
        extraction. The validator enforces
        ``window_samples == round(window_ms * 1e-3 * fs_hz)``.
    band_hz
        Two-element ``[low, high]`` band-pass edges in Hz.
    calibration_method, calibration_target_qrs_pp_mv
        Calibration scheme (open string) and its target peak-to-peak
        QRS amplitude in mV. Pass ``None`` for the target when the
        method has no notion of one (e.g. ``"none"`` or ``"fixed_gain"``).
    threshold_mode, threshold_value
        ``"absolute"`` (mV cutoff) or ``"percentile"`` (0..100), and
        the corresponding numeric threshold.
    source_records
        Record identifiers from the upstream dataset that contributed
        at least one segment.
    description
        Free-form human-readable note about the run. Default empty.
    per_trace_provenance
        Optional mapping with the four parallel arrays
        ``patient_id`` / ``start_sample`` / ``peak_to_peak_mv`` /
        ``calibration_scalar``, aligned to the bank's traces. Producers
        SHOULD include this for paper-citable banks; smaller test
        fixtures may omit it.
    """
    doc: dict[str, Any] = {
        "schema_version": current_version("noise_bank_run_record"),
        "created_utc": _utc_now(),
        "source": source,
        "description": description,
        "fs_hz": float(fs_hz),
        "windowing": {
            "window_ms": float(window_ms),
            "window_samples": int(window_samples),
            "hop_ms": float(hop_ms),
        },
        "band_hz": [float(x) for x in band_hz],
        "calibration": {
            "method": str(calibration_method),
            "target_qrs_pp_mv": (
                float(calibration_target_qrs_pp_mv)
                if calibration_target_qrs_pp_mv is not None
                else None
            ),
        },
        "selection": {
            "threshold_mode": str(threshold_mode),
            "threshold_value": float(threshold_value),
        },
        "source_records": [str(r) for r in source_records],
    }
    if per_trace_provenance is not None:
        doc["per_trace_provenance"] = {
            "patient_id": [str(x) for x in per_trace_provenance["patient_id"]],
            "start_sample": [int(x) for x in per_trace_provenance["start_sample"]],
            "peak_to_peak_mv": [float(x) for x in per_trace_provenance["peak_to_peak_mv"]],
            "calibration_scalar": [float(x) for x in per_trace_provenance["calibration_scalar"]],
        }
    return doc


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_noise_bank_run_record(path: Path | str, doc: Mapping[str, Any]) -> Path:
    """Write a pre-built noise_bank_run_record document.

    Strict JSON: ``allow_nan=False`` so a downstream parser cannot
    encounter NaN/Infinity tokens; ``sort_keys=True`` so the byte
    stream is deterministic across runs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(dict(doc)), f, indent=2, allow_nan=False, sort_keys=True)
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _json_safe(obj: Any) -> Any:
    """Recursively make a value strictly-JSON-safe (mirrors records/writers.py).

    Converts non-finite floats to None, tuples to lists, Path to str.
    Used so producers can hand us numpy-derived numbers without us
    crashing on NaN at json.dump time.
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
