"""Build, write, and read the ``noise_bank_run_record`` JSON sidecar.

The schema (``myocard_egm_contracts.noise_bank_run_record``, >= 1.0)
captures the extraction provenance for a NoiseBank HDF5 file:
calibration scheme, threshold strategy, filter band, sliding-window
parameters, and optional per-trace audit arrays. Written as a sibling
JSON file next to the bank with a matching name stem
(``foo_noise.h5`` ↔ ``foo_noise_run_record.json``), following the same
convention as ``metrics.csv`` ↔ ``run.json`` in egm-classifier.

The mixer never opens this file; debugging, reproducibility audits,
and the white paper's methods section do. Producers
(iafdb-pipeline's noise-extraction CLI) write both the bank and the
sidecar in the same run; the schema does not enforce the pair — the
convention does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from myocard_egm_contracts._generated.python.noise_bank_run_record import (
    Calibration,
    NoiseBankRunRecord,
    PerTraceProvenance,
    Selection,
    ThresholdMode,
    Windowing,
)
from myocard_egm_contracts._generated.python.noise_bank_run_record import (
    SchemaVersion as NoiseBankRunRecordSchemaVersion,
)
from myocard_egm_contracts.schema_info import current_version

from .._serialization import _load_pydantic_json, _utc_now, _write_pydantic_json

__all__ = [
    "Calibration",
    "NoiseBankRunRecord",
    "PerTraceProvenance",
    "Selection",
    "ThresholdMode",
    "Windowing",
    "build_noise_bank_run_record",
    "load_noise_bank_run_record",
    "write_noise_bank_run_record",
]


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
    bank_id: str | None = None,
    description: str = "",
    per_trace_provenance: Mapping[str, Sequence[Any]] | None = None,
) -> NoiseBankRunRecord:
    """Assemble a :class:`NoiseBankRunRecord` with version + timestamp stamped.

    Producers pass scalar / flat-list values for every nested block;
    this helper builds the typed :class:`Windowing` / :class:`Calibration`
    / :class:`Selection` / :class:`PerTraceProvenance` sub-models so the
    call site stays ergonomic. The validator enforces the cross-field
    invariant ``window_samples == round(window_ms * 1e-3 * fs_hz)``.

    Parameters
    ----------
    source
        Free-form provenance tag identifying the upstream dataset.
        Must match the value stamped on the sibling NoiseBank.
    fs_hz, window_ms, window_samples, hop_ms
        Sampling rate and sliding-window parameters used during
        extraction.
    band_hz
        Two-element ``[low, high]`` band-pass edges in Hz.
    calibration_method, calibration_target_qrs_pp_mv
        Calibration scheme (open string) and its target peak-to-peak
        QRS amplitude in mV. Pass ``None`` for the target when the
        method has no notion of one (e.g. ``"none"`` or
        ``"fixed_gain"``).
    threshold_mode, threshold_value
        ``"absolute"`` (mV cutoff) or ``"percentile"`` (0..100), and
        the corresponding numeric threshold.
    source_records
        Record identifiers from the upstream dataset that contributed
        at least one segment.
    bank_id
        Optional stable cross-artifact id of the noise bank this sidecar
        describes (egm-contracts v0.5.0; e.g. ``nbank_iafdb_2026-06-15``).
        Optional; the producer (iafdb-pipeline) stamps it. Pattern
        validated by the Pydantic model.
    description
        Free-form human-readable note about the run. Default empty.
    per_trace_provenance
        Optional mapping with the four parallel arrays
        ``patient_id`` / ``start_sample`` / ``peak_to_peak_mv`` /
        ``calibration_scalar``, aligned to the bank's traces. Producers
        SHOULD include this for paper-citable banks; smaller test
        fixtures may omit it.
    """
    ptp: PerTraceProvenance | None = None
    if per_trace_provenance is not None:
        # ``model_validate`` is the easiest way to construct a model
        # whose fields are RootModel wrappers (StartSampleItem, etc.)
        # from raw int / float lists — Pydantic handles the per-item
        # wrapping internally.
        ptp = PerTraceProvenance.model_validate(
            {
                "patient_id": [str(x) for x in per_trace_provenance["patient_id"]],
                "start_sample": [int(x) for x in per_trace_provenance["start_sample"]],
                "peak_to_peak_mv": [float(x) for x in per_trace_provenance["peak_to_peak_mv"]],
                "calibration_scalar": [
                    float(x) for x in per_trace_provenance["calibration_scalar"]
                ],
            }
        )
    return NoiseBankRunRecord.model_validate(
        {
            "schema_version": NoiseBankRunRecordSchemaVersion(
                current_version("noise_bank_run_record"),
            ),
            "created_utc": _utc_now(),
            "bank_id": bank_id,
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
                "threshold_mode": ThresholdMode(threshold_mode),
                "threshold_value": float(threshold_value),
            },
            "source_records": [str(r) for r in source_records],
            "per_trace_provenance": ptp,
        }
    )


def write_noise_bank_run_record(path: Path | str, record: NoiseBankRunRecord) -> Path:
    """Write a :class:`NoiseBankRunRecord` to ``path`` as strict, deterministic JSON."""
    return _write_pydantic_json(path, record)


def load_noise_bank_run_record(path: Path | str) -> NoiseBankRunRecord:
    """Load a ``noise_bank_run_record.json`` document into a typed model.

    Raises ``pydantic.ValidationError`` on shape mismatch.
    """
    return _load_pydantic_json(path, NoiseBankRunRecord)
