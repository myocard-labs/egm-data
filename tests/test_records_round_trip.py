"""Round-trip the record writers/readers against the contracts' validators.

Covers run_record / metrics / hybrid_eval_metrics / model_metadata.
Per-trace predictions live inside ClassifierBank now — covered by
test_banks_round_trip.py.
"""

from __future__ import annotations

from pathlib import Path

from myocard_egm_contracts._generated.python.run_record import EpochRecord, ReliabilityBin
from myocard_egm_contracts.validators import (
    validate_hybrid_eval_metrics,
    validate_metrics,
    validate_model_metadata,
    validate_noise_bank_run_record,
    validate_run_record,
)

from myocard_egm_data.records import (
    build_noise_bank_run_record,
    load_metrics_csv,
    load_noise_bank_run_record,
    load_run_record,
    write_hybrid_eval_metrics,
    write_metrics_csv,
    write_model_metadata,
    write_noise_bank_run_record,
    write_run_record,
)
from myocard_egm_data.records.writers import build_run_record


def _epoch_records() -> list[EpochRecord]:
    return [
        EpochRecord(
            epoch=1,
            lr=1e-3,
            train_loss=0.5,
            val_loss=0.45,
            epoch_seconds=12.0,
            val_metrics={
                "auroc": 0.85,
                "accuracy": 0.80,
                "precision": 0.81,
                "recall": 0.78,
                "f1": 0.79,
                "ece": 0.05,
            },
            val_reliability=[
                ReliabilityBin(lo=0.0, hi=0.5, count=10, confidence=0.25, accuracy=0.20),
                ReliabilityBin(lo=0.5, hi=1.0, count=10, confidence=0.75, accuracy=0.80),
            ],
        ),
        EpochRecord(
            epoch=2,
            lr=9e-4,
            train_loss=0.4,
            val_loss=0.42,
            epoch_seconds=11.5,
            val_metrics={
                "auroc": 0.87,
                "accuracy": 0.82,
                "precision": 0.83,
                "recall": 0.80,
                "f1": 0.81,
                "ece": 0.04,
            },
            val_reliability=[],
        ),
    ]


# ---------------------------------------------------------------------------
# run_record
# ---------------------------------------------------------------------------


def test_run_record_round_trips(tmp_path: Path) -> None:
    """Build a complete run.json (config + epochs + best + test block),
    write it, validate against the contract, read it back. Also checks
    that `best_epoch` picks epoch 2 over epoch 1 because the val_auroc
    is higher there."""
    doc = build_run_record(
        config={"input_length": 512, "batch_size": 64, "lr": 1e-3},
        run_meta={
            "run_id": "test-run-001",
            "git_sha": "deadbeef",
            "host": "localhost",
            "model_version": "v1.0",
            "training_started_utc": "2026-06-16T12:00:00+00:00",
            "training_ended_utc": "2026-06-16T13:00:00+00:00",
        },
        epoch_records=_epoch_records(),
        select_metric="auroc",
        test_loss=0.40,
        test_metrics={"auroc": 0.86, "accuracy": 0.81, "reliability": []},
    )
    path = tmp_path / "run.json"
    write_run_record(path, doc)

    result = validate_run_record(path)
    assert result.ok, result.issues

    loaded = load_run_record(path)
    assert loaded["schema_version"] == "1.0"
    assert loaded["best"]["epoch"] == 2  # epoch 2 has higher auroc
    assert len(loaded["epochs"]) == 2


# ---------------------------------------------------------------------------
# metrics.csv
# ---------------------------------------------------------------------------


def test_metrics_csv_round_trips(tmp_path: Path) -> None:
    """Per-epoch metrics.csv writer produces a file the contracts'
    validator accepts and the reader parses back into the right per-row
    dicts (with correct numeric types on val_* columns)."""
    path = tmp_path / "metrics.csv"
    write_metrics_csv(path, _epoch_records())

    result = validate_metrics(path)
    assert result.ok, result.issues

    rows = load_metrics_csv(path)
    assert len(rows) == 2
    assert rows[0]["epoch"] == 1
    assert rows[0]["val_auroc"] == 0.85
    assert rows[1]["val_auroc"] == 0.87


def test_metrics_csv_null_round_trip(tmp_path: Path) -> None:
    """Non-finite val metrics (NaN AUROC on a single-class val split, etc.)
    must serialize as empty CSV cells and read back as None — distinct
    from "this metric was zero". Important because the writer emits NaN
    silently otherwise."""
    er = EpochRecord(
        epoch=1,
        lr=1e-3,
        train_loss=0.5,
        val_loss=0.45,
        epoch_seconds=12.0,
        val_metrics={"auroc": float("nan"), "accuracy": 0.80},
        val_reliability=[],
    )
    path = tmp_path / "metrics.csv"
    write_metrics_csv(path, [er])

    result = validate_metrics(path)
    assert result.ok, result.issues

    rows = load_metrics_csv(path)
    assert rows[0]["val_auroc"] is None
    assert rows[0]["val_accuracy"] == 0.80


# Per-trace predictions interchange has been retired from records — the
# ClassifierBank format carries them. See test_banks_round_trip.py for
# the prediction-round-trip coverage.


# ---------------------------------------------------------------------------
# hybrid_eval_metrics
# ---------------------------------------------------------------------------


def test_hybrid_eval_metrics_round_trip(tmp_path: Path) -> None:
    """Hybrid (mixed synthetic + IAFDB) eval summary writes a document the
    contracts' validator accepts. Confirms the schema-required `mixed`
    and `iafdb_only` blocks have the right nested shapes."""
    path = tmp_path / "hybrid_eval_metrics.json"
    write_hybrid_eval_metrics(
        path,
        producer={
            "run_id": "test-run-001",
            "model_checkpoint": "checkpoints/v1/best.pt",
        },
        mixed={
            "n": 100,
            "n_positives": 50,
            "n_negatives": 50,
            "metrics": {
                "auroc": 0.85,
                "accuracy": 0.80,
                "precision": 0.81,
                "recall": 0.78,
                "f1": 0.79,
            },
        },
        iafdb_only={
            "n": 40,
            "mean_prob_fibrotic": 0.22,
            "fpr_at_0_5": 0.18,
        },
    )

    result = validate_hybrid_eval_metrics(path)
    assert result.ok, result.issues


# ---------------------------------------------------------------------------
# model_metadata
# ---------------------------------------------------------------------------


def test_model_metadata_round_trip(tmp_path: Path) -> None:
    """Inference-side model_metadata.json writes the six required blocks
    (model_artifact, input, output, preprocessing, decision,
    training_provenance) in shapes the contracts' validator accepts.
    This is what a C++/TensorRT runtime would read alongside the ONNX
    artifact, so getting the shape right is load-bearing."""
    path = tmp_path / "model_metadata.json"
    write_model_metadata(
        path,
        model_artifact={
            "filename": "best.onnx",
            "framework": "onnx",
            "sha256": "a" * 64,
        },
        input_spec={
            "name": "signal",
            "shape": ["?", 1, 512],
            "dtype": "float32",
        },
        output_spec={
            "name": "logit",
            "shape": ["?", 1],
            "dtype": "float32",
            "semantics": "binary_logit",
        },
        preprocessing={
            "expected_fs_hz": 1000.0,
            "expected_trace_samples": 512,
            "bandpass_hz": [30.0, 300.0],
            "normalization": {"scheme": "zscore", "mean": [0.0], "std": [1.0]},
        },
        decision={
            "threshold": 0.5,
            "class_labels": ["healthy", "fibrotic"],
        },
        training_provenance={
            "run_id": "test-run-001",
            "training_bank_path": "data/hybrid_v1.h5",
            "training_bank_schema_version": "1.0",
        },
    )

    result = validate_model_metadata(path)
    assert result.ok, result.issues


# ---------------------------------------------------------------------------
# noise_bank_run_record (JSON sidecar to noise_bank)
# ---------------------------------------------------------------------------


def test_noise_bank_run_record_round_trip_with_provenance(tmp_path: Path) -> None:
    """Build a complete run record (windowing + calibration + selection +
    per_trace_provenance), write it, validate against the contract, read
    it back. The builder shape must agree with what the contracts'
    validator expects — this is the alignment point between producer
    code in iafdb-pipeline (which will call build_noise_bank_run_record)
    and the schema."""
    doc = build_noise_bank_run_record(
        source="iafdb v1.0.0",
        fs_hz=1000.0,
        window_ms=512.0,
        window_samples=512,
        hop_ms=256.0,
        band_hz=[30.0, 300.0],
        calibration_method="r_wave_anchoring",
        calibration_target_qrs_pp_mv=1.0,
        threshold_mode="percentile",
        threshold_value=10.0,
        source_records=["iaf1_afw"],
        description="round-trip test fixture",
        per_trace_provenance={
            "patient_id": ["iaf1", "iaf1"],
            "start_sample": [0, 256],
            "peak_to_peak_mv": [0.05, 0.07],
            "calibration_scalar": [0.0035, 0.0035],
        },
    )
    path = tmp_path / "noise_bank_run_record.json"
    write_noise_bank_run_record(path, doc)

    result = validate_noise_bank_run_record(path)
    assert result.ok, result.issues

    loaded = load_noise_bank_run_record(path)
    assert loaded["schema_version"] == "1.0"
    assert loaded["windowing"]["window_samples"] == 512
    assert loaded["calibration"]["method"] == "r_wave_anchoring"
    assert loaded["selection"]["threshold_mode"] == "percentile"
    # per_trace_provenance must round-trip with array alignment intact
    # — that's the contract with downstream auditing tools.
    assert loaded["per_trace_provenance"]["patient_id"] == ["iaf1", "iaf1"]
    assert loaded["per_trace_provenance"]["start_sample"] == [0, 256]


def test_noise_bank_run_record_round_trip_without_provenance(tmp_path: Path) -> None:
    """Per-trace provenance is OPTIONAL in the schema — smaller test
    fixtures or producers that don't have it can omit it. Confirm the
    builder + writer + validator path still works with no
    per_trace_provenance block."""
    doc = build_noise_bank_run_record(
        source="iafdb v1.0.0",
        fs_hz=1000.0,
        window_ms=512.0,
        window_samples=512,
        hop_ms=256.0,
        band_hz=[30.0, 300.0],
        calibration_method="none",
        calibration_target_qrs_pp_mv=None,
        threshold_mode="percentile",
        threshold_value=10.0,
        source_records=["iaf1_afw"],
    )
    path = tmp_path / "noise_bank_run_record.json"
    write_noise_bank_run_record(path, doc)

    result = validate_noise_bank_run_record(path)
    assert result.ok, result.issues

    loaded = load_noise_bank_run_record(path)
    # When calibration has no notion of a target, target_qrs_pp_mv is
    # null in the record. Producers using calibration_method='none' for
    # raw-signal pretraining banks rely on this.
    assert loaded["calibration"]["target_qrs_pp_mv"] is None
    assert "per_trace_provenance" not in loaded
