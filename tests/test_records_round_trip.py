"""Round-trip the record writers/readers against the contracts' validators.

Covers training_run_record / training_metrics / hybrid_eval_metrics /
egm_class_model_metadata / noise_bank_run_record. Per-trace predictions
live inside ClassifierBank now — covered by test_banks_round_trip.py.

Every build_* helper returns a typed Pydantic model from
myocard-egm-contracts; every write_* takes a model; every load_*
returns a model. These tests assert (a) the build helper produces a
valid instance, (b) the writer emits a file the contracts' validator
accepts, and (c) the reader returns a typed model with the right
field values.
"""

from __future__ import annotations

from pathlib import Path

from myocard_egm_contracts.validators import (
    validate_egm_class_model_metadata,
    validate_hybrid_eval_metrics,
    validate_noise_bank_run_record,
    validate_training_metrics,
    validate_training_run_record,
)

from myocard_egm_data.records import (
    EgmClassModelMetadata,
    EpochRecord,
    HybridEvalMetrics,
    NoiseBankRunRecord,
    ReliabilityBin,
    TrainingMetricsRow,
    TrainingRunRecord,
    build_egm_class_model_metadata,
    build_hybrid_eval_metrics,
    build_noise_bank_run_record,
    build_training_run_record,
    load_noise_bank_run_record,
    load_training_metrics,
    load_training_run_record,
    make_epoch_record,
    write_egm_class_model_metadata,
    write_hybrid_eval_metrics,
    write_noise_bank_run_record,
    write_training_metrics,
    write_training_run_record,
)


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
# training_run_record
# ---------------------------------------------------------------------------


def test_training_run_record_round_trips(tmp_path: Path) -> None:
    """Build a complete run.json (config + epochs + best + test block),
    write it, validate against the contract, read it back as a typed
    TrainingRunRecord. Also checks that ``build_training_run_record`` picks epoch 2 over
    epoch 1 because val_auroc is higher there."""
    record = build_training_run_record(
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
    assert isinstance(record, TrainingRunRecord)
    assert record.best.epoch == 2  # epoch 2 has higher val_auroc

    path = tmp_path / "run.json"
    write_training_run_record(path, record)

    result = validate_training_run_record(path)
    assert result.ok, result.issues

    loaded = load_training_run_record(path)
    assert isinstance(loaded, TrainingRunRecord)
    assert loaded.schema_version.value == "1.0"
    assert loaded.best.epoch == 2
    assert len(loaded.epochs) == 2


def test_make_epoch_record_accepts_three_reliability_shapes() -> None:
    """``make_epoch_record`` is the trainer-facing smart constructor.
    It must accept reliability bins as (a) typed Pydantic
    :class:`ReliabilityBin`, (b) plain dicts with the five fields, or
    (c) any duck-typed object with the five attributes. All three end
    up as the same typed list inside the resulting :class:`EpochRecord`.
    Also confirms NaN scalar values in ``val_metrics`` are sanitized
    to ``None``."""

    class _DuckBin:
        def __init__(self, lo: float, hi: float) -> None:
            self.lo = lo
            self.hi = hi
            self.count = 0
            self.confidence = 0.0
            self.accuracy = 0.0

    val_metrics = {
        "auroc": float("nan"),  # sanitized to None
        "accuracy": 0.8,
        "reliability": [
            ReliabilityBin(lo=0.0, hi=0.5, count=10, confidence=0.25, accuracy=0.20),
            {"lo": 0.5, "hi": 0.75, "count": 5, "confidence": 0.6, "accuracy": 0.7},
            _DuckBin(0.75, 1.0),
        ],
    }
    rec = make_epoch_record(
        epoch=1,
        lr=1e-3,
        train_loss=0.5,
        val_loss=0.45,
        epoch_seconds=12.0,
        val_metrics=val_metrics,
    )
    assert isinstance(rec, EpochRecord)
    # All three bins are now typed ReliabilityBin instances.
    assert len(rec.val_reliability) == 3
    assert all(isinstance(b, ReliabilityBin) for b in rec.val_reliability)
    assert rec.val_reliability[0].count == 10
    assert rec.val_reliability[1].confidence == 0.6
    assert rec.val_reliability[2].lo == 0.75
    # The reliability key is split out of val_metrics; NaN goes to None.
    assert "reliability" not in rec.val_metrics
    assert rec.val_metrics["auroc"] is None
    assert rec.val_metrics["accuracy"] == 0.8


# ---------------------------------------------------------------------------
# training_metrics CSV
# ---------------------------------------------------------------------------


def test_training_metrics_round_trips(tmp_path: Path) -> None:
    """Per-epoch metrics.csv writer produces a file the contracts'
    validator accepts and the reader parses back into typed TrainingMetricsRow
    instances with the right numeric types on val_* columns."""
    path = tmp_path / "metrics.csv"
    write_training_metrics(path, _epoch_records())

    result = validate_training_metrics(path)
    assert result.ok, result.issues

    rows = load_training_metrics(path)
    assert len(rows) == 2
    assert isinstance(rows[0], TrainingMetricsRow)
    assert rows[0].epoch == 1
    assert rows[0].val_auroc == 0.85
    assert rows[1].val_auroc == 0.87


def test_training_metrics_null_round_trip(tmp_path: Path) -> None:
    """Non-finite val metrics (NaN AUROC on a single-class val split, etc.)
    must serialize as empty CSV cells and read back as None — distinct
    from "this metric was zero". The writer's _csv_cell turns
    non-finite floats into empty strings; the reader's _coerce_row
    turns empty cells back into None."""
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
    write_training_metrics(path, [er])

    result = validate_training_metrics(path)
    assert result.ok, result.issues

    rows = load_training_metrics(path)
    assert rows[0].val_auroc is None
    assert rows[0].val_accuracy == 0.80


# Per-trace predictions interchange has been retired from records — the
# ClassifierBank format carries them. See test_banks_round_trip.py for
# the prediction-round-trip coverage.


# ---------------------------------------------------------------------------
# hybrid_eval_metrics
# ---------------------------------------------------------------------------


def test_hybrid_eval_metrics_round_trip(tmp_path: Path) -> None:
    """Hybrid (mixed synthetic + IAFDB) eval summary builds a typed
    HybridEvalMetrics, writes a document the contracts' validator
    accepts. Confirms the schema-required ``mixed`` and ``iafdb_only``
    blocks have the right nested shapes when passed as dicts (Pydantic
    auto-validates them into the typed sub-models)."""
    record = build_hybrid_eval_metrics(
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
    assert isinstance(record, HybridEvalMetrics)

    path = tmp_path / "hybrid_eval_metrics.json"
    write_hybrid_eval_metrics(path, record)

    result = validate_hybrid_eval_metrics(path)
    assert result.ok, result.issues


# ---------------------------------------------------------------------------
# egm_class_model_metadata
# ---------------------------------------------------------------------------


def test_egm_class_model_metadata_round_trip(tmp_path: Path) -> None:
    """Inference-side model_metadata.json builds a typed EgmClassModelMetadata,
    writes the six required blocks (model_artifact, input, output,
    preprocessing, decision, training_provenance) in shapes the
    contracts' validator accepts. This is what a C++/TensorRT runtime
    would read alongside the ONNX artifact, so getting the shape right
    is load-bearing."""
    record = build_egm_class_model_metadata(
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
            "normalization": {"scheme": "zscore"},
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
    assert isinstance(record, EgmClassModelMetadata)

    path = tmp_path / "model_metadata.json"
    write_egm_class_model_metadata(path, record)

    result = validate_egm_class_model_metadata(path)
    assert result.ok, result.issues


# ---------------------------------------------------------------------------
# noise_bank_run_record (JSON sidecar to noise_bank)
# ---------------------------------------------------------------------------


def test_noise_bank_run_record_round_trip_with_provenance(tmp_path: Path) -> None:
    """Build a complete run record (windowing + calibration + selection +
    per_trace_provenance) as a typed NoiseBankRunRecord, write it,
    validate against the contract, read it back. The builder shape must
    agree with what the contracts' validator expects — this is the
    alignment point between producer code in iafdb-pipeline (which calls
    build_noise_bank_run_record) and the schema."""
    record = build_noise_bank_run_record(
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
    assert isinstance(record, NoiseBankRunRecord)

    path = tmp_path / "noise_bank_run_record.json"
    write_noise_bank_run_record(path, record)

    result = validate_noise_bank_run_record(path)
    assert result.ok, result.issues

    loaded = load_noise_bank_run_record(path)
    assert isinstance(loaded, NoiseBankRunRecord)
    assert loaded.schema_version.value == "1.0"
    assert loaded.windowing.window_samples == 512
    assert loaded.calibration.method == "r_wave_anchoring"
    assert loaded.selection.threshold_mode.value == "percentile"
    # per_trace_provenance must round-trip with array alignment intact
    # — that's the contract with downstream auditing tools.
    assert loaded.per_trace_provenance is not None
    assert loaded.per_trace_provenance.patient_id == ["iaf1", "iaf1"]
    # Pydantic wraps the int / float items in RootModel; .root pulls
    # the underlying numeric value out.
    assert [s.root for s in loaded.per_trace_provenance.start_sample] == [0, 256]


def test_noise_bank_run_record_round_trip_without_provenance(tmp_path: Path) -> None:
    """Per-trace provenance is OPTIONAL in the schema — smaller test
    fixtures or producers that don't have it can omit it. Confirm the
    builder + writer + validator + loader path still works with no
    per_trace_provenance block."""
    record = build_noise_bank_run_record(
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
    write_noise_bank_run_record(path, record)

    result = validate_noise_bank_run_record(path)
    assert result.ok, result.issues

    loaded = load_noise_bank_run_record(path)
    # When calibration has no notion of a target, target_qrs_pp_mv is
    # null in the record. Producers using calibration_method='none' for
    # raw-signal pretraining banks rely on this.
    assert loaded.calibration.target_qrs_pp_mv is None
    assert loaded.per_trace_provenance is None
