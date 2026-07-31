"""Round-trip the record writers/readers against the contracts' validators.

Covers training_run_record / training_metrics /
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

import json
from pathlib import Path

from myocard_egm_contracts.schema_info import current_version
from myocard_egm_contracts.validators import (
    validate_egm_class_model_metadata,
    validate_noise_bank_run_record,
    validate_training_metrics,
    validate_training_run_record,
)

from myocard_egm_data.records import (
    EgmClassModelMetadata,
    EpochRecord,
    NoiseBankRunRecord,
    ReliabilityBin,
    TrainingMetricsRow,
    TrainingRunRecord,
    best_epoch,
    build_egm_class_model_metadata,
    build_noise_bank_run_record,
    build_training_run_record,
    load_noise_bank_run_record,
    load_training_metrics,
    load_training_run_record,
    make_epoch_record,
    write_egm_class_model_metadata,
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


def test_make_epoch_record_carries_train_metrics(tmp_path: Path) -> None:
    """train_metrics is split, sanitized, and round-trips (CL-022, P1).

    Without it a validation-only record cannot distinguish an
    overfitting run from a genuinely hard task, which is the whole point
    of the 1.2 addition.
    """
    er = make_epoch_record(
        epoch=1,
        lr=1e-3,
        train_loss=0.5,
        val_loss=0.45,
        epoch_seconds=12.0,
        val_metrics={"auroc": 0.85, "accuracy": 0.80},
        train_metrics={"auroc": 0.99, "accuracy": 0.98},
    )
    assert er.train_metrics is not None
    assert er.train_metrics["auroc"] == 0.99
    # Train metrics must not leak into the val bundle, or divergence
    # would read as zero.
    assert er.val_metrics["auroc"] == 0.85

    record = build_training_run_record(
        config={"model": {"name": "mobilevit_1d"}},
        run_meta={"run_id": "r1"},
        epoch_records=[er],
        select_metric="auroc",
    )
    path = tmp_path / "run.json"
    write_training_run_record(path, record)
    assert validate_training_run_record(path).ok

    reloaded = load_training_run_record(path)
    assert reloaded.epochs[0].train_metrics is not None
    assert reloaded.epochs[0].train_metrics["auroc"] == 0.99


def test_epoch_record_without_train_metrics_still_validates(tmp_path: Path) -> None:
    """Omitting train_metrics is valid — that is what splits the waves.

    The field is optional-in-schema precisely so a producer can adopt
    the 1.2 record before it emits the field (egm-classifier's CLF5
    migration lands a wave before CLF2's emit). A record written in
    between must stay valid, or the two could not ship separately.
    """
    er = make_epoch_record(
        epoch=1,
        lr=1e-3,
        train_loss=0.5,
        val_loss=0.45,
        epoch_seconds=12.0,
        val_metrics={"auroc": 0.85},
    )
    assert er.train_metrics is None

    record = build_training_run_record(
        config={}, run_meta={}, epoch_records=[er], select_metric="auroc"
    )
    path = tmp_path / "run_no_train.json"
    write_training_run_record(path, record)
    assert validate_training_run_record(path).ok
    # Absent, not null — the schema types it as an object, not nullable.
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "train_metrics" not in raw["epochs"][0]


def test_reliability_inside_train_metrics_is_dropped() -> None:
    """A 'reliability' key in train_metrics is discarded, not stored.

    Deliberately asymmetric with the val path: train_reliability bins
    are out of scope for 1.2 (FB-10), so there is nowhere to put them.
    Dropping beats raising because a producer computing one metrics dict
    per split will naturally pass reliability on both, and failing would
    force it to special-case a field it cannot store anyway.
    """
    er = make_epoch_record(
        epoch=1,
        lr=1e-3,
        train_loss=0.5,
        val_loss=0.45,
        epoch_seconds=12.0,
        val_metrics={"auroc": 0.85, "reliability": []},
        train_metrics={
            "auroc": 0.99,
            "reliability": [ReliabilityBin(lo=0.0, hi=1.0, count=4, confidence=0.5, accuracy=0.5)],
        },
    )
    assert er.train_metrics is not None
    assert "reliability" not in er.train_metrics
    assert er.train_metrics["auroc"] == 0.99


def test_best_epoch_ignores_train_metrics() -> None:
    """Selection stays on val_metrics only.

    If train metrics could influence selection, the selected epoch would
    be the most overfit one — exactly backwards.
    """
    records = [
        make_epoch_record(
            epoch=1,
            lr=1e-3,
            train_loss=0.5,
            val_loss=0.45,
            epoch_seconds=1.0,
            val_metrics={"auroc": 0.90},
            train_metrics={"auroc": 0.10},
        ),
        make_epoch_record(
            epoch=2,
            lr=1e-3,
            train_loss=0.1,
            val_loss=0.60,
            epoch_seconds=1.0,
            val_metrics={"auroc": 0.70},
            train_metrics={"auroc": 0.99},
        ),
    ]
    assert best_epoch(records, "auroc").epoch == 1


def test_held_out_test_reliability_is_coerced_like_val(tmp_path: Path) -> None:
    """Test bins accept the same shapes as val bins (B18 parity).

    Before this, val bins went through the coercion path while test bins
    were passed through raw — so a plain mapping that worked for val
    failed for test. A producer should be able to hand both splits'
    metrics in one shape.
    """
    record = build_training_run_record(
        config={},
        run_meta={},
        epoch_records=_epoch_records(),
        select_metric="auroc",
        test_loss=0.41,
        test_metrics={
            "auroc": 0.86,
            # A plain mapping, not a ReliabilityBin.
            "reliability": [
                {"lo": 0.0, "hi": 1.0, "count": 20, "confidence": 0.5, "accuracy": 0.55}
            ],
        },
    )
    assert record.test is not None
    assert isinstance(record.test.reliability[0], ReliabilityBin)
    assert record.test.reliability[0].count == 20

    path = tmp_path / "run_with_test.json"
    write_training_run_record(path, record)
    assert validate_training_run_record(path).ok


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
    assert loaded.schema_version.value == current_version("training_run_record")
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


def test_training_metrics_carries_train_columns(tmp_path: Path) -> None:
    """The six train_* columns are written and read back (P1, CL-037).

    metrics.csv is the file a human opens; the run.json carries the same
    numbers but nobody scans JSON for a divergence trend.
    """
    records = [
        make_epoch_record(
            epoch=1,
            lr=1e-3,
            train_loss=0.5,
            val_loss=0.45,
            epoch_seconds=12.0,
            val_metrics={"auroc": 0.85, "accuracy": 0.80, "f1": 0.79, "ece": 0.05},
            train_metrics={"auroc": 0.99, "accuracy": 0.97, "f1": 0.98, "ece": 0.01},
        )
    ]
    path = tmp_path / "metrics.csv"
    write_training_metrics(path, records)
    assert validate_training_metrics(path).ok

    rows = load_training_metrics(path)
    assert rows[0].train_auroc == 0.99
    assert rows[0].val_auroc == 0.85
    # A metric absent from both dicts stays empty rather than defaulting.
    assert rows[0].train_precision is None
    assert rows[0].val_precision is None


def test_training_metrics_column_order_pairs_the_splits(tmp_path: Path) -> None:
    """The header pairs train_* with val_* rather than appending train.

    The whole reason for carrying both splits is reading their
    divergence, and that only reads clearly when the pairs are adjacent
    — appending the train block at the end puts eight columns between
    train_auroc and val_auroc in a spreadsheet. Asserting on the raw
    header, since the typed model cannot express order.
    """
    path = tmp_path / "metrics.csv"
    write_training_metrics(path, _epoch_records())

    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header[:4] == ["epoch", "lr", "train_loss", "train_auroc"]
    assert header[-1] == "epoch_seconds"
    # Every train_* column precedes every val_* column's twin, in the
    # same relative order.
    train_cols = [
        c[len("train_") :] for c in header if c.startswith("train_") and c != "train_loss"
    ]
    val_cols = [c[len("val_") :] for c in header if c.startswith("val_") and c != "val_loss"]
    assert train_cols == val_cols


def test_training_metrics_without_train_metrics_writes_empty_cells(tmp_path: Path) -> None:
    """A record with no train_metrics yields empty cells, not missing columns.

    This is the CLF5-migration shape: the 1.2 record is adopted a wave
    before the emit. The header has to stay stable across that gap or a
    consumer plotting the file would see the schema change mid-phase.
    """
    records = [
        make_epoch_record(
            epoch=1,
            lr=1e-3,
            train_loss=0.5,
            val_loss=0.45,
            epoch_seconds=12.0,
            val_metrics={"auroc": 0.85},
        )
    ]
    path = tmp_path / "metrics_no_train.csv"
    write_training_metrics(path, records)
    assert validate_training_metrics(path).ok

    lines = path.read_text(encoding="utf-8").splitlines()
    assert "train_auroc" in lines[0].split(",")
    rows = load_training_metrics(path)
    assert rows[0].train_auroc is None
    assert rows[0].val_auroc == 0.85
    # train_loss is a separate, always-required column and is unaffected.
    assert rows[0].train_loss == 0.5


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
    assert loaded.schema_version.value == current_version("noise_bank_run_record")
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
