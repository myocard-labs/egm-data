"""Cross-artifact-linkage additions in egm-data (v0.4.0).

Covers: producer-bank ``bank_id`` surfacing + write-time enforcement, the
converter carrying the source bank's stable id forward, the ClassifierBank
stable ``id`` (validation + round-trip + legacy back-compat), record-side id
surfacing, and the ``phases/`` subpackage round-trips.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from myocard_egm_data.banks.classifier_bank import (
    ClassifierBank,
    ClassifierBankMetaData,
    ClassifierTrace,
)
from myocard_egm_data.banks.classifier_bank_io import load_classifier_bank, write_classifier_bank
from myocard_egm_data.banks.converters import synthetic_bank_to_classifier
from myocard_egm_data.banks.readers import read_iafdb_bank_hdf5, read_synthetic_bank_hdf5
from myocard_egm_data.banks.writers import write_synthetic_bank
from myocard_egm_data.phases import (
    FigureSpec,
    Observation,
    PhaseManifest,
    load_figure_spec,
    load_observation,
    load_phase_manifest,
    write_figure_spec,
    write_observation,
    write_phase_manifest,
)
from myocard_egm_data.records import (
    build_training_run_record,
    load_training_run_record,
    make_epoch_record,
    write_training_run_record,
)

# ---------------------------------------------------------------------------
# Producer-bank bank_id surfacing + write enforcement
# ---------------------------------------------------------------------------


def test_synthetic_bank_id_round_trips(synthetic_bank_path: Path) -> None:
    bank = read_synthetic_bank_hdf5(synthetic_bank_path)
    assert bank.bank_id == "tbank_synthetic_test_2026-06-27"


def test_iafdb_bank_id_round_trips(iafdb_bank_path: Path) -> None:
    bank = read_iafdb_bank_hdf5(iafdb_bank_path)
    assert bank.bank_id == "tbank_iafdb_test_2026-06-27"


def test_legacy_synthetic_bank_without_bank_id_reads_as_none(synthetic_bank_path: Path) -> None:
    """A bank written before v0.5.0 has no bank_id attr; it must still read,
    with bank_id=None (the optional-for-legacy decision)."""
    with h5py.File(synthetic_bank_path, "r+") as f:
        del f.attrs["bank_id"]
    bank = read_synthetic_bank_hdf5(synthetic_bank_path)
    assert bank.bank_id is None


def test_write_synthetic_bank_requires_bank_id(synthetic_bank_path: Path, tmp_path: Path) -> None:
    bank = read_synthetic_bank_hdf5(synthetic_bank_path)
    no_id = bank.model_copy(update={"bank_id": None})
    with pytest.raises(ValueError, match="bank_id"):
        write_synthetic_bank(no_id, tmp_path / "no_id.h5")


def test_converter_sets_stable_bank_id(synthetic_bank_path: Path) -> None:
    """The converted ClassifierBank's source entry + every trace are keyed by
    the source bank's stable bank_id (no integer index / source_bank_id)."""
    bank = read_synthetic_bank_hdf5(synthetic_bank_path)
    cbank = synthetic_bank_to_classifier(bank, label_fn=None, bank_path=synthetic_bank_path)
    assert cbank.banks[0].bank_id == "tbank_synthetic_test_2026-06-27"
    assert all(t.bank_id == "tbank_synthetic_test_2026-06-27" for t in cbank.traces)


# ---------------------------------------------------------------------------
# ClassifierBank stable id
# ---------------------------------------------------------------------------


def _one_trace_bank(artifact_id: str | None) -> ClassifierBank:
    tr = ClassifierTrace(
        bank_id="tbank_test_2026-06-27",
        signal=np.zeros(8, dtype=np.float32),
        freq_hz=1000.0,
        amp_type="mv",
        split=None,
        label_truth=1,
        prediction=None,
        trace_metadata={"patient_id": "p0"},
    )
    meta = ClassifierBankMetaData(
        bank_id="tbank_test_2026-06-27", bank_type="iafdb", bank_path="x.h5", bank_metadata={}
    )
    return ClassifierBank(
        id=artifact_id, banks=[meta], traces=[tr], labels={0: "healthy", 1: "fibrotic"}
    )


def test_classifier_bank_id_round_trips(tmp_path: Path) -> None:
    # `tbank_`, not `upred_`: the fixture's trace carries a label_truth
    # and no prediction, which makes it a training bank. The id here was
    # `upred_` until the S11 content check (B16) rejected it — a fair
    # demonstration of the mislabelling that check exists to prevent.
    bank = _one_trace_bank("tbank_iafdb_v1_5_2026-06-27")
    write_classifier_bank(bank, tmp_path / "c.classifier.h5")
    loaded = load_classifier_bank(tmp_path / "c.classifier.h5")
    assert loaded.id == "tbank_iafdb_v1_5_2026-06-27"
    assert loaded.schema_version == "0.2"


def test_classifier_bank_id_optional_round_trips(tmp_path: Path) -> None:
    bank = _one_trace_bank(None)
    write_classifier_bank(bank, tmp_path / "c.classifier.h5")
    loaded = load_classifier_bank(tmp_path / "c.classifier.h5")
    assert loaded.id is None


def test_classifier_bank_rejects_malformed_id() -> None:
    with pytest.raises(ValueError, match="artifact id"):
        _one_trace_bank("NOT-a-valid-id")


# ---------------------------------------------------------------------------
# Record-side id surfacing
# ---------------------------------------------------------------------------


def test_training_run_record_surfaces_ids(tmp_path: Path) -> None:
    ep = make_epoch_record(
        epoch=1,
        lr=1e-3,
        train_loss=0.5,
        val_loss=0.4,
        epoch_seconds=1.0,
        val_metrics={"auroc": 0.9},
    )
    record = build_training_run_record(
        config={"model": {}},
        run_meta={"run_id": "run_v1_5_courtemanche_2026-06-27"},
        epoch_records=[ep],
        select_metric="auroc",
        run_id="run_v1_5_courtemanche_2026-06-27",
        trained_on_bank_id="tbank_synthetic_courtemanche_v1_5_2026-06-27",
        produced_model_id="model_egm_classifier_v1_5_2026-06-27",
    )
    write_training_run_record(tmp_path / "run.json", record)
    loaded = load_training_run_record(tmp_path / "run.json")
    assert loaded.run_id == "run_v1_5_courtemanche_2026-06-27"
    assert loaded.trained_on_bank_id == "tbank_synthetic_courtemanche_v1_5_2026-06-27"
    assert loaded.produced_model_id == "model_egm_classifier_v1_5_2026-06-27"


# ---------------------------------------------------------------------------
# phases/ round-trips
# ---------------------------------------------------------------------------


def test_phase_manifest_round_trip(tmp_path: Path) -> None:
    manifest = PhaseManifest.model_validate(
        {
            "schema_version": "1",
            "phase": 1.5,
            "status": "in_progress",
            "egm_banks": [
                {
                    "id": "tbank_synthetic_courtemanche_v1_5_2026-06-27",
                    "path": "banks/tbank.h5",
                    "produced_by_package": "synthetic-egm-pipeline",
                    "produced_by_version": "v0.2.0",
                },
                {
                    "id": "upred_iafdb_v1_5_2026-06-27",
                    "path": "preds/upred.h5",
                    "produced_by_package": "egm-classifier",
                    "produced_by_version": "v0.2.0",
                    "model": "model_egm_classifier_v1_5_2026-06-27",
                    "source_bank": "tbank_iafdb_v1_2026-06-15",
                },
            ],
            "noise_banks": [
                {
                    "id": "nbank_iafdb_2026-06-15",
                    "path": "banks/nbank.h5",
                    "produced_by_package": "iafdb-pipeline",
                    "produced_by_version": "v0.2.0",
                }
            ],
        }
    )
    write_phase_manifest(tmp_path / "manifest.json", manifest)
    loaded = load_phase_manifest(tmp_path / "manifest.json")
    assert loaded.model_dump() == manifest.model_dump()
    assert loaded.egm_banks is not None and loaded.noise_banks is not None
    assert len(loaded.egm_banks) == 2
    assert len(loaded.noise_banks) == 1


def test_observation_round_trip(tmp_path: Path) -> None:
    obs = Observation.model_validate(
        {
            "schema_version": "1",
            "id": "obs_courtemanche_high_entropy_tail_2026-06-27",
            "date": "2026-06-27",
            "title": "high-entropy tail",
            "description": "long tail above 1.5",
            "traces": [{"bank": "tbank_synthetic_courtemanche_v1_5_2026-06-27", "index": 1247}],
        }
    )
    write_observation(tmp_path / "obs.json", obs)
    loaded = load_observation(tmp_path / "obs.json")
    assert loaded.model_dump() == obs.model_dump()
    assert loaded.description == "long tail above 1.5"


def test_figure_spec_round_trip(tmp_path: Path) -> None:
    spec = FigureSpec.model_validate(
        {
            "schema_version": "1",
            "id": "fig_feature_distributions_synth_vs_iafdb",
            "description": "overlay of per-feature distributions",
            "recipe": "feature-distribution-overlay",
            "inputs": {
                "groups": [
                    {"name": "Synth", "bank_id": "tbank_synthetic_courtemanche_v1_5_2026-06-27"}
                ]
            },
            "output": {"format": "pdf", "path": "../paper/f.pdf"},
        }
    )
    write_figure_spec(tmp_path / "fig.json", spec)
    loaded = load_figure_spec(tmp_path / "fig.json")
    assert loaded.model_dump() == spec.model_dump()
