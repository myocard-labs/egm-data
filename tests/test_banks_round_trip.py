"""Round-trip the bank writers/readers/converters against the contracts.

End-to-end coverage for each source bank schema:

1. Writer produces a file that the contracts' file-level validator
   accepts.
2. The Pydantic-mode reader loads it back into a typed model.
3. The converter turns the Pydantic model into a ClassifierBank,
   applying a caller-supplied label_fn.

And then a separate round-trip for the ClassifierBank HDF5 format
itself (the egm-data-owned intermediate).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from myocard_egm_contracts.validators import (
    validate_iafdb_bank,
    validate_noise_bank,
    validate_synthetic_bank,
)

from myocard_egm_data.banks import (
    CLASSIFIER_BANK_VERSION,
    ClassifierBank,
    ClassifierBankMetaData,
    ClassifierPrediction,
    ClassifierTrace,
    iafdb_bank_to_classifier,
    load_classifier_bank,
    load_iafdb_bank_as_classifier,
    load_synthetic_bank_as_classifier,
    read_iafdb_bank_hdf5,
    read_noise_bank_hdf5,
    read_synthetic_bank_hdf5,
    synthetic_bank_to_classifier,
    write_classifier_bank,
    write_noise_bank,
)

# ---------------------------------------------------------------------------
# Synthetic bank
# ---------------------------------------------------------------------------


def test_synthetic_bank_validates(synthetic_bank_path: Path) -> None:
    """A synthetic bank written by our writer must pass the contracts'
    file-level validator. If this fails, our writer drifted from the
    schema."""
    result = validate_synthetic_bank(synthetic_bank_path)
    assert result.ok, result.issues


def test_synthetic_bank_to_classifier(synthetic_bank_path: Path, n_samples: int) -> None:
    """Read a synthetic bank into a Pydantic model, convert to
    ClassifierBank with a binary fibrotic-vs-healthy label_fn that
    returns (labels_array, labels_dict), and confirm the shape +
    provenance + label propagation."""
    pyd_bank = read_synthetic_bank_hdf5(synthetic_bank_path)

    def label_fn(b: object) -> tuple[np.ndarray, dict[int, str]]:
        densities = np.asarray(pyd_bank.traces.fibrosis_density)
        return (densities > 0.0).astype(np.int64), {0: "healthy", 1: "fibrotic"}

    cb = synthetic_bank_to_classifier(
        pyd_bank,
        label_fn=label_fn,
        bank_path=synthetic_bank_path,
    )
    assert isinstance(cb, ClassifierBank)
    assert cb.n_traces == 6  # 2 patients x 3 traces
    assert cb.n_samples_first == n_samples
    assert cb.labels == {0: "healthy", 1: "fibrotic"}
    # Patient 0 healthy, patient 1 fibrotic — labels propagated.
    labels = cb.label_truth_array()
    assert labels.tolist() == [0, 0, 0, 1, 1, 1]
    # Source-bank provenance reaches bank_metadata.
    assert len(cb.banks) == 1
    assert cb.banks[0].bank_type == "synthetic"
    assert cb.banks[0].bank_metadata["simulator"] == "finitewave"
    # patient_id propagated as a string in trace_metadata for the splitter.
    assert all("patient_id" in t.trace_metadata for t in cb.traces)


def test_load_synthetic_as_classifier_shortcut(synthetic_bank_path: Path) -> None:
    """The convenience function should chain read + convert and produce
    the same result as the two-step path."""
    cb = load_synthetic_bank_as_classifier(
        synthetic_bank_path,
        label_fn=lambda b: (
            np.asarray([int(d > 0.0) for d in b.traces.fibrosis_density], dtype=np.int64),
            {0: "healthy", 1: "fibrotic"},
        ),
    )
    assert cb.n_traces == 6
    assert cb.banks[0].bank_type == "synthetic"


def test_load_synthetic_without_label_fn(synthetic_bank_path: Path) -> None:
    """Omitting label_fn entirely leaves every trace's label_truth=None
    and an empty labels dict — the pretraining / inference-time mode."""
    cb = load_synthetic_bank_as_classifier(synthetic_bank_path)
    assert all(t.label_truth is None for t in cb.traces)
    assert cb.labels == {}


# ---------------------------------------------------------------------------
# IAFDB bank
# ---------------------------------------------------------------------------


def test_iafdb_bank_validates(iafdb_bank_path: Path) -> None:
    """An IAFDB bank written by our writer must pass the v0.1.2
    contracts validator (which no longer requires a label column)."""
    result = validate_iafdb_bank(iafdb_bank_path)
    assert result.ok, result.issues


def test_iafdb_bank_to_classifier(iafdb_bank_path: Path, n_samples: int) -> None:
    """The IAFDB bank carries no labels. The converter applies a
    caller-supplied label_fn — here a "healthy by selection" policy
    that emits 0 for every trace — and produces a labeled
    ClassifierBank."""
    pyd_bank = read_iafdb_bank_hdf5(iafdb_bank_path)

    def label_fn(b: object) -> tuple[np.ndarray, dict[int, str]]:
        n = len(pyd_bank.traces.signal)
        return np.zeros(n, dtype=np.int64), {0: "healthy", 1: "fibrotic"}

    cb = iafdb_bank_to_classifier(
        pyd_bank,
        label_fn=label_fn,
        bank_path=iafdb_bank_path,
    )
    assert cb.n_traces == 4  # 2 patients x 2 channels
    assert cb.n_samples_first == n_samples
    assert cb.label_truth_array().tolist() == [0, 0, 0, 0]
    assert cb.banks[0].bank_type == "iafdb"
    assert cb.banks[0].bank_metadata["calibration_method"] == "r_wave_anchoring"
    # patient_id in trace_metadata round-trips as Python str (h5py default
    # is bytes; the read helper decodes).
    for t in cb.traces:
        assert isinstance(t.trace_metadata["patient_id"], str)


def test_iafdb_label_fn_returning_none(iafdb_bank_path: Path) -> None:
    """A label_fn returning None signals 'no ground-truth labels';
    every ClassifierTrace ends up with label_truth=None."""
    cb = load_iafdb_bank_as_classifier(iafdb_bank_path, label_fn=lambda b: None)
    assert all(t.label_truth is None for t in cb.traces)
    assert cb.labels == {}


# ---------------------------------------------------------------------------
# ClassifierBank.concat
# ---------------------------------------------------------------------------


def test_classifier_bank_concat_preserves_stable_bank_ids(
    synthetic_bank_path: Path,
    iafdb_bank_path: Path,
) -> None:
    """Merging two ClassifierBanks keeps each source bank's stable bank_id
    (no integer remap); every trace's bank_id still points at its source
    entry by that stable id."""
    syn = load_synthetic_bank_as_classifier(
        synthetic_bank_path,
        label_fn=lambda b: (
            (np.asarray(b.traces.fibrosis_density) > 0.0).astype(np.int64),
            {0: "healthy", 1: "fibrotic"},
        ),
    )
    iaf = load_iafdb_bank_as_classifier(
        iafdb_bank_path,
        label_fn=lambda b: (
            np.zeros(len(b.traces.signal), dtype=np.int64),
            {0: "healthy", 1: "fibrotic"},
        ),
    )
    merged = ClassifierBank.concat([syn, iaf])
    assert merged.n_traces == syn.n_traces + iaf.n_traces
    assert len(merged.banks) == 2
    bank_types = {b.bank_id: b.bank_type for b in merged.banks}
    assert bank_types == {
        "tbank_synthetic_test_2026-06-27": "synthetic",
        "tbank_iafdb_test_2026-06-27": "iafdb",
    }
    # Every trace's bank_id is one of the source banks' stable ids.
    valid_ids = set(bank_types)
    assert all(t.bank_id in valid_ids for t in merged.traces)


def test_classifier_bank_concat_rejects_label_mismatch(
    synthetic_bank_path: Path,
    iafdb_bank_path: Path,
) -> None:
    """Concat with mismatched labels dicts should raise — the locked-in
    rule is "integer labels mean the same thing in every contributing
    bank, or concat is a bug"."""
    import pytest

    syn = load_synthetic_bank_as_classifier(
        synthetic_bank_path,
        label_fn=lambda b: (
            (np.asarray(b.traces.fibrosis_density) > 0.0).astype(np.int64),
            {0: "healthy", 1: "fibrotic"},
        ),
    )
    iaf = load_iafdb_bank_as_classifier(
        iafdb_bank_path,
        label_fn=lambda b: (
            np.zeros(len(b.traces.signal), dtype=np.int64),
            {0: "OTHER_LABEL"},  # deliberately different
        ),
    )
    with pytest.raises(ValueError, match="labels dict at index"):
        ClassifierBank.concat([syn, iaf])


# ---------------------------------------------------------------------------
# ClassifierBank HDF5 round-trip
# ---------------------------------------------------------------------------


def test_classifier_bank_hdf5_round_trip(tmp_path: Path) -> None:
    """Write a ClassifierBank with prediction-laden, label-laden, AND
    label-less traces; read it back; assert every field reaches
    in-memory form. Sentinel masks must keep None distinct from 0."""
    cb = ClassifierBank(
        schema_version=CLASSIFIER_BANK_VERSION,
        banks=[
            ClassifierBankMetaData(
                bank_id="tbank_synthetic_test_2026-06-27",
                bank_type="synthetic",
                bank_path="data/syn.h5",
                bank_metadata={"simulator": "finitewave", "patch_size_mm": 40.0},
            )
        ],
        traces=[
            # Trace 0 — labeled, no prediction yet.
            ClassifierTrace(
                bank_id="tbank_synthetic_test_2026-06-27",
                signal=np.arange(8, dtype=np.float32),
                freq_hz=1000.0,
                amp_type="mv",
                split="train",
                label_truth=0,
                prediction=None,
                trace_metadata={"patient_id": "0", "fibrosis_density": 0.0},
            ),
            # Trace 1 — labeled + predicted.
            ClassifierTrace(
                bank_id="tbank_synthetic_test_2026-06-27",
                signal=np.arange(8, dtype=np.float32) + 10,
                freq_hz=1000.0,
                amp_type="mv",
                split="val",
                label_truth=1,
                prediction=ClassifierPrediction(
                    label_pred=1,
                    label_prob=0.95,
                    pred_logits={0: -2.0, 1: 3.0},
                ),
                trace_metadata={"patient_id": "1", "fibrosis_density": 0.3},
            ),
            # Trace 2 — unlabeled (inference-time data style).
            ClassifierTrace(
                bank_id="tbank_synthetic_test_2026-06-27",
                signal=np.arange(8, dtype=np.float32) + 20,
                freq_hz=1000.0,
                amp_type="mv",
                split=None,
                label_truth=None,
                prediction=None,
                trace_metadata={"patient_id": "2"},
            ),
        ],
        labels={0: "healthy", 1: "fibrotic"},
    )
    path = tmp_path / "cb.h5"
    write_classifier_bank(cb, path)
    loaded = load_classifier_bank(path)

    assert loaded.schema_version == CLASSIFIER_BANK_VERSION
    assert loaded.labels == {0: "healthy", 1: "fibrotic"}
    assert len(loaded.banks) == 1
    assert loaded.banks[0].bank_type == "synthetic"
    assert loaded.banks[0].bank_metadata["simulator"] == "finitewave"
    assert loaded.n_traces == 3
    # Trace 0: labeled with 0 (NOT the sentinel "missing"), no prediction.
    assert loaded.traces[0].label_truth == 0
    assert loaded.traces[0].prediction is None
    assert loaded.traces[0].split == "train"
    # Trace 1: labeled + predicted; logits round-trip with int keys.
    assert loaded.traces[1].label_truth == 1
    pred = loaded.traces[1].prediction
    assert pred is not None
    assert pred.label_pred == 1
    assert abs(pred.label_prob - 0.95) < 1e-9
    assert pred.pred_logits == {0: -2.0, 1: 3.0}
    # Trace 2: unlabeled, no prediction, no split.
    assert loaded.traces[2].label_truth is None
    assert loaded.traces[2].prediction is None
    assert loaded.traces[2].split is None


# ---------------------------------------------------------------------------
# Noise bank — slim schema; no ClassifierBank converter (per Daniel 2026-06-17)
# ---------------------------------------------------------------------------


def test_noise_bank_round_trip_validates(noise_bank_path: Path) -> None:
    """The slim noise_bank writer must produce a file the contracts'
    file-level validator accepts. The schema carries only signal +
    source_record + source_channel per trace, plus schema_version /
    created_utc / bank_id / source / fs_hz at the root."""
    result = validate_noise_bank(noise_bank_path)
    assert result.ok, result.issues


def test_noise_bank_reader_round_trips(noise_bank_path: Path) -> None:
    """Write a noise bank via the egm-data writer, read it back via the
    Pydantic-mode reader, and confirm the per-trace columns round-trip
    untouched. Catches dtype confusion (utf-8 bytes vs str) and any
    silent shape mangling between the two paths."""
    pyd_bank = read_noise_bank_hdf5(noise_bank_path)
    assert pyd_bank.fs_hz == 1000.0
    assert pyd_bank.source == "iafdb v1.0.0"
    # noise_bank 1.1: the stable id now rides on the bank itself, so
    # egm-studio's Noise view no longer has to open the sibling run
    # record just to learn which bank it is looking at (B20).
    assert pyd_bank.bank_id == "nbank_iafdb_test_2026-07-31"
    assert len(pyd_bank.traces.signal) == 4
    assert len(pyd_bank.traces.signal[0]) == 512
    # source_record / source_channel are the audit fields propagated to
    # each hybrid output trace; failing this would make noise-origin
    # debugging impossible downstream.
    assert list(pyd_bank.traces.source_record) == ["iaf1_afw"] * 4
    assert list(pyd_bank.traces.source_channel) == ["CS12", "CS34", "CS12", "CS34"]


def test_noise_bank_overwrite_guard(tmp_path: Path, noise_bank_path: Path) -> None:
    """The writer refuses to clobber an existing file unless overwrite=True.
    Prevents producers from silently destroying a previous extraction
    they meant to keep — the same guard pattern as write_iafdb_bank."""
    import pytest
    from myocard_egm_contracts import noise_bank as noise_bank_models

    pyd_bank = read_noise_bank_hdf5(noise_bank_path)
    target = tmp_path / "second.h5"
    write_noise_bank(pyd_bank, target)
    with pytest.raises(FileExistsError):
        write_noise_bank(pyd_bank, target)
    # overwrite=True succeeds and round-trips cleanly.
    write_noise_bank(pyd_bank, target, overwrite=True)
    reloaded = read_noise_bank_hdf5(target)
    assert reloaded.source == pyd_bank.source
    # Quiet the unused-import warning for type-only reference.
    assert noise_bank_models.NoiseBank is not None


def test_noise_bank_without_bank_id_reads_as_none(tmp_path: Path, noise_bank_path: Path) -> None:
    """A pre-1.1 noise bank carries no bank_id root attr and must still
    read, with bank_id None rather than "".

    ``bank_id`` is optional-in-schema precisely so banks written before
    egm-contracts v0.6.0 stay readable. The distinction matters: an
    empty string would satisfy "a str is present" at every call site and
    then fail the ArtifactId pattern deep inside some later consumer,
    whereas None is explicitly "this bank predates stable ids"."""
    import h5py

    legacy = tmp_path / "legacy_noise.h5"
    legacy.write_bytes(noise_bank_path.read_bytes())
    with h5py.File(legacy, "a") as f:
        del f.attrs["bank_id"]

    pyd_bank = read_noise_bank_hdf5(legacy)
    assert pyd_bank.bank_id is None
    # Everything else still round-trips — dropping the id is not a
    # partial read.
    assert pyd_bank.source == "iafdb v1.0.0"
    assert len(pyd_bank.traces.signal) == 4


def test_write_noise_bank_requires_bank_id(tmp_path: Path, noise_bank_path: Path) -> None:
    """Reading a legacy bank is allowed; writing one back is not.

    This is the "optional-in-schema, required-on-write" convention the
    linkage design applies to every producer bank — the schema cannot
    express it, so the writer enforces it. Without this guard a
    round-trip through egm-data would silently launder a legacy bank
    into a new file that still has no stable id, and nothing downstream
    could reference it."""
    import h5py
    import pytest

    legacy = tmp_path / "legacy_noise.h5"
    legacy.write_bytes(noise_bank_path.read_bytes())
    with h5py.File(legacy, "a") as f:
        del f.attrs["bank_id"]
    pyd_bank = read_noise_bank_hdf5(legacy)
    assert pyd_bank.bank_id is None

    with pytest.raises(ValueError, match="bank_id"):
        write_noise_bank(pyd_bank, tmp_path / "rewritten.h5")
