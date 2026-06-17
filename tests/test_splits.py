"""Tests for patient-aware splitting."""

from __future__ import annotations

import numpy as np

from myocard_egm_data.splits import patient_aware_split


def test_no_patient_leak_between_splits() -> None:
    """The cardinal rule: every trace from one patient lands entirely in
    one split. Build a 10-patient dataset, split it, assert the three
    patient sets are disjoint and their union is the full patient set.
    A failure here means the patient-aware guarantee is broken (which
    silently inflates val/test metrics — the exact failure mode this
    module exists to prevent)."""
    # 10 patients * 5 traces each. 4 fibrotic patients, 6 healthy.
    sim_id = np.repeat(np.arange(10), 5).astype(np.int64)
    fibrotic_patients = {0, 1, 2, 3}
    labels = np.array([1 if sid in fibrotic_patients else 0 for sid in sim_id], dtype=np.int64)

    split = patient_aware_split(sim_id, labels, fractions=(0.6, 0.2, 0.2), seed=42)

    train_pids = set(sim_id[split.train].tolist())
    val_pids = set(sim_id[split.val].tolist())
    test_pids = set(sim_id[split.test].tolist())

    assert not (train_pids & val_pids)
    assert not (train_pids & test_pids)
    assert not (val_pids & test_pids)
    # Every patient ends up in exactly one split.
    assert train_pids | val_pids | test_pids == set(range(10))


def test_split_is_reproducible() -> None:
    """Same input, same seed -> identical row indices in each split.
    Reproducibility matters for re-running experiments and for the
    "compare two runs" case where any non-determinism in the split would
    invalidate the comparison."""
    sim_id = np.repeat(np.arange(8), 4).astype(np.int64)
    labels = np.zeros_like(sim_id)
    labels[sim_id >= 4] = 1

    a = patient_aware_split(sim_id, labels, fractions=(0.5, 0.25, 0.25), seed=7)
    b = patient_aware_split(sim_id, labels, fractions=(0.5, 0.25, 0.25), seed=7)
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.val, b.val)
    assert np.array_equal(a.test, b.test)


# NOTE: `labels_from_density` was removed from egm-data — label extraction
# is now the consumer's job (passed into build_dataloaders as label_fn).
# The test for the threshold semantics moves with the helper to wherever
# egm-classifier ends up declaring its label_fn.
