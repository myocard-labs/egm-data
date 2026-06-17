"""Patient-aware train/val/test splitting.

The cardinal rule: every trace from one ``simulation_id`` (the patient
analog — one simulator substrate realization, or one IAFDB recording for
the real-data bank) lands entirely in one split. Random per-trace
splitting lets the model see the same fibrosis-pattern realization in
train and val, which inflates val metrics — a failure mode that has
burned multiple cardiology-ML papers. Splitting by patient closes that
leak.

We additionally *stratify* by per-patient label so the (typically
imbalanced) class ratio is preserved across the three splits. Without
this, a small dataset can land all healthy patients in one split by
chance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitIndices:
    """Row indices into the bank arrays for each split."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"train": self.train, "val": self.val, "test": self.test}


def _patient_label(simulation_id: np.ndarray, labels: np.ndarray, pid: int) -> int:
    """The label shared by every trace of a patient (asserts consistency)."""
    pid_labels = labels[simulation_id == pid]
    uniq = np.unique(pid_labels)
    if uniq.size != 1:
        raise ValueError(
            f"simulation_id={pid} has mixed labels {uniq.tolist()}; the "
            "binary label is supposed to be constant within a patient."
        )
    return int(uniq[0])


def _allocate(n_patients: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    """Turn fractions into integer counts that sum to ``n_patients``.

    Train takes the rounded share; val and test split the remainder with
    test getting the leftover, so tiny patient counts still place at least
    what's available rather than silently dropping rows.
    """
    f_train, f_val, _ = fractions
    n_train = round(n_patients * f_train)
    n_val = round(n_patients * f_val)
    n_train = min(n_train, n_patients)
    n_val = min(n_val, n_patients - n_train)
    n_test = n_patients - n_train - n_val
    return n_train, n_val, n_test


def patient_aware_split(
    simulation_id: np.ndarray,
    labels: np.ndarray,
    fractions: tuple[float, float, float],
    seed: int,
) -> SplitIndices:
    """Split trace rows into train/val/test by patient, stratified by label.

    Parameters
    ----------
    simulation_id
        ``[N]`` patient id per trace.
    labels
        ``[N]`` binary label per trace (must be constant within a patient).
    fractions
        ``(train, val, test)`` patient fractions; must sum to ~1.
    seed
        RNG seed for the patient shuffle (reproducible splits).

    Returns
    -------
    SplitIndices of row indices into the original arrays.
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {fractions} (sum {sum(fractions)}).")
    rng = np.random.default_rng(seed)

    patients = np.unique(simulation_id)
    by_label: dict[int, list[int]] = {}
    for pid in patients:
        lbl = _patient_label(simulation_id, labels, int(pid))
        by_label.setdefault(lbl, []).append(int(pid))

    train_pids: list[int] = []
    val_pids: list[int] = []
    test_pids: list[int] = []
    for lbl in sorted(by_label):
        group = np.array(by_label[lbl])
        rng.shuffle(group)
        n_train, n_val, _n_test = _allocate(group.size, fractions)
        train_pids.extend(group[:n_train].tolist())
        val_pids.extend(group[n_train : n_train + n_val].tolist())
        test_pids.extend(group[n_train + n_val :].tolist())

    train_set, val_set, test_set = set(train_pids), set(val_pids), set(test_pids)
    assert not (train_set & val_set), "patient leak: train ∩ val"
    assert not (train_set & test_set), "patient leak: train ∩ test"
    assert not (val_set & test_set), "patient leak: val ∩ test"

    def rows_for(pid_set: set[int]) -> np.ndarray:
        mask = np.isin(simulation_id, list(pid_set))
        return np.nonzero(mask)[0]

    return SplitIndices(
        train=rows_for(train_set),
        val=rows_for(val_set),
        test=rows_for(test_set),
    )
