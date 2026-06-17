"""Tie ClassifierBank + splits + datasets into ``DataLoader``s.

``build_dataloaders`` is the one call training/eval code makes: hand
it a :class:`ClassifierBank` and the data knobs, and get back
train/val/test loaders plus an ``info`` dict.

Conventions the ClassifierBank must satisfy:

- Every trace's ``label_truth`` must be populated (we're training on
  these; an unlabeled trace is a bug).
- Every trace's ``trace_metadata`` must carry a ``patient_id`` key (the
  patient-aware splitter reads it). The converters in
  :mod:`~myocard_egm_data.banks.converters` populate this.

This module's job is gluing — the upstream policy decisions
(label semantics, label_fn, source bank choice) happen at conversion
time, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..augmentation import TraceTransform
from ..banks import ClassifierBank
from ..splits import patient_aware_split
from .trace_dataset import EGMTraceDataset

_TrainBatch = tuple[torch.Tensor, torch.Tensor]


@dataclass
class LoaderBundle:
    """The loaders plus the metadata the trainer needs."""

    train: DataLoader[_TrainBatch]
    val: DataLoader[_TrainBatch]
    test: DataLoader[_TrainBatch]
    info: dict[str, Any]


def _class_counts(labels: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts, strict=True)}


def build_dataloaders(
    bank: ClassifierBank,
    *,
    input_length: int,
    batch_size: int,
    num_workers: int,
    binary: bool,
    znorm: bool,
    znorm_eps: float,
    augment_train: bool,
    max_gain: float,
    max_shift_frac: float,
    split_fractions: tuple[float, float, float],
    split_seed: int,
    dataset_seed: int,
    pin_memory: bool,
) -> LoaderBundle:
    """Build train/val/test loaders with a patient-aware split.

    No kwarg has a default. The consumer (typically egm-classifier) owns
    the full training-config policy and passes every value explicitly.

    Parameters
    ----------
    bank
        Source data. All traces must have a ``label_truth`` and a
        ``patient_id`` in ``trace_metadata``.
    binary
        Single-logit BCE labels (float ``[1]``) when True; long labels
        for the multi-class head when False.
    augment_train
        Apply gain + time-shift augmentation to the train split only.

    Returns
    -------
    LoaderBundle with ``.train``/``.val``/``.test`` loaders and an
    ``.info`` dict (split sizes, class counts, pos_weight, etc.).
    """
    if bank.n_traces == 0:
        raise ValueError("build_dataloaders: ClassifierBank is empty.")

    # Pull the columnar views the splitter and dataset need.
    signal = bank.signal_array()  # [N, T] float32
    labels = bank.label_truth_array()  # [N] int64
    patient_id = bank.patient_id_array()  # [N] str

    # patient_aware_split needs an integer group key; factorize patient_id.
    _, group_id = np.unique(patient_id, return_inverse=True)
    group_id = group_id.astype(np.int64)

    split = patient_aware_split(group_id, labels, fractions=split_fractions, seed=split_seed)

    label_dtype = torch.float32 if binary else torch.long
    train_tf = TraceTransform(
        input_length=input_length,
        znorm=znorm,
        znorm_eps=znorm_eps,
        augment=augment_train,
        max_gain=max_gain,
        max_shift_frac=max_shift_frac,
    )
    eval_tf = TraceTransform(
        input_length=input_length,
        znorm=znorm,
        znorm_eps=znorm_eps,
        augment=False,
        max_gain=max_gain,
        max_shift_frac=max_shift_frac,
    )

    def make_ds(indices: np.ndarray, transform: TraceTransform) -> EGMTraceDataset:
        return EGMTraceDataset(
            signal=signal,
            labels=labels,
            indices=indices,
            transform=transform,
            label_dtype=label_dtype,
            seed=dataset_seed,
        )

    train_ds = make_ds(split.train, train_tf)
    val_ds = make_ds(split.val, eval_tf)
    test_ds = make_ds(split.test, eval_tf)

    # Explicit kwargs (rather than a packed **common dict) keep the
    # DataLoader call typed; a Mapping with mixed bool/int values gets
    # inferred as dict[str, int] and trips mypy on pin_memory: bool.
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=False,
    )

    # BCE pos_weight = (#neg / #pos) on the train split, to counter the
    # typical fibrosis-class imbalance. 1.0 if a class is absent (degenerate
    # small bank).
    train_labels = labels[split.train]
    counts = _class_counts(train_labels)
    n_pos = counts.get(1, 0)
    n_neg = counts.get(0, 0)
    pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    info: dict[str, Any] = {
        "bank_schema_version": bank.schema_version,
        "n_traces": bank.n_traces,
        "n_samples": bank.n_samples_first,
        "input_length": input_length,
        "n_patients": int(np.unique(group_id).size),
        "split_sizes": {
            "train": int(split.train.size),
            "val": int(split.val.size),
            "test": int(split.test.size),
        },
        "split_patients": {
            "train": int(np.unique(group_id[split.train]).size),
            "val": int(np.unique(group_id[split.val]).size),
            "test": int(np.unique(group_id[split.test]).size),
        },
        "train_class_counts": counts,
        "pos_weight": float(pos_weight),
        "binary": binary,
    }
    return LoaderBundle(train=train_loader, val=val_loader, test=test_loader, info=info)
