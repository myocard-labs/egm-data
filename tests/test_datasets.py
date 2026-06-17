"""Torch dataset wrapper tests.

The fixture banks are intentionally too small for a meaningful
patient-aware split (the three-way fraction allocation would put zero
patients into val/test for two-patient banks), so we exercise
``EGMTraceDataset`` directly and build a bigger ClassifierBank for the
``build_dataloaders`` smoke test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from myocard_egm_data.augmentation import TraceTransform  # noqa: E402
from myocard_egm_data.banks import (  # noqa: E402
    ClassifierBank,
    ClassifierBankMetaData,
    ClassifierTrace,
    load_synthetic_bank_as_classifier,
)
from myocard_egm_data.datasets import EGMTraceDataset, build_dataloaders  # noqa: E402


def test_dataset_returns_expected_shapes(synthetic_bank_path: Path, n_samples: int) -> None:
    """A single __getitem__ call returns (signal, label) with the right
    torch shapes/dtypes: signal is [1, T] float32 (one channel), label
    is [1] float32 (BCE single-logit head). Sanity-check before the
    model sees a batch."""
    cb = load_synthetic_bank_as_classifier(
        synthetic_bank_path,
        label_fn=lambda b: (
            (np.asarray(b.traces.fibrosis_density) > 0.0).astype(np.int64),
            {0: "healthy", 1: "fibrotic"},
        ),
    )
    signal = cb.signal_array()
    labels = cb.label_truth_array()
    tf = TraceTransform(
        input_length=n_samples,
        znorm=True,
        znorm_eps=1e-6,
        augment=False,
        max_gain=0.0,
        max_shift_frac=0.0,
    )
    ds = EGMTraceDataset(
        signal=signal,
        labels=labels,
        indices=np.arange(cb.n_traces),
        transform=tf,
        label_dtype=torch.float32,
    )
    signal_t, label_t = ds[0]
    assert signal_t.shape == (1, n_samples)
    assert label_t.shape == (1,)
    assert label_t.dtype == torch.float32


def _build_bigger_classifier_bank(n_samples: int, fs_hz: float) -> ClassifierBank:
    """Construct a ClassifierBank with enough patients to actually split.

    8 patients x 4 traces each. 4 healthy, 4 fibrotic. The default
    0.8/0.1/0.1 fractions put 6 train / 1 val / 1 test patients —
    enough to hit every code path in build_dataloaders.
    """
    rng = np.random.default_rng(0)
    traces: list[ClassifierTrace] = []
    for patient in range(8):
        label = 0 if patient < 4 else 1
        for pair_index in range(4):
            traces.append(
                ClassifierTrace(
                    bank_id=0,
                    signal=rng.standard_normal(n_samples).astype(np.float32),
                    freq_hz=fs_hz,
                    amp_type="mv",
                    split=None,
                    label_truth=label,
                    prediction=None,
                    trace_metadata={
                        "patient_id": str(patient),
                        "pair_index": pair_index,
                    },
                )
            )
    return ClassifierBank(
        banks=[
            ClassifierBankMetaData(
                bank_id=0,
                bank_type="synthetic",
                bank_path="test://memory",
                bank_metadata={"simulator": "finitewave"},
            )
        ],
        traces=traces,
        labels={0: "healthy", 1: "fibrotic"},
    )


def test_build_dataloaders_smoke(fs_hz: float, n_samples: int) -> None:
    """End-to-end: ClassifierBank -> patient-aware split -> three
    DataLoaders + an info dict. The test supplies all kwargs explicitly
    (egm-data ships no defaults). Iterate one batch through each loader.
    Catches breakage in the gluing layer that wires
    ClassifierBank/splits/datasets together."""
    bank = _build_bigger_classifier_bank(n_samples=n_samples, fs_hz=fs_hz)
    bundle = build_dataloaders(
        bank,
        input_length=n_samples,
        batch_size=4,
        num_workers=0,
        binary=True,
        znorm=True,
        znorm_eps=1e-6,
        augment_train=False,
        max_gain=0.0,
        max_shift_frac=0.0,
        split_fractions=(0.8, 0.1, 0.1),
        split_seed=0,
        dataset_seed=0,
        pin_memory=False,
    )
    assert bundle.info["n_traces"] == 32
    assert bundle.info["n_patients"] == 8
    sizes = bundle.info["split_sizes"]
    assert sizes["train"] + sizes["val"] + sizes["test"] == 32
    # Pull a single batch through each loader.
    for loader in (bundle.train, bundle.val, bundle.test):
        for x, y in loader:
            assert x.shape[-1] == n_samples
            assert x.dtype == torch.float32
            assert y.dtype == torch.float32
            break
