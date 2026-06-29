"""Unit tests for ClassifierBank columnar accessors (uniform_fs_hz)."""

from __future__ import annotations

import numpy as np
import pytest

from myocard_egm_data.banks import (
    ClassifierBank,
    ClassifierBankMetaData,
    ClassifierTrace,
)

_BANK_ID = "tbank_accessor_fixture_2026-06-28"


def _bank(freqs: list[float]) -> ClassifierBank:
    """Minimal ClassifierBank with one trace per entry in ``freqs``."""
    traces = [
        ClassifierTrace(
            bank_id=_BANK_ID,
            signal=np.zeros(8, dtype=np.float32),
            freq_hz=f,
            amp_type="mv",
            split=None,
            label_truth=0,
            prediction=None,
            trace_metadata={},
        )
        for f in freqs
    ]
    meta = ClassifierBankMetaData(
        bank_id=_BANK_ID, bank_type="synthetic", bank_path="<in-memory>", bank_metadata={}
    )
    return ClassifierBank(banks=[meta], traces=traces, labels={0: "healthy"})


def test_uniform_fs_hz_returns_shared_rate() -> None:
    assert _bank([1000.0, 1000.0, 1000.0]).uniform_fs_hz() == 1000.0


def test_uniform_fs_hz_raises_on_mixed_rates() -> None:
    with pytest.raises(ValueError, match="single shared sample rate"):
        _bank([1000.0, 500.0]).uniform_fs_hz()


def test_uniform_fs_hz_raises_on_empty_bank() -> None:
    meta = ClassifierBankMetaData(
        bank_id=_BANK_ID, bank_type="synthetic", bank_path="<in-memory>", bank_metadata={}
    )
    empty = ClassifierBank(banks=[meta], traces=[], labels={0: "healthy"})
    with pytest.raises(ValueError, match="no traces"):
        empty.uniform_fs_hz()
