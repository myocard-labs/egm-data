"""Stable ids must tell the truth about their artifacts (B16, S11).

egm-contracts checks the *shape* of an id — the prefix is a known role.
These checks cover what JSON Schema structurally cannot: whether the
artifact actually is what its prefix claims, and whether two files that
both name a bank agree about which bank.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from myocard_egm_data.banks import (
    ClassifierBank,
    ClassifierBankMetaData,
    ClassifierPrediction,
    ClassifierTrace,
    check_classifier_bank_id_matches_content,
    check_noise_bank_id_agreement,
    write_classifier_bank,
)

_SOURCE_ID = "tbank_synthetic_test_2026-06-27"


def _trace(*, labeled: bool, predicted: bool) -> ClassifierTrace:
    return ClassifierTrace(
        bank_id=_SOURCE_ID,
        signal=np.zeros(8, dtype=np.float32),
        freq_hz=1000.0,
        amp_type="mv",
        split=None,
        label_truth=1 if labeled else None,
        prediction=(
            ClassifierPrediction(label_pred=1, label_prob=0.9, pred_logits={0: -1.0, 1: 1.0})
            if predicted
            else None
        ),
        trace_metadata={"patient_id": "0"},
    )


def _bank(bank_id: str | None, *, labeled: bool, predicted: bool) -> ClassifierBank:
    return ClassifierBank(
        id=bank_id,
        banks=[
            ClassifierBankMetaData(
                bank_id=_SOURCE_ID, bank_type="synthetic", bank_path="x.h5", bank_metadata={}
            )
        ],
        traces=[_trace(labeled=labeled, predicted=predicted) for _ in range(3)],
        labels={0: "healthy", 1: "fibrotic"},
    )


# ---------------------------------------------------------------------------
# Ids that tell the truth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bank_id", "labeled", "predicted"),
    [
        # The four roles are the product of two exact claims:
        #                        labels   predictions
        ("tbank_synthetic_v1_5_2026-07-31", True, False),
        ("lpred_run42_2026-07-31", True, True),
        ("ptbank_iafdb_v1_5_2026-07-31", False, False),
        ("upred_run42_2026-07-31", False, True),
    ],
)
def test_consistent_ids_pass(bank_id: str, labeled: bool, predicted: bool) -> None:
    check_classifier_bank_id_matches_content(_bank(bank_id, labeled=labeled, predicted=predicted))


def test_bank_without_an_id_is_skipped() -> None:
    """An untracked intermediate bank makes no claim, so there is nothing
    to check. The id is optional precisely for this case."""
    check_classifier_bank_id_matches_content(_bank(None, labeled=False, predicted=False))


# ---------------------------------------------------------------------------
# Ids that lie
# ---------------------------------------------------------------------------


def test_training_bank_without_labels_is_refused() -> None:
    """`tbank_` claims ground truth; a bank with none cannot train anything."""
    with pytest.raises(ValueError, match="label_truth"):
        check_classifier_bank_id_matches_content(
            _bank("tbank_empty_2026-07-31", labeled=False, predicted=False)
        )


def test_unlabeled_prediction_bank_with_labels_is_refused() -> None:
    """`upred_` claims no ground truth, so labels here mislead in the
    expensive direction.

    A consumer seeing `upred_` skips every comparison against ground
    truth — so a mislabelled bank does not produce a wrong number, it
    produces a silently *missing* evaluation the data could have
    supported.
    """
    with pytest.raises(ValueError, match="no ground truth"):
        check_classifier_bank_id_matches_content(
            _bank("upred_run42_2026-07-31", labeled=True, predicted=True)
        )


def test_prediction_bank_without_predictions_is_refused() -> None:
    """A predictions bank with no predictions is an empty result set
    that reads as a real one."""
    with pytest.raises(ValueError, match="predictions"):
        check_classifier_bank_id_matches_content(
            _bank("lpred_run42_2026-07-31", labeled=True, predicted=False)
        )


def test_evaluated_training_bank_must_be_lpred() -> None:
    """A labeled bank carrying predictions is a labeled *prediction* bank.

    `tbank_` claims no predictions. Evaluating a training bank makes it
    an `lpred_` — "predictions + truth labels, full metric suite
    available" per the linkage doc's role table — so the id has to keep
    up with what the artifact became.

    Barely reachable today: eval writes a new bank rather than mutating
    the source (CLF4b settled that the predictions artifact is separate,
    "*not* a mutation of the source bank"). The rule is enforced anyway,
    so that attaching predictions in place later cannot silently leave a
    stale role on the id.
    """
    with pytest.raises(ValueError, match="labeled_prediction_bank"):
        check_classifier_bank_id_matches_content(
            _bank("tbank_synthetic_v1_5_2026-07-31", labeled=True, predicted=True)
        )


def test_evaluated_pretraining_bank_must_be_upred() -> None:
    """The unlabeled half of the same rule: `ptbank_` + predictions -> `upred_`."""
    with pytest.raises(ValueError, match="unlabeled_prediction_bank"):
        check_classifier_bank_id_matches_content(
            _bank("ptbank_iafdb_v1_5_2026-07-31", labeled=False, predicted=True)
        )


def test_non_bank_role_is_refused() -> None:
    """`run_` / `model_` name artifacts that are not ClassifierBanks."""
    with pytest.raises(ValueError, match="not a ClassifierBank role"):
        check_classifier_bank_id_matches_content(
            _bank("model_mobilevit_2026-07-31", labeled=True, predicted=False)
        )


def test_writer_enforces_the_check(tmp_path: Path) -> None:
    """The check is wired into the writer, not left to callers.

    Catching this on write beats catching it on read: the producer is
    still in a position to fix it, and the mislabelled file never
    reaches a phase manifest where other artifacts start pointing at it.
    """
    bad = _bank("upred_run42_2026-07-31", labeled=True, predicted=True)
    with pytest.raises(ValueError, match="no ground truth"):
        write_classifier_bank(bad, tmp_path / "bad.classifier.h5")
    assert not (tmp_path / "bad.classifier.h5").exists()


# ---------------------------------------------------------------------------
# noise_bank <-> noise_bank_run_record agreement
# ---------------------------------------------------------------------------


def test_matching_noise_bank_ids_pass() -> None:
    check_noise_bank_id_agreement("nbank_iafdb_2026-07-31", "nbank_iafdb_2026-07-31")


def test_absent_noise_bank_id_is_not_a_mismatch() -> None:
    """Either side may be absent on a legacy artifact; only a genuine
    disagreement is an error."""
    check_noise_bank_id_agreement(None, "nbank_iafdb_2026-07-31")
    check_noise_bank_id_agreement("nbank_iafdb_2026-07-31", None)
    check_noise_bank_id_agreement(None, None)


def test_mismatched_noise_bank_ids_are_refused() -> None:
    """A bank sitting beside someone else's run record.

    Worse than a missing record: the provenance reads as present while
    describing different data, so a reader attributes one extraction's
    calibration and windowing to another's traces. JSON Schema cannot
    catch this — it validates one document at a time — which is why the
    noise_bank schema delegates the comparison to egm-data.
    """
    with pytest.raises(ValueError, match="does not match"):
        check_noise_bank_id_agreement("nbank_iafdb_2026-07-31", "nbank_other_2026-07-30")
