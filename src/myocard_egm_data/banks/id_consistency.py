"""Checks that a stable artifact id tells the truth about its artifact.

egm-contracts owns the *shape* half of this: ``common.ArtifactId``'s
pattern accepts only a known role prefix, single-sourced from
``codegen/roles.json``. What JSON Schema structurally cannot check is
whether the artifact actually *is* what its prefix claims — a
``tbank_`` id on a bank with no labels validates perfectly and is
wrong. Nor can it compare two files, which is what the ``noise_bank`` ↔
``noise_bank_run_record`` id agreement needs. Both gaps land here,
because egm-data is the layer that has the artifact in hand at the
moment it is written.

Why bother, when the ids are producer-chosen strings: they are the
project's cross-artifact currency. A phase manifest, a run record and a
figure spec all point at artifacts by id, and a consumer that sees
``upred_...`` is entitled to assume there are no ground-truth labels to
compare against. A mislabelled id is not a cosmetic problem — it makes
every downstream reference quietly mean something else, and it is
invisible until someone reads a metric that cannot exist.

The checks are wired into the writers, so a mislabelled bank cannot be
written in the first place. Catching it on write is worth more than
catching it on read: the producer is still in a position to fix it, and
the bad file never reaches a manifest.
"""

from __future__ import annotations

from pathlib import Path

from myocard_egm_contracts.roles import Role, role_of

from .classifier_bank import ClassifierBank

__all__ = [
    "check_classifier_bank_id_matches_content",
    "check_noise_bank_id_agreement",
    "check_noise_bank_pair",
]

#: Bank roles this module knows how to check the *content* of. The other
#: roles in the vocabulary (``run_``, ``model_``, ``obs_``) name
#: artifacts that are not banks, so a ClassifierBank should never carry
#: one; ``nbank_`` is a bank but not a ClassifierBank.
_CLASSIFIER_BANK_ROLES = frozenset(
    {
        Role.training_bank,
        Role.pretraining_bank,
        Role.labeled_prediction_bank,
        Role.unlabeled_prediction_bank,
    }
)


def check_classifier_bank_id_matches_content(bank: ClassifierBank) -> None:
    """Assert a ClassifierBank's stable id matches what it actually holds.

    The four bank roles are the product of two independent, **exact**
    claims — which is what makes the check meaningful rather than
    tautological:

    ======================  ==========  ==============
    role                    labels      predictions
    ======================  ==========  ==============
    ``tbank_``  training    present     absent
    ``lpred_``  labeled     present     present
    ``ptbank_`` pretraining absent      absent
    ``upred_``  unlabeled   absent      present
    ======================  ==========  ==============

    Both axes are checked in both directions. In particular a
    ``tbank_`` carrying predictions is **wrong**, not merely
    unremarkable: a labeled bank that has been evaluated *is* a labeled
    prediction bank and belongs under ``lpred_``, which is what the
    linkage doc's role table says ("Predictions + truth labels — full
    metric suite available"). The same applies to ``ptbank_`` →
    ``upred_``.

    In practice today the situation barely arises, because eval writes a
    **new** bank rather than mutating the source — CLF4b settled that
    explicitly ("the predictions artifact gains a per-trace split +
    prediction … *not* a mutation of the source bank"). The check is
    written for the exact rule anyway, so that if predictions are ever
    attached in place the id is forced to keep up.

    A bank with no ``id`` is skipped: the id is optional precisely
    because intermediate banks are not tracked artifacts, and there is
    no claim to check.

    Raises
    ------
    ValueError
        If the id's role contradicts the content, naming both the id and
        what was actually found.
    """
    if bank.id is None:
        return

    role = role_of(bank.id)
    if role not in _CLASSIFIER_BANK_ROLES:
        raise ValueError(
            f"ClassifierBank id {bank.id!r} has role {role.name!r}, which is "
            "not a ClassifierBank role. Expected one of: "
            f"{sorted(r.name for r in _CLASSIFIER_BANK_ROLES)}."
        )

    n_labeled = sum(1 for t in bank.traces if t.label_truth is not None)
    n_predicted = sum(1 for t in bank.traces if t.prediction is not None)
    total = len(bank.traces)

    claims_labels = role in (Role.training_bank, Role.labeled_prediction_bank)
    claims_predictions = role in (
        Role.labeled_prediction_bank,
        Role.unlabeled_prediction_bank,
    )

    if claims_labels and n_labeled == 0 and total > 0:
        raise ValueError(
            f"ClassifierBank id {bank.id!r} claims role {role.name!r}, which "
            f"carries ground-truth labels, but none of its {total} traces has "
            "a label_truth. Either label the bank or give it a "
            "'ptbank_' / 'upred_' id."
        )
    if not claims_labels and n_labeled > 0:
        raise ValueError(
            f"ClassifierBank id {bank.id!r} claims role {role.name!r}, which "
            f"carries no ground truth, but {n_labeled} of {total} traces have "
            "a label_truth. A consumer would skip comparisons this bank can "
            "actually support — use a 'tbank_' / 'lpred_' id."
        )
    if claims_predictions and n_predicted == 0 and total > 0:
        raise ValueError(
            f"ClassifierBank id {bank.id!r} claims role {role.name!r}, which "
            f"carries model predictions, but none of its {total} traces has "
            "one. A predictions bank without predictions is an empty result "
            "set that reads as a real one."
        )
    if not claims_predictions and n_predicted > 0:
        evaluated_role = (
            Role.labeled_prediction_bank if claims_labels else Role.unlabeled_prediction_bank
        )
        raise ValueError(
            f"ClassifierBank id {bank.id!r} claims role {role.name!r}, which "
            f"carries no model predictions, but {n_predicted} of {total} "
            "traces have one. An evaluated bank is a prediction bank — use a "
            f"{evaluated_role.name!r} id (prefix "
            f"{'lpred_' if claims_labels else 'upred_'})."
        )


def check_noise_bank_id_agreement(bank_id: str | None, record_bank_id: str | None) -> None:
    """Assert a noise bank and its sibling run record name the same bank.

    ``noise_bank`` 1.1 and ``noise_bank_run_record`` 1.1 both carry
    ``bank_id``, and the schema requires them to agree when both are
    present — but JSON Schema validates one document at a time, so it
    explicitly delegates the comparison to egm-data ("egm-data checks
    that, since JSON Schema cannot compare across two files").

    Disagreement means the pair has been mis-assembled: a bank beside
    someone else's run record. That is worse than a missing record,
    because the provenance reads as present and is describing different
    data — a reader would attribute one extraction's calibration and
    windowing to another's traces.

    Either side being ``None`` is fine (both are optional for legacy
    artifacts); only a genuine mismatch raises.
    """
    if bank_id is None or record_bank_id is None:
        return
    if bank_id != record_bank_id:
        raise ValueError(
            f"noise bank id {bank_id!r} does not match its run record's "
            f"bank_id {record_bank_id!r}. The two files describe different "
            "banks; check that the sidecar beside this bank is the one that "
            "was written with it."
        )


def check_noise_bank_pair(bank_path: Path | str, record_path: Path | str) -> None:
    """Convenience wrapper: read both files and check their ids agree.

    Kept separate from the id-only check so callers that already hold
    the models (the producer, mid-write) don't pay for a re-read.
    """
    from ..records.noise_bank_run_record import load_noise_bank_run_record
    from .noise_bank import read_noise_bank_hdf5

    bank = read_noise_bank_hdf5(bank_path)
    record = load_noise_bank_run_record(record_path)
    check_noise_bank_id_agreement(bank.bank_id, record.bank_id)
