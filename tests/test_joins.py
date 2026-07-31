"""ClassifierBank ⋈ synthetic_bank — the T4 join (DAT1, S8).

The two banks are parallel artifacts sharing ``simulation_id``: features
come from the ClassifierBank, θ and the per-simulation config from the
synthetic bank. These tests cover the correspondence and, more
importantly, the three ways it can silently produce the *wrong* answer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import SYNTHETIC_2_0_BANK_ID
from myocard_egm_data.banks import (
    join_traces_with_simulations,
    load_iafdb_bank_as_classifier,
    load_synthetic_bank_as_classifier,
    read_synthetic_bank_hdf5,
    simulation_configs,
)


def _unwrap(value: object) -> object:
    return getattr(value, "root", value)


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_simulation_configs_transposes_the_columns(synthetic_bank_path: Path) -> None:
    """The column-oriented Simulations model becomes one row per simulation."""
    bank = read_synthetic_bank_hdf5(synthetic_bank_path)
    configs = simulation_configs(bank)

    assert sorted(configs) == [0, 1]
    assert configs[0].seed == 100
    assert configs[1].seed == 200
    # Objects are passed through as the contracts models they already
    # are — this is a reshape, not a reinterpretation.
    assert _unwrap(configs[1].substrate).density == 0.3
    assert configs[0].label_names["1"] == "fibrotic"


def test_every_trace_resolves_to_its_own_simulation(synthetic_bank_path: Path) -> None:
    """Each trace is paired with the config that actually generated it.

    The fixture's two simulations differ in substrate density (0.0 vs
    0.3) and in label, so a join that is off by one simulation — or that
    attaches the same config to everything — fails here rather than
    producing plausible-looking pairs.
    """
    cb = load_synthetic_bank_as_classifier(synthetic_bank_path)
    bank = read_synthetic_bank_hdf5(synthetic_bank_path)

    joined = join_traces_with_simulations(cb, bank)

    assert len(joined) == cb.n_traces
    for pair in joined:
        expected_id = pair.trace.trace_metadata["simulation_id"]
        assert pair.simulation.simulation_id == expected_id
        # Substrate density is what T4 varies, so check the join lands on
        # the right value rather than merely on *a* config.
        density = _unwrap(pair.simulation.substrate).density
        assert density == (0.0 if expected_id == 0 else 0.3)
        # And the label the classifier sees agrees with that substrate.
        assert pair.trace.label_truth == (0 if expected_id == 0 else 1)


def test_join_preserves_bank_order(synthetic_bank_path: Path) -> None:
    """Joined rows come back in trace order.

    STU1's feature-vs-θ scatter pairs these against a feature matrix
    computed from the same bank; a reordering here would silently
    misalign the two axes.
    """
    cb = load_synthetic_bank_as_classifier(synthetic_bank_path)
    bank = read_synthetic_bank_hdf5(synthetic_bank_path)

    joined = join_traces_with_simulations(cb, bank)
    assert [p.trace.trace_metadata["pair_index"] for p in joined] == [
        t.trace_metadata["pair_index"] for t in cb.traces
    ]


# ---------------------------------------------------------------------------
# The three silent-wrong-answer failures
# ---------------------------------------------------------------------------


def test_joining_against_a_different_bank_is_refused(
    synthetic_bank_path: Path, synthetic_bank_2_0_raw_path: Path
) -> None:
    """A ClassifierBank will not join against a *different* synthetic bank.

    This is the worst failure available here: simulation ids restart at
    0 in every bank, so a mismatched join produces a complete set of
    confident, wrong pairings and no error at all. Stable bank ids exist
    precisely so it is checkable, so it is a hard stop.
    """
    cb = load_synthetic_bank_as_classifier(synthetic_bank_path)
    other = read_synthetic_bank_hdf5(synthetic_bank_2_0_raw_path)
    assert other.bank_id == SYNTHETIC_2_0_BANK_ID
    assert other.bank_id != cb.banks[0].bank_id

    with pytest.raises(ValueError, match="not among the ClassifierBank's source banks"):
        join_traces_with_simulations(cb, other)


def test_iafdb_sourced_bank_is_refused(iafdb_bank_path: Path, synthetic_bank_path: Path) -> None:
    """An IAFDB-sourced ClassifierBank cannot be joined at all.

    Real recordings have no generation config. Returning an empty join
    would read as "no correspondence found" when the truth is "this
    question does not apply to this data".

    In practice the bank-correspondence guard catches this first — an
    IAFDB ClassifierBank does not list the synthetic bank as a source —
    and that message is the more useful one, so the test asserts the
    refusal rather than forcing execution down to the missing-key check.
    The missing-key path has its own test below.
    """
    cb = load_iafdb_bank_as_classifier(iafdb_bank_path)
    bank = read_synthetic_bank_hdf5(synthetic_bank_path)

    with pytest.raises(ValueError):
        join_traces_with_simulations(cb, bank)


def test_trace_without_the_join_key_is_refused(synthetic_bank_path: Path) -> None:
    """A trace missing simulation_id raises, naming the join as synthetic-only.

    Reachable in practice if a consumer builds a ClassifierBank by hand,
    or strips trace_metadata down before saving.
    """
    cb = load_synthetic_bank_as_classifier(synthetic_bank_path)
    bank = read_synthetic_bank_hdf5(synthetic_bank_path)
    del cb.traces[2].trace_metadata["simulation_id"]

    with pytest.raises(ValueError, match="synthetic-only"):
        join_traces_with_simulations(cb, bank)


def test_unknown_simulation_id_is_refused(synthetic_bank_path: Path) -> None:
    """A trace pointing at a simulation the bank lacks raises.

    Returning None for that row would push the decision onto every
    consumer, and the plausible consumer response — skip it — quietly
    drops traces from a comparison.
    """
    cb = load_synthetic_bank_as_classifier(synthetic_bank_path)
    bank = read_synthetic_bank_hdf5(synthetic_bank_path)
    cb.traces[-1].trace_metadata["simulation_id"] = 99

    with pytest.raises(ValueError, match="99"):
        join_traces_with_simulations(cb, bank)


def test_duplicate_simulation_ids_are_refused(synthetic_bank_path: Path) -> None:
    """Two simulations sharing an id make the join ambiguous."""
    bank = read_synthetic_bank_hdf5(synthetic_bank_path)
    bank.simulations.simulation_id[1] = bank.simulations.simulation_id[0]

    with pytest.raises(ValueError, match="two simulations"):
        simulation_configs(bank)
