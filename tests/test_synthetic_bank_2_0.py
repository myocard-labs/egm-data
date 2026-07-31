"""synthetic_bank 2.0 I/O — the breaking restructure (DAT1, S5-S8).

These tests read a bank written **by hand** from the schema's
``x-hdf5-mapping`` (see ``conftest.write_synthetic_bank_2_0_by_hand``)
rather than one produced by our own writer. Reader and writer are the
two halves of the same restructure, so a shared misreading of the schema
would round-trip perfectly and still be wrong; the hand-built file is
the independent reference both must agree with.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest
from myocard_egm_contracts.validators import validate_synthetic_bank

from conftest import (
    GENERATION_PARAMS_2_0,
    SIM_2_0,
    SYNTHETIC_2_0_BANK_ID,
    build_synthetic_bank_2_0_model,
    write_synthetic_bank_2_0_by_hand,
)
from myocard_egm_data.banks import read_synthetic_bank_hdf5, write_synthetic_bank


def _unwrap(value: object) -> object:
    """Unwrap a codegen ``RootModel`` container, if this value is one.

    The generated models are **inconsistent** about this, and the
    inconsistency is load-bearing enough to be worth stating: a union
    with a *single* variant today (Geometry, Substrate, Electrodes,
    Backend) codegens as a ``RootModel`` needing ``.root``, while a
    union with several (CellModel, Activation, LabelPolicy) codegens as
    a direct discriminated union with no wrapper. Constrained scalars
    (``PairIndexItem``, ``LabelItem``) are wrapped too.

    The trap is that the shape *flips when a variant is added* — the day
    Phase 7 adds a second Geometry, ``geometry.root`` stops existing.
    Unwrapping defensively means these tests keep passing across that
    change instead of encoding today's variant count.
    """
    return getattr(value, "root", value)


def _type_name(value: object) -> str:
    """Return a config object's discriminator as a plain string.

    Second asymmetry in the same family as ``_unwrap``: a variant that
    is currently the *only* member of its union gets a ``const``
    discriminator, which codegens to a plain ``str``, while a variant in
    a multi-member union gets an ``Enum`` needing ``.value``. So
    ``geometry.type`` is a ``str`` but ``cell_model.type`` is an
    ``Enum`` — and, again, that flips when a variant is added.
    """
    raw = getattr(_unwrap(value), "type", None)
    return str(getattr(raw, "value", raw))


# ---------------------------------------------------------------------------
# Reader (S5)
# ---------------------------------------------------------------------------


def test_hand_built_2_0_bank_is_schema_valid(synthetic_bank_2_0_raw_path: Path) -> None:
    """The fixture itself must satisfy the contracts validator.

    Guards the reference: if the hand-built layout drifts from the
    schema, every other test in this module is asserting against a
    fiction rather than against the contract.
    """
    result = validate_synthetic_bank(synthetic_bank_2_0_raw_path)
    assert result.ok, result.issues


def test_reads_root_attrs_and_theta_spec(synthetic_bank_2_0_raw_path: Path) -> None:
    """Root attrs decode, including the JSON-encoded theta-spec."""
    bank = read_synthetic_bank_hdf5(synthetic_bank_2_0_raw_path)

    assert bank.bank_id == SYNTHETIC_2_0_BANK_ID
    assert bank.fs_hz == 1000.0
    assert bank.trace_duration_ms == 512.0
    assert bank.noise_bank_source == "iafdb_noise_v1.h5"

    # generation_params is bank-scoped, not per-simulation: one theta-spec
    # describes what the whole sweep varied.
    knobs = bank.generation_params.knobs
    assert len(knobs) == 1
    assert knobs[0].path == "substrate.density"
    assert [float(b) for b in knobs[0].bounds] == [0.0, 0.5]


def test_simulations_group_decodes_to_typed_variants(synthetic_bank_2_0_raw_path: Path) -> None:
    """Each ``<name>_json`` column decodes to the right discriminated variant.

    The two simulations deliberately use *different* variants on the
    cell_model and activation unions, so this fails if the reader
    resolves a union once per column instead of per row.
    """
    bank = read_synthetic_bank_hdf5(synthetic_bank_2_0_raw_path)
    sims = bank.simulations

    assert list(sims.simulation_id) == [0, 1]
    assert list(sims.seed) == [100, 200]

    assert _type_name(sims.geometry[0]) == "patch_2d"
    assert _unwrap(sims.geometry[0]).size_mm == 40.0

    assert _type_name(sims.cell_model[0]) == "courtemanche"
    assert _type_name(sims.cell_model[1]) == "aliev_panfilov"
    assert _unwrap(sims.cell_model[1]).ap_time_unit_ms == 12.9

    activation_0 = _unwrap(sims.activation[0])
    assert _type_name(sims.activation[0]) == "planar_edge"
    assert [e.value for e in activation_0.edges] == ["top"]
    assert _type_name(sims.activation[1]) == "point"

    assert _unwrap(sims.substrate[0]).density == 0.0
    assert _unwrap(sims.substrate[1]).density == 0.3
    assert _unwrap(sims.substrate_summary[1]).n_fibrotic_nodes == 4768

    # Label policy is per-simulation and threshold-list shaped, so
    # multiclass severity needs no new variant (CL-088).
    policy = _unwrap(sims.label_policy[0])
    assert _type_name(sims.label_policy[0]) == "global_density"
    assert [float(_unwrap(t)) for t in policy.thresholds] == [0.1]
    assert sims.label_names[0]["1"] == "fibrotic"

    # The realized per-pair detail traces index by pair_index.
    electrodes = _unwrap(sims.electrodes[0])
    assert len(electrodes.pairs) == 3
    assert electrodes.pairs[2].height_mm == 1.0


def test_traces_group_is_collapsed(synthetic_bank_2_0_raw_path: Path) -> None:
    """traces/ carries signal + two FKs + int label + noise provenance only.

    The generation columns 1.1 kept per-trace (fibrosis_density,
    electrode_row, stim_edge, ...) moved to simulations/; each of them
    encoded a Phase-1 assumption and was a per-trace copy of a
    per-simulation fact.
    """
    bank = read_synthetic_bank_hdf5(synthetic_bank_2_0_raw_path)
    t = bank.traces

    assert len(t.signal) == 6
    assert len(t.signal[0]) == 512
    assert list(t.simulation_id) == [0, 0, 0, 1, 1, 1]
    assert [int(_unwrap(p)) for p in t.pair_index] == [0, 1, 2, 0, 1, 2]
    # Plain int label; the name map lives per simulation.
    assert [int(_unwrap(x)) for x in t.label] == [0, 0, 0, 1, 1, 1]
    assert list(t.noise_record) == ["iaf1_afw"] * 6

    for gone in (
        "fibrosis_density",
        "fibrosis_density_realized",
        "electrode_row",
        "electrode_height_mm",
        "stim_edge",
        "seed",
    ):
        assert not hasattr(t, gone), f"{gone} should have moved out of traces/"


def test_activation_position_absent_by_default(synthetic_bank_2_0_raw_path: Path) -> None:
    """Wave-1 banks carry no activation_position; absence reads as None.

    SEP2 populates it in Wave 2. Absence means "unknown position" — a
    zero-filled column would read as "every activation at the very start
    of its trace", which is a meaningful and wrong claim.
    """
    bank = read_synthetic_bank_hdf5(synthetic_bank_2_0_raw_path)
    assert bank.traces.activation_position is None


def test_activation_position_round_trips_when_present(
    tmp_path: Path, fs_hz: float, trace_duration_ms: float, n_samples: int
) -> None:
    """When the splitter has run, the column reads back at full precision."""
    path = write_synthetic_bank_2_0_by_hand(
        tmp_path / "with_positions.h5",
        fs_hz=fs_hz,
        trace_duration_ms=trace_duration_ms,
        n_samples=n_samples,
        with_activation_position=True,
    )
    bank = read_synthetic_bank_hdf5(path)
    assert bank.traces.activation_position is not None
    positions = [float(_unwrap(x)) for x in bank.traces.activation_position]
    assert positions == pytest.approx([0.0, 0.25, 0.5, 0.5, 0.75, 1.0])


# ---------------------------------------------------------------------------
# Refusals — the reader must fail loudly, not partially read
# ---------------------------------------------------------------------------


def test_pre_2_0_bank_is_refused(tmp_path: Path, synthetic_bank_2_0_raw_path: Path) -> None:
    """A 1.1 bank raises rather than being read on a best-effort basis.

    There is no migration path by decision (egm-contracts v0.6.0): the
    generation parameters a 1.1 bank carries live in columns 2.0 does
    not have, so a partial read would silently discard the provenance
    the bank exists to record. Regenerating is the supported path.
    """
    legacy = tmp_path / "legacy_1_1.h5"
    legacy.write_bytes(synthetic_bank_2_0_raw_path.read_bytes())
    with h5py.File(legacy, "a") as f:
        f.attrs["schema_version"] = "1.1"

    with pytest.raises(ValueError, match="schema_version"):
        read_synthetic_bank_hdf5(legacy)


def test_missing_simulations_group_raises(
    tmp_path: Path, synthetic_bank_2_0_raw_path: Path
) -> None:
    """A bank with traces but no simulations/ is malformed, not readable.

    Every trace's config is reached through the simulation FK, so
    without the group there is nothing to resolve against.
    """
    broken = tmp_path / "no_sims.h5"
    broken.write_bytes(synthetic_bank_2_0_raw_path.read_bytes())
    with h5py.File(broken, "a") as f:
        del f["simulations"]

    with pytest.raises(ValueError, match="simulations"):
        read_synthetic_bank_hdf5(broken)


def test_malformed_config_json_names_the_column_and_row(
    tmp_path: Path, synthetic_bank_2_0_raw_path: Path
) -> None:
    """A row that isn't valid JSON fails with its column and row index.

    The contracts validator deliberately leaves an unparseable row as a
    raw string so it can report a schema error; this reader cannot --
    its caller gets a model or an exception -- so it must say precisely
    which row of which column is at fault rather than surfacing a bare
    JSONDecodeError from somewhere inside the decode loop.
    """
    broken = tmp_path / "bad_json.h5"
    broken.write_bytes(synthetic_bank_2_0_raw_path.read_bytes())
    with h5py.File(broken, "a") as f:
        del f["simulations/substrate_json"]
        f.create_dataset(
            "simulations/substrate_json",
            data=[json.dumps(SIM_2_0[0]["substrate"]), "{not json"],
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    with pytest.raises(ValueError, match=r"substrate_json.*row 1"):
        read_synthetic_bank_hdf5(broken)


# ---------------------------------------------------------------------------
# Writer (S6)
# ---------------------------------------------------------------------------


def test_writer_output_validates(synthetic_bank_path: Path) -> None:
    """A bank written by our writer passes the contracts validator."""
    result = validate_synthetic_bank(synthetic_bank_path)
    assert result.ok, result.issues


def test_writer_matches_the_hand_built_layout(
    synthetic_bank_path: Path, synthetic_bank_2_0_raw_path: Path
) -> None:
    """Our writer produces the same HDF5 *structure* as the hand-built file.

    This is the point of keeping an independent reference: a reader and
    writer that share a misreading of the schema round-trip perfectly.
    Comparing group and dataset names against a file laid out from
    ``x-hdf5-mapping`` catches a column written under the wrong name, a
    missing ``_json`` suffix, or a stray 1.1 leftover — none of which a
    round-trip alone would notice.
    """
    import h5py as _h5py

    def _layout(path: Path) -> dict[str, list[str]]:
        with _h5py.File(path, "r") as f:
            return {
                "root_attrs": sorted(f.attrs),
                "simulations": sorted(f["simulations"]),
                "traces": sorted(f["traces"]),
            }

    assert _layout(synthetic_bank_path) == _layout(synthetic_bank_2_0_raw_path)


def test_round_trip_preserves_config_and_traces(
    synthetic_bank_path: Path, fs_hz: float, trace_duration_ms: float, n_samples: int
) -> None:
    """model -> write -> read returns an equivalent bank."""
    original = build_synthetic_bank_2_0_model(
        fs_hz=fs_hz, trace_duration_ms=trace_duration_ms, n_samples=n_samples
    )
    reloaded = read_synthetic_bank_hdf5(synthetic_bank_path)

    assert reloaded.bank_id == original.bank_id
    assert reloaded.simulations.model_dump(mode="json") == original.simulations.model_dump(
        mode="json"
    )
    assert reloaded.generation_params.model_dump(mode="json") == (
        original.generation_params.model_dump(mode="json")
    )
    assert [int(_unwrap(x)) for x in reloaded.traces.label] == [0, 0, 0, 1, 1, 1]
    assert list(reloaded.traces.simulation_id) == list(original.traces.simulation_id)


def test_writer_round_trips_activation_position(
    tmp_path: Path, fs_hz: float, trace_duration_ms: float, n_samples: int
) -> None:
    """The optional column survives our own write path too (CL-062)."""
    bank = build_synthetic_bank_2_0_model(
        fs_hz=fs_hz,
        trace_duration_ms=trace_duration_ms,
        n_samples=n_samples,
        with_activation_position=True,
    )
    path = write_synthetic_bank(bank, tmp_path / "written_positions.h5")
    reloaded = read_synthetic_bank_hdf5(path)

    assert reloaded.traces.activation_position is not None
    positions = [float(_unwrap(x)) for x in reloaded.traces.activation_position]
    assert positions == pytest.approx([0.0, 0.25, 0.5, 0.5, 0.75, 1.0])


def test_writer_refuses_an_orphan_trace(
    tmp_path: Path, fs_hz: float, trace_duration_ms: float, n_samples: int
) -> None:
    """A trace whose simulation_id has no simulation is refused on write.

    The cross-group FK is checked by the contracts validator, but JSON
    Schema cannot express a cross-group reference, so the schema hands
    this to egm-data. Catching it at the write call surfaces a producer
    bug where it happened, rather than as a validation failure against a
    file already on disk (CL-089).
    """
    bank = build_synthetic_bank_2_0_model(
        fs_hz=fs_hz, trace_duration_ms=trace_duration_ms, n_samples=n_samples
    )
    # Point one trace at a simulation that was never recorded.
    bank.traces.simulation_id[-1] = 99

    with pytest.raises(ValueError, match=r"99.*simulations/"):
        write_synthetic_bank(bank, tmp_path / "orphan.h5")


def test_theta_spec_survives_byte_for_byte(synthetic_bank_2_0_raw_path: Path) -> None:
    """The theta-spec decodes to exactly what was written.

    STU1 and STU4 walk this structurally for bounds / transform / role,
    so a lossy decode would quietly change what the estimator searches
    over.
    """
    bank = read_synthetic_bank_hdf5(synthetic_bank_2_0_raw_path)
    dumped = bank.generation_params.model_dump(mode="json", exclude_none=True)
    assert dumped["regime"] == GENERATION_PARAMS_2_0["regime"]
    assert dumped["knobs"][0]["path"] == "substrate.density"
