"""Shared fixtures — Pydantic bank builders that the writers serialize.

We build the source-bank artifacts (synthetic, IAFDB, noise) by
constructing the Pydantic models codegen'd from egm-contracts schemas
and handing them to ``write_synthetic_bank`` / ``write_iafdb_bank`` /
``write_noise_bank``. There is no parallel set of dataclasses in
egm-data anymore; the contracts package is the only place those types
live.

One deliberate exception: ``synthetic_bank_2_0_raw_path`` writes its
HDF5 **by hand with h5py**, not through our writer. The reader and the
writer are the two halves of the same restructure, so testing the
reader against writer output would let a shared misreading of the
schema pass unnoticed. The hand-built file is laid out from the
schema's ``x-hdf5-mapping`` directly, which is also what the producer
(SEP12) will target.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest
from myocard_egm_contracts import iafdb_bank as iafdb_bank_models
from myocard_egm_contracts import noise_bank as noise_bank_models
from myocard_egm_contracts import synthetic_bank as synthetic_bank_models
from myocard_egm_contracts.schema_info import current_version

from myocard_egm_data.banks import write_iafdb_bank, write_noise_bank, write_synthetic_bank

_STR_DTYPE = h5py.string_dtype(encoding="utf-8")


@pytest.fixture
def fs_hz() -> float:
    return 1000.0


@pytest.fixture
def trace_duration_ms() -> float:
    return 512.0


@pytest.fixture
def n_samples(fs_hz: float, trace_duration_ms: float) -> int:
    return round(trace_duration_ms * 1e-3 * fs_hz)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# synthetic_bank 2.0 — hand-built HDF5 (independent of our writer)
# ---------------------------------------------------------------------------

#: Per-simulation config for the two simulations in the 2.0 fixture.
#: Simulation 0 is healthy (density 0.0), simulation 1 fibrotic (0.3),
#: so the int labels differ and a converter that mixes up the join shows
#: up as a label mismatch rather than a silent metadata swap.
SIM_2_0 = [
    {
        "simulation_id": 0,
        "seed": 100,
        "geometry": {"type": "patch_2d", "size_mm": 40.0, "dr_mm": 0.25},
        "cell_model": {"type": "courtemanche", "params": {"g_CaL_scale": 1.0}},
        "substrate": {"type": "uniform_random_fibrosis", "density": 0.0},
        "substrate_summary": {"realized_density": 0.0, "n_fibrotic_nodes": 0},
        "activation": {"type": "planar_edge", "edges": ["top"]},
        "electrodes": {
            "type": "centered_grid_2d",
            "n_rows": 5,
            "n_cols": 5,
            "spacing_mm": 2.0,
            "height_mm": 0.5,
            "pairs": [
                {"pair_index": 0, "electrode_row": 0, "height_mm": 0.5},
                {"pair_index": 1, "electrode_row": 0, "height_mm": 0.5},
                {"pair_index": 2, "electrode_row": 1, "height_mm": 1.0},
            ],
        },
        "backend": {"type": "finitewave", "output_fs_hz": 1000.0},
        "label_policy": {"type": "global_density", "thresholds": [0.1]},
        "label_names": {"0": "healthy", "1": "fibrotic"},
    },
    {
        "simulation_id": 1,
        "seed": 200,
        "geometry": {"type": "patch_2d", "size_mm": 40.0, "dr_mm": 0.25},
        # A different variant on the polymorphic union, so the fixture
        # proves the discriminator is honoured per row rather than being
        # decided once for the whole column.
        "cell_model": {"type": "aliev_panfilov", "ap_time_unit_ms": 12.9},
        "substrate": {"type": "uniform_random_fibrosis", "density": 0.3},
        "substrate_summary": {"realized_density": 0.298, "n_fibrotic_nodes": 4768},
        "activation": {"type": "point", "position_mm": [20.0, 20.0]},
        "electrodes": {
            "type": "centered_grid_2d",
            "n_rows": 5,
            "n_cols": 5,
            "spacing_mm": 2.0,
            "height_mm": 0.5,
            "pairs": [
                {"pair_index": 0, "electrode_row": 0, "height_mm": 0.5},
                {"pair_index": 1, "electrode_row": 0, "height_mm": 0.5},
                {"pair_index": 2, "electrode_row": 1, "height_mm": 1.0},
            ],
        },
        "backend": {"type": "finitewave", "output_fs_hz": 1000.0},
        "label_policy": {"type": "global_density", "thresholds": [0.1]},
        "label_names": {"0": "healthy", "1": "fibrotic"},
    },
]

#: The bank-scoped theta-spec. Deliberately non-trivial (one swept knob)
#: even though Wave-1 producer output has an empty sweep, so the reader
#: is exercised against a populated spec rather than only the empty case.
GENERATION_PARAMS_2_0 = {
    "regime": {
        "geometry": "patch_2d",
        "cell_model": "courtemanche",
        "substrate": "uniform_random_fibrosis",
    },
    "knobs": [
        {
            "path": "substrate.density",
            "bounds": [0.0, 0.5],
            "transform": "identity",
            "role": "label_param",
        }
    ],
}

SYNTHETIC_2_0_BANK_ID = "tbank_synthetic_v2_test_2026-07-31"

_SIM_JSON_COLUMNS = (
    "geometry",
    "cell_model",
    "substrate",
    "substrate_summary",
    "activation",
    "electrodes",
    "backend",
    "label_policy",
    "label_names",
)


SYNTHETIC_BANK_ID = "tbank_synthetic_test_2026-06-27"


def build_synthetic_bank_2_0_model(
    *,
    fs_hz: float,
    trace_duration_ms: float,
    n_samples: int,
    with_activation_position: bool = False,
) -> synthetic_bank_models.SyntheticBank:
    """Build the Pydantic SyntheticBank the 2.0 writer serializes.

    Same two simulations and theta-spec as the hand-built HDF5 fixture,
    so writer output and hand-built reference describe the same bank.
    """
    rng = np.random.default_rng(0)
    n = 6
    doc: dict = {
        "schema_version": current_version("synthetic_bank"),
        "created_utc": _now_iso(),
        "bank_id": SYNTHETIC_BANK_ID,
        "description": "Synthetic bank test fixture",
        "fs_hz": fs_hz,
        "trace_duration_ms": trace_duration_ms,
        "noise_bank_source": "iafdb_noise_v1.h5",
        "generation_params": GENERATION_PARAMS_2_0,
        "simulations": {
            "simulation_id": [s["simulation_id"] for s in SIM_2_0],
            "seed": [s["seed"] for s in SIM_2_0],
            **{name: [s[name] for s in SIM_2_0] for name in _SIM_JSON_COLUMNS},
        },
        "traces": {
            "signal": [
                rng.standard_normal(n_samples).astype(np.float32).tolist() for _ in range(n)
            ],
            "simulation_id": [0, 0, 0, 1, 1, 1],
            "pair_index": [0, 1, 2, 0, 1, 2],
            "label": [0, 0, 0, 1, 1, 1],
            "snr_db": [15.0] * n,
            "noise_record": ["iaf1_afw"] * n,
            "noise_channel": ["CS12"] * n,
        },
    }
    if with_activation_position:
        doc["traces"]["activation_position"] = [0.0, 0.25, 0.5, 0.5, 0.75, 1.0]
    return synthetic_bank_models.SyntheticBank.model_validate(doc)


def write_synthetic_bank_2_0_by_hand(
    path: Path,
    *,
    fs_hz: float,
    trace_duration_ms: float,
    n_samples: int,
    with_activation_position: bool = False,
) -> Path:
    """Write a schema-2.0 synthetic bank directly with h5py.

    Laid out from the schema's ``x-hdf5-mapping``: root attrs (with the
    theta-spec JSON-encoded as ``generation_params_json``), a
    ``simulations/`` group whose polymorphic config objects are one JSON
    document per row under ``<name>_json``, and the collapsed
    ``traces/`` group.

    Six traces over two simulations, three bipolar pairs each. Labels
    follow the per-simulation substrate: sim 0 healthy -> 0, sim 1
    fibrotic -> 1.
    """
    rng = np.random.default_rng(0)
    n = 6
    signal = rng.standard_normal((n, n_samples)).astype(np.float32)
    sim_id = [0, 0, 0, 1, 1, 1]
    pair_index = [0, 1, 2, 0, 1, 2]
    label = [0, 0, 0, 1, 1, 1]

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = current_version("synthetic_bank")
        f.attrs["created_utc"] = _now_iso()
        f.attrs["bank_id"] = SYNTHETIC_2_0_BANK_ID
        f.attrs["description"] = "synthetic_bank 2.0 hand-built fixture"
        f.attrs["fs_hz"] = float(fs_hz)
        f.attrs["trace_duration_ms"] = float(trace_duration_ms)
        f.attrs["noise_bank_source"] = "iafdb_noise_v1.h5"
        f.attrs["generation_params_json"] = json.dumps(GENERATION_PARAMS_2_0, sort_keys=True)

        sims = f.create_group("simulations")
        sims.create_dataset(
            "simulation_id", data=np.asarray([s["simulation_id"] for s in SIM_2_0], dtype=np.int64)
        )
        sims.create_dataset("seed", data=np.asarray([s["seed"] for s in SIM_2_0], dtype=np.int64))
        for name in _SIM_JSON_COLUMNS:
            sims.create_dataset(
                f"{name}_json",
                data=np.asarray(
                    [json.dumps(s[name], sort_keys=True) for s in SIM_2_0], dtype=object
                ),
                dtype=_STR_DTYPE,
            )

        traces = f.create_group("traces")
        traces.create_dataset("signal", data=signal, dtype=np.float32)
        traces.create_dataset("simulation_id", data=np.asarray(sim_id, dtype=np.int64))
        traces.create_dataset("pair_index", data=np.asarray(pair_index, dtype=np.int64))
        traces.create_dataset("label", data=np.asarray(label, dtype=np.int64))
        traces.create_dataset("snr_db", data=np.asarray([15.0] * n, dtype=np.float64))
        traces.create_dataset(
            "noise_record", data=np.asarray(["iaf1_afw"] * n, dtype=object), dtype=_STR_DTYPE
        )
        traces.create_dataset(
            "noise_channel", data=np.asarray(["CS12"] * n, dtype=object), dtype=_STR_DTYPE
        )
        if with_activation_position:
            traces.create_dataset(
                "activation_position",
                data=np.asarray([0.0, 0.25, 0.5, 0.5, 0.75, 1.0], dtype=np.float32),
            )
    return path


@pytest.fixture
def synthetic_bank_2_0_raw_path(
    tmp_path: Path, fs_hz: float, trace_duration_ms: float, n_samples: int
) -> Path:
    """A schema-2.0 bank written by hand, without our writer (see module docstring)."""
    return write_synthetic_bank_2_0_by_hand(
        tmp_path / "synthetic_bank_2_0.h5",
        fs_hz=fs_hz,
        trace_duration_ms=trace_duration_ms,
        n_samples=n_samples,
    )


@pytest.fixture
def synthetic_bank_path(
    tmp_path: Path, fs_hz: float, trace_duration_ms: float, n_samples: int
) -> Path:
    """Build a tiny Pydantic SyntheticBank (schema 2.0) and write it to HDF5.

    Two simulations x three bipolar pairs. Simulation 0 healthy
    (substrate density 0.0, label 0), simulation 1 fibrotic (0.3, label
    1). Signals random — physiological realism isn't needed for the I/O
    tests.

    Shares its per-simulation config and theta-spec with the hand-built
    fixture (``SIM_2_0`` / ``GENERATION_PARAMS_2_0``), so the writer and
    the hand-built reference describe the same bank and any divergence
    between them shows up as a round-trip failure rather than as two
    tests quietly asserting different things.
    """
    pyd_bank = build_synthetic_bank_2_0_model(
        fs_hz=fs_hz, trace_duration_ms=trace_duration_ms, n_samples=n_samples
    )
    path = tmp_path / "synthetic_bank.h5"
    write_synthetic_bank(pyd_bank, path)
    return path


@pytest.fixture
def iafdb_bank_path(tmp_path: Path, fs_hz: float, trace_duration_ms: float, n_samples: int) -> Path:
    """Build a tiny Pydantic IafdbBank and write it to HDF5.

    Two patients x two segments. The bank schema (1.3) carries no label
    column — labeling happens at ClassifierBank conversion time.

    Deliberately omits both 1.3 optional fields (``run_record_path`` and
    per-trace ``activation_position``), so this fixture is the shape a
    Wave-1 sliding-window bank actually has and every test using it
    exercises the absent path. See ``iafdb_bank_with_optionals_path``
    for the populated counterpart.
    """
    rng = np.random.default_rng(1)
    n = 4
    signal = [rng.standard_normal(n_samples).astype(np.float32).tolist() for _ in range(n)]
    patient_id = ["iaf1", "iaf1", "iaf2", "iaf2"]
    source_channel = ["CS12", "CS34", "CS12", "CS34"]

    pyd_bank = iafdb_bank_models.IafdbBank.model_validate(
        {
            "schema_version": current_version("iafdb_bank"),
            "created_utc": _now_iso(),
            "bank_id": "tbank_iafdb_test_2026-06-27",
            "source": "iafdb v1.0.0",
            "fs_hz": fs_hz,
            "trace_duration_ms": trace_duration_ms,
            "calibration_method": "r_wave_anchoring",
            "calibration_target_qrs_pp_mv": 1.0,
            "threshold_mode": "absolute",
            "threshold_value": 0.5,
            "band_hz": [30.0, 300.0],
            "window_ms": trace_duration_ms,
            "window_samples": n_samples,
            "hop_ms": 256.0,
            "source_records": ["iaf1_afw", "iaf2_afw"],
            "traces": {
                "signal": signal,
                "patient_id": patient_id,
                "source_record": [f"{p}_afw" for p in patient_id],
                "source_channel": source_channel,
                "start_sample": [0, 256, 0, 256],
                "peak_to_peak_mv": [1.0] * n,
                "calibration_scalar": [0.95] * n,
            },
        }
    )
    path = tmp_path / "iafdb_bank.h5"
    write_iafdb_bank(pyd_bank, path)
    return path


@pytest.fixture
def iafdb_bank_with_optionals_path(
    tmp_path: Path, fs_hz: float, trace_duration_ms: float, n_samples: int
) -> Path:
    """An IafdbBank carrying both iafdb_bank 1.3 optional fields.

    The shape an activation-split bank has once IAF1 populates it in
    Wave 2: a ``run_record_path`` pointer to the sibling audit report
    (B11) and a per-trace ``activation_position`` fraction (CL-053).
    Positions are deliberately spread across [0, 1] — including the
    endpoints — so a reader that clamps, rounds, or drops the column
    fails loudly.
    """
    rng = np.random.default_rng(3)
    n = 4
    signal = [rng.standard_normal(n_samples).astype(np.float32).tolist() for _ in range(n)]
    patient_id = ["iaf1", "iaf1", "iaf2", "iaf2"]

    pyd_bank = iafdb_bank_models.IafdbBank.model_validate(
        {
            "schema_version": current_version("iafdb_bank"),
            "created_utc": _now_iso(),
            "bank_id": "tbank_iafdb_activation_test_2026-07-31",
            "run_record_path": "iafdb_activation_v1_run_record.json",
            "source": "iafdb v1.0.0",
            "fs_hz": fs_hz,
            "trace_duration_ms": trace_duration_ms,
            "calibration_method": "r_wave_anchoring",
            "calibration_target_qrs_pp_mv": 1.0,
            "threshold_mode": "absolute",
            "threshold_value": 0.5,
            "band_hz": [30.0, 300.0],
            "window_ms": trace_duration_ms,
            "window_samples": n_samples,
            "hop_ms": 256.0,
            "source_records": ["iaf1_afw", "iaf2_afw"],
            "traces": {
                "signal": signal,
                "patient_id": patient_id,
                "source_record": [f"{p}_afw" for p in patient_id],
                "source_channel": ["CS12", "CS34", "CS12", "CS34"],
                "start_sample": [0, 256, 0, 256],
                "peak_to_peak_mv": [1.0] * n,
                "calibration_scalar": [0.95] * n,
                "activation_position": [0.0, 0.25, 0.5, 1.0],
            },
        }
    )
    path = tmp_path / "iafdb_bank_activation.h5"
    write_iafdb_bank(pyd_bank, path)
    return path


@pytest.fixture
def noise_bank_path(tmp_path: Path, fs_hz: float, n_samples: int) -> Path:
    """Build a tiny Pydantic NoiseBank and write it to HDF5.

    Four traces from two patients x two bipolar channels. The schema
    (v1.1) is intentionally minimal — only signal + source_record +
    source_channel per trace plus schema_version / created_utc /
    bank_id / source / fs_hz at the root. Extraction provenance lives
    in the sibling noise_bank_run_record JSON; see the records-side
    fixture for that.
    """
    rng = np.random.default_rng(2)
    n = 4
    signal = [rng.standard_normal(n_samples).astype(np.float32).tolist() for _ in range(n)]

    pyd_bank = noise_bank_models.NoiseBank.model_validate(
        {
            "schema_version": current_version("noise_bank"),
            "created_utc": _now_iso(),
            "bank_id": "nbank_iafdb_test_2026-07-31",
            "source": "iafdb v1.0.0",
            "fs_hz": fs_hz,
            "traces": {
                "signal": signal,
                "source_record": ["iaf1_afw"] * n,
                "source_channel": ["CS12", "CS34", "CS12", "CS34"],
            },
        }
    )
    path = tmp_path / "noise_bank.h5"
    write_noise_bank(pyd_bank, path)
    return path
