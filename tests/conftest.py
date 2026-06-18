"""Shared fixtures — Pydantic bank builders that the writers serialize.

We build the source-bank artifacts (synthetic, IAFDB, noise) by
constructing the Pydantic models codegen'd from egm-contracts schemas
and handing them to ``write_synthetic_bank`` / ``write_iafdb_bank`` /
``write_noise_bank``. There is no parallel set of dataclasses in
egm-data anymore; the contracts package is the only place those types
live.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from myocard_egm_contracts import iafdb_bank as iafdb_bank_models
from myocard_egm_contracts import noise_bank as noise_bank_models
from myocard_egm_contracts import synthetic_bank as synthetic_bank_models

from myocard_egm_data.banks import write_iafdb_bank, write_noise_bank, write_synthetic_bank


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


@pytest.fixture
def synthetic_bank_path(
    tmp_path: Path, fs_hz: float, trace_duration_ms: float, n_samples: int
) -> Path:
    """Build a tiny Pydantic SyntheticBank and write it to HDF5.

    Two patients x three traces each. Patient 0 healthy (density 0.0),
    patient 1 fibrotic (density 0.3). Signals random — physiological
    realism isn't needed for the I/O tests.
    """
    rng = np.random.default_rng(0)
    n = 6
    signal = [rng.standard_normal(n_samples).astype(np.float32).tolist() for _ in range(n)]
    sim_id = [0, 0, 0, 1, 1, 1]
    pair_index = [0, 1, 2, 0, 1, 2]
    electrode_row = [0, 0, 1, 0, 0, 1]
    densities = [0.0, 0.0, 0.0, 0.3, 0.3, 0.3]

    pyd_bank = synthetic_bank_models.SyntheticBank.model_validate(
        {
            "schema_version": "1.0",
            "created_utc": _now_iso(),
            "description": "Synthetic bank test fixture",
            "fs_hz": fs_hz,
            "trace_duration_ms": trace_duration_ms,
            "simulator": "finitewave",
            "cell_model": "aliev_panfilov",
            "patch_size_mm": 40.0,
            "patch_dr_mm": 0.25,
            "ap_time_unit_ms": 12.9,
            "fibrosis_strategy_name": "uniform_random",
            "fibrosis_params": {"density_min": 0.0, "density_max": 0.5},
            "electrode_config": {"grid_rows": 5, "grid_cols": 5, "spacing_mm": 2.0},
            "mixer_config": {"snr_db_min": 10.0, "snr_db_max": 25.0},
            "experiment_config": {"name": "test_fixture"},
            "noise_bank_source": "iafdb_noise_v1.h5",
            "traces": {
                "signal": signal,
                "simulation_id": sim_id,
                "pair_index": pair_index,
                "electrode_row": electrode_row,
                "fibrosis_density": densities,
                "fibrosis_density_realized": densities,
                "electrode_height_mm": [0.5] * n,
                "seed": [i * 100 for i in range(n)],
                "snr_db": [15.0] * n,
                "stim_edge": ["left"] * n,
                "noise_record": ["iaf1_afw"] * n,
                "noise_channel": ["CS12"] * n,
            },
        }
    )
    path = tmp_path / "synthetic_bank.h5"
    write_synthetic_bank(pyd_bank, path)
    return path


@pytest.fixture
def iafdb_bank_path(tmp_path: Path, fs_hz: float, trace_duration_ms: float, n_samples: int) -> Path:
    """Build a tiny Pydantic IafdbBank and write it to HDF5.

    Two patients x two segments. The bank schema (v0.1.2) carries no
    label column — labeling happens at ClassifierBank conversion time.
    """
    rng = np.random.default_rng(1)
    n = 4
    signal = [rng.standard_normal(n_samples).astype(np.float32).tolist() for _ in range(n)]
    patient_id = ["iaf1", "iaf1", "iaf2", "iaf2"]
    source_channel = ["CS12", "CS34", "CS12", "CS34"]

    pyd_bank = iafdb_bank_models.IafdbBank.model_validate(
        {
            "schema_version": "1.1",
            "created_utc": _now_iso(),
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
def noise_bank_path(tmp_path: Path, fs_hz: float, n_samples: int) -> Path:
    """Build a tiny Pydantic NoiseBank and write it to HDF5.

    Four traces from two patients x two bipolar channels. The schema
    (v1.0) is intentionally minimal — only signal + source_record +
    source_channel per trace plus schema_version / created_utc / source
    / fs_hz at the root. Extraction provenance lives in the sibling
    noise_bank_run_record JSON; see the records-side fixture for that.
    """
    rng = np.random.default_rng(2)
    n = 4
    signal = [rng.standard_normal(n_samples).astype(np.float32).tolist() for _ in range(n)]

    pyd_bank = noise_bank_models.NoiseBank.model_validate(
        {
            "schema_version": "1.0",
            "created_utc": _now_iso(),
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
