# Using myocard-egm-data

`myocard-egm-data` is the pure I/O layer on top of
`myocard-egm-contracts`. If you're building an eval script, a viewer,
or the data side of a training pipeline over intracardiac-EGM data,
this is the package you reach for first.

The headline type is `ClassifierBank` — the unified in-memory and
on-disk format every downstream component agrees on. Source-specific
banks (synthetic, IAFDB) are read into the Pydantic models from
`egm-contracts` and converted into ClassifierBank via the converters
in `myocard_egm_data.banks`.

## Install

During pre-1.0 iteration:

```bash
pip install "git+https://github.com/myocard-labs/egm-data.git"
```

egm-data is a pure I/O package — it does not depend on torch.

## Common workflows

### Load a source bank as a ClassifierBank

The most common entry point. Open a synthetic-bank HDF5 file produced
by the synthetic-EGM pipeline and get back a labeled `ClassifierBank`:

```python
from myocard_egm_data.banks import load_synthetic_bank_as_classifier

cb = load_synthetic_bank_as_classifier("data/synthetic_v1_5.h5")
print(cb.n_traces, cb.labels)  # 2000 {0: 'healthy', 1: 'fibrotic'}
```

Since `synthetic_bank` **2.0** the bank carries its own labels — a plain
int per trace, with the `{int: name}` map recorded per simulation
alongside the policy that produced it — so **no `label_fn` is needed**.
Supply one only to deliberately *re-label* a bank, e.g. collapsing a
multiclass severity bank to binary for a comparison run:

```python
import numpy as np

def to_binary(bank):
    raw = np.asarray([int(getattr(x, "root", x)) for x in bank.traces.label])
    return (raw > 0).astype(np.int64), {0: "healthy", 1: "fibrotic"}

binary = load_synthetic_bank_as_classifier("data/severity_v2.h5", label_fn=to_binary)
```

To treat a bank as **unlabeled** (pretraining, inference-time), pass a
`label_fn` that returns `None`. Note this changed at 2.0: *omitting*
`label_fn` used to mean "unlabeled" and now means "use the bank's own
labels".

```python
unlabeled = load_synthetic_bank_as_classifier("data/synthetic_v1_5.h5", label_fn=lambda b: None)
```

The IAFDB bank carries no labels at all, so there a `label_fn` is still
the only way to get them:

```python
from myocard_egm_data.banks import load_iafdb_bank_as_classifier

def iafdb_label_fn(bank):
    n = len(bank.traces.signal)
    return np.zeros(n, dtype=np.int64), {0: "healthy"}

iafdb = load_iafdb_bank_as_classifier("data/iafdb_v1.h5", label_fn=iafdb_label_fn)
```

### Read generation parameters (θ) from a synthetic bank

The `ClassifierBank` is a **source-agnostic ML compression**: signal,
label, and the keys needed to join back. Generation parameters live on
the `synthetic_bank`, which is a **parallel artifact sharing a key** —
not a thing the ClassifierBank is derived from. Read θ from the typed
bank, never from `trace_metadata`:

```python
from myocard_egm_data.banks import read_synthetic_bank_hdf5, simulation_configs

bank = read_synthetic_bank_hdf5("data/synthetic_v1_5.h5")

# The bank-scoped theta-spec: which knobs this sweep varied.
for knob in bank.generation_params.knobs:
    print(knob.path, knob.bounds, knob.role)

# Per-simulation config, as a row view keyed by simulation_id.
configs = simulation_configs(bank)
print(configs[0].substrate)     # typed Substrate variant
print(configs[0].label_policy)  # typed LabelPolicy variant
```

To line features up with θ per trace, join the two banks on
`simulation_id`:

```python
from myocard_egm_data.banks import join_traces_with_simulations

for pair in join_traces_with_simulations(cb, bank):
    trace, sim = pair.trace, pair.simulation
    ...  # trace.signal / trace.label_truth beside sim.substrate, sim.cell_model, ...
```

The join refuses mismatched banks, unknown `simulation_id`s and
duplicate simulation ids rather than returning partial results —
`simulation_id` restarts at 0 in every bank, so a mismatched join would
otherwise produce a full set of confident, wrong pairings.

### Read or write a noise bank

Noise banks are low-amplitude bipolar windows used as additive noise by
the synthetic-EGM mixer. The schema is intentionally minimal — the
mixer only consumes the signal and the per-trace source identifiers —
so the reader and writer are correspondingly thin:

```python
from myocard_egm_data.banks import read_noise_bank_hdf5, write_noise_bank

# Read: returns a Pydantic NoiseBank from myocard-egm-contracts.
nb = read_noise_bank_hdf5("data/iafdb_noise_v1.h5")
print(nb.fs_hz, len(nb.traces.signal))

# Write: producers (today: iafdb-pipeline) construct the Pydantic model
# and hand it to the writer.
write_noise_bank(nb, "out/copy.h5", overwrite=True)
```

There is **no ClassifierBank converter** for noise banks — noise is
input to the synthetic mixer, not to the classifier. If you ever need
to treat a noise bank as classifier input (e.g. unsupervised
pretraining), that converter is a deliberate later addition.

### Build and read a noise_bank_run_record (provenance sidecar)

A noise bank is paired with a JSON `noise_bank_run_record` sidecar
that captures extraction provenance — calibration scheme, threshold
strategy, filter band, windowing parameters, and optional per-trace
audit arrays. The mixer ignores the sidecar; debuggers, audits, and
the white paper's methods section open it. Convention is matching
name stem in the same directory (`foo_noise.h5` ↔
`foo_noise_run_record.json`).

The builder stamps `schema_version` and `created_utc` automatically
and takes substantive fields by keyword. It returns a typed
`NoiseBankRunRecord` Pydantic model (from `myocard-egm-contracts`);
the writer takes that model directly.

```python
from myocard_egm_data.records import (
    build_noise_bank_run_record,
    load_noise_bank_run_record,
    write_noise_bank_run_record,
)

record = build_noise_bank_run_record(
    source="iafdb v1.0.0",
    fs_hz=1000.0,
    window_ms=512.0,
    window_samples=512,
    hop_ms=256.0,
    band_hz=[30.0, 300.0],
    calibration_method="r_wave_anchoring",
    calibration_target_qrs_pp_mv=1.0,
    threshold_mode="percentile",
    threshold_value=10.0,
    source_records=["iaf1_afw", "iaf2_afw"],
    description="IAFDB-derived noise, calibrated to 1.0 mV QRS",
    # Optional but recommended for paper-citable banks:
    per_trace_provenance={
        "patient_id": [...],
        "start_sample": [...],
        "peak_to_peak_mv": [...],
        "calibration_scalar": [...],
    },
)
write_noise_bank_run_record("out/iafdb_noise_v1_run_record.json", record)

# Round-trip — load_* returns the typed NoiseBankRunRecord, not a dict.
again = load_noise_bank_run_record("out/iafdb_noise_v1_run_record.json")
assert again.calibration.method == "r_wave_anchoring"
assert again.windowing.window_samples == 512
```

Per-trace provenance is optional — producers building small test
fixtures can omit it; producers building anything paper-citable
should include it so future audits can answer "where did segment i
come from?" without going back to the upstream dataset.

### Build and read a training_run_record (the trainer's run.json)

The training-run record is the full, versioned per-run artifact
the trainer writes at end of training: schema version, run metadata,
the resolved training config, per-epoch records, the best epoch by
the configured selection metric, and an optional held-out test
block. The builder takes a list of typed `EpochRecord` instances and
returns a typed `TrainingRunRecord`:

```python
from myocard_egm_data.records import (
    EpochRecord,
    ReliabilityBin,
    build_training_run_record,
    load_training_run_record,
    write_training_run_record,
)

epochs = [
    EpochRecord(
        epoch=1, lr=1e-3, train_loss=0.5, val_loss=0.45, epoch_seconds=12.0,
        val_metrics={"auroc": 0.85, "accuracy": 0.80, "f1": 0.79, "ece": 0.05},
        # training_run_record 1.2: the train-split metrics, so the record
        # shows train-vs-val divergence. Optional — omit it and the field
        # is simply unset, which is what lets a producer adopt 1.2 before
        # it emits them.
        train_metrics={"auroc": 0.99, "accuracy": 0.97, "f1": 0.98, "ece": 0.01},
        val_reliability=[
            ReliabilityBin(lo=0.0, hi=0.5, count=10, confidence=0.25, accuracy=0.20),
            ReliabilityBin(lo=0.5, hi=1.0, count=10, confidence=0.75, accuracy=0.80),
        ],
    ),
    # ... one EpochRecord per epoch
]

record = build_training_run_record(
    # Artifact paths here should be repo-relative, not absolute: an
    # absolute path records the machine that trained, not the artifact.
    config={"input_length": 192, "batch_size": 64, "lr": 1e-3},
    # `host` is deliberately not a well-known key — it identified the
    # machine, which nothing downstream read.
    run_meta={"run_id": "v1-baseline", "git_sha": "abc1234"},
    epoch_records=epochs,
    select_metric="auroc",
    test_loss=0.40,
    test_metrics={"auroc": 0.86, "accuracy": 0.81, "reliability": []},
)
write_training_run_record("out/run.json", record)

# Round-trip — load_* returns the typed TrainingRunRecord.
loaded = load_training_run_record("out/run.json")
assert loaded.best.epoch == 1  # BestEpoch instance
assert loaded.test.loss == 0.40  # HeldOutTest instance
```

The companion `write_training_metrics` flattens those same epoch
records into `metrics.csv`:

```python
from myocard_egm_data.records import (
    load_training_metrics,
    write_training_metrics,
)

write_training_metrics("out/metrics.csv", epochs)
rows = load_training_metrics("out/metrics.csv")
assert rows[0].epoch == 1 and rows[0].val_auroc == 0.85
```

`training_metrics` has no `build_*` helper because the schema
describes one CSV row — the file is the concatenation of N rows
with a header, and the row's fields all come from the source
`EpochRecord`s directly.

### Build and read an egm_class_model_metadata sidecar

The EGM-classifier `model_metadata.json` sidecar pairs with a deployed
model artifact (typically the ONNX export). It captures the
deployment-time constants the inference runtime needs — sample rate,
trace length, per-trace normalization scheme, decision threshold,
model artifact hash, training provenance. The schema is specific to
the 1-D EGM-classifier family; future 2-D electrode-grid or
sparse-3-D electrode models get their own per-topology schemas.

```python
from myocard_egm_data.records import (
    build_egm_class_model_metadata,
    load_egm_class_model_metadata,
    write_egm_class_model_metadata,
)

record = build_egm_class_model_metadata(
    model_artifact={
        "filename": "best.onnx",
        "framework": "onnx",
        "sha256": "a" * 64,
    },
    input_spec={"name": "signal", "shape": ["?", 1, 512], "dtype": "float32"},
    output_spec={
        "name": "logit", "shape": ["?", 1], "dtype": "float32",
        "semantics": "binary_logit",
    },
    preprocessing={
        "expected_fs_hz": 1000.0,
        "expected_trace_samples": 512,
        "bandpass_hz": [30.0, 300.0],
        "normalization": {"scheme": "zscore"},
    },
    decision={"threshold": 0.5, "class_labels": ["healthy", "fibrotic"]},
    training_provenance={"run_id": "v1-baseline", "training_bank_path": "data/hybrid_v1.h5"},
)
write_egm_class_model_metadata("out/best.model_metadata.json", record)
```

Each sub-block accepts either a typed Pydantic instance (e.g. a
`ModelArtifact` pulled from a registry) or a plain dict — Pydantic
validates dicts into the nested sub-models at construction time.

### Read or write phase artifacts (manifest / observation / figure spec)

The `phases` subpackage is the typed I/O for the cross-artifact-linkage
JSON formats (egm-contracts v0.5.0) that organize a project phase's
artifacts. egm-studio is their canonical curator; this is the layer it
(and `intracardiac-platform/scripts/validate_manifest.py`) read and write
them through.

```python
from myocard_egm_data.phases import (
    load_phase_manifest, write_phase_manifest,
    Observation, write_observation,
)

# Load, inspect, and re-write a per-phase manifest.
manifest = load_phase_manifest("phases/phase_1_5/manifest.json")
print(len(manifest.egm_banks), len(manifest.noise_banks))
write_phase_manifest("phases/phase_1_5/manifest.json", manifest)

# Construct + write an observation (a free-text `description` is required).
obs = Observation.model_validate({
    "schema_version": "1",
    "id": "obs_courtemanche_high_entropy_tail_2026-06-27",
    "date": "2026-06-27",
    "title": "high-entropy tail",
    "description": "Synthetic sample_entropy has a long tail IAFDB never reaches.",
})
write_observation("phases/phase_1_5/observations/obs_courtemanche_high_entropy_tail_2026-06-27.json", obs)
```

Every `load_*` returns a typed Pydantic model — an invalid id pattern, a
missing `description`, or an unknown key raises `pydantic.ValidationError`;
every `write_*` emits strict, deterministic JSON with optional-defaulted
fields omitted.

### Combine multiple banks into a hybrid bank

Use `ClassifierBank.concat` to build a hybrid (synthetic + real) bank
in one call. Concat refuses to merge banks with disagreeing `labels`
dicts — by design, integer labels must mean the same thing in every
contributing bank.

```python
from myocard_egm_data.banks import ClassifierBank

# Both banks below must use the same labels dict.
hybrid = ClassifierBank.concat([synthetic_bank, iafdb_bank])
```

### Training: splits, augmentation, DataLoaders

The torch-based training-data layer — patient-aware splits, the
`TraceTransform` augmentation, `EGMTraceDataset`, and the
`build_dataloaders` convenience — moved to its sole consumer,
**egm-classifier** (`myocard_egm_classifier.data`), in the Refactor Step 8
code-placement audit. See egm-classifier's `docs/usage.md` for the
training-pipeline walkthrough. egm-data now stops at the `ClassifierBank`:
it reads banks, converts them to the unified in-memory shape, and writes
the run records / predictions the classifier emits.

### Save and load a ClassifierBank

```python
from myocard_egm_data.banks import write_classifier_bank, load_classifier_bank

write_classifier_bank(hybrid, "out/hybrid_eval.classifier.h5")
again = load_classifier_bank("out/hybrid_eval.classifier.h5")
```

The on-disk format is documented in `project/classifier_bank_format.md`
(internal); the salient points are HDF5, JSON-encoded metadata dicts,
and sentinel-mask columns so `label_truth=None` round-trips distinct
from `label_truth=0`.

## Module map

| Module | What's in it |
|---|---|
| `myocard_egm_data.banks` | `ClassifierBank` + per-trace types, converters from Pydantic `SyntheticBank` / `IafdbBank`, `read_*_hdf5` Pydantic readers (synthetic / iafdb / noise), `write_*` Pydantic writers (synthetic / iafdb / noise), ClassifierBank HDF5 I/O, the `simulation_configs` / `join_traces_with_simulations` bank-join, and the stable-id content checks |
| `myocard_egm_data.records` | One per-file module per schema, mirroring the per-schema layout in `myocard-egm-contracts._generated.python`: `training_run_record` (run.json), `training_metrics` (metrics.csv), `egm_class_model_metadata` (1-D EGM-classifier inference sidecar), `noise_bank_run_record` (noise-bank provenance sidecar). Each module owns `build_*` (where applicable) + `write_*` + `load_*` and re-exports its Pydantic models |
| `myocard_egm_data.phases` | Typed I/O for the cross-artifact-linkage JSON formats (egm-contracts v0.5.0): `phase_manifest` (per-phase `manifest.json`), `observation`, `figure_spec`. Each module owns `load_*` / `write_*` and re-exports its Pydantic models |

> The torch-based training-data layer (`datasets`, `splits`, `augmentation`) moved to **egm-classifier** (`myocard_egm_classifier.data`) in the Refactor Step 8 code-placement audit.

## Where to read more

- For the on-disk ClassifierBank format details:
  `project/classifier_bank_format.md`.
- For schema definitions — `synthetic_bank`, `iafdb_bank`,
  `noise_bank`, `noise_bank_run_record`, `training_run_record`,
  `training_metrics`, `egm_class_model_metadata` — see the JSON Schema files in
  `myocard-egm-contracts`, with prose companions under
  `docs/schemas/`.
- For the noise_bank ↔ noise_bank_run_record pairing convention and
  the reason the bank schema is intentionally minimal: the
  `noise_bank` and `noise_bank_run_record` entries under
  `myocard-egm-contracts/docs/schemas/`.
- For training-time pre-processing rationale:
  `project/trace_transform_review.md`.
