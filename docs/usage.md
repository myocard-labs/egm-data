# Using myocard-egm-data

`myocard-egm-data` is the I/O and dataset-ergonomics layer on top of
`myocard-egm-contracts`. If you're building a training pipeline, an
eval script, or a viewer over intracardiac-EGM data, this is the
package you reach for first.

The headline type is `ClassifierBank` — the unified in-memory and
on-disk format every downstream component agrees on. Source-specific
banks (synthetic, IAFDB) are read into the Pydantic models from
`egm-contracts` and converted into ClassifierBank via the converters
in `myocard_egm_data.banks`.

## Install

During pre-1.0 iteration:

```bash
pip install "myocard-egm-data[torch] @ git+https://github.com/myocard-labs/egm-data.git"
```

The `torch` extra is only needed for `myocard_egm_data.datasets`; the
banks / records / splits / augmentation subpackages are torch-free.

## Common workflows

### Load a source bank as a ClassifierBank

The most common entry point. Open a synthetic-bank HDF5 file produced
by the synthetic-EGM pipeline, apply a labeling policy, and get back a
labeled `ClassifierBank`:

```python
import numpy as np
from myocard_egm_data.banks import load_synthetic_bank_as_classifier

def label_fn(bank):
    # The Pydantic SyntheticBank exposes per-trace columns.
    labels = (np.asarray(bank.traces.fibrosis_density) > 0.0).astype(np.int64)
    return labels, {0: "healthy", 1: "fibrotic"}

cb = load_synthetic_bank_as_classifier(
    "data/hybrid_v1.h5",
    label_fn=label_fn,
)
print(cb.n_traces, cb.labels)  # 2000 {0: 'healthy', 1: 'fibrotic'}
```

The labeling policy is *your* choice — the library does not pre-decide
what counts as "fibrotic". If you omit `label_fn` (or it returns
`None`), every `ClassifierTrace.label_truth` ends up as `None` and the
`labels` dict is empty. Useful for pretraining or inference-time use
on unlabeled data:

```python
unlabeled = load_synthetic_bank_as_classifier("data/hybrid_v1.h5")
```

The IAFDB side is symmetric:

```python
from myocard_egm_data.banks import load_iafdb_bank_as_classifier

def iafdb_label_fn(bank):
    n = len(bank.traces.signal)
    return np.zeros(n, dtype=np.int64), {0: "healthy"}

iafdb = load_iafdb_bank_as_classifier("data/iafdb_v1.h5", label_fn=iafdb_label_fn)
```

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
        val_reliability=[
            ReliabilityBin(lo=0.0, hi=0.5, count=10, confidence=0.25, accuracy=0.20),
            ReliabilityBin(lo=0.5, hi=1.0, count=10, confidence=0.75, accuracy=0.80),
        ],
    ),
    # ... one EpochRecord per epoch
]

record = build_training_run_record(
    config={"input_length": 512, "batch_size": 64, "lr": 1e-3},
    run_meta={"run_id": "v1-baseline", "git_sha": "abc1234", "host": "workstation"},
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
trace length, per-channel normalization, decision threshold, model
artifact hash, training provenance. The schema is specific to the
1-D EGM-classifier family; future 2-D electrode-grid or sparse-3-D
electrode models get their own per-topology schemas.

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
        "normalization": {"scheme": "zscore", "mean": [0.0], "std": [1.0]},
    },
    decision={"threshold": 0.5, "class_labels": ["healthy", "fibrotic"]},
    training_provenance={"run_id": "v1-baseline", "training_bank_path": "data/hybrid_v1.h5"},
)
write_egm_class_model_metadata("out/best.model_metadata.json", record)
```

Each sub-block accepts either a typed Pydantic instance (e.g. a
`ModelArtifact` pulled from a registry) or a plain dict — Pydantic
validates dicts into the nested sub-models at construction time.

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

### Split for training

Run a patient-aware split and write the split assignment back onto each trace:

```python
from myocard_egm_data.splits import split_classifier_bank

indices = split_classifier_bank(
    hybrid,
    fractions=(0.8, 0.1, 0.1),
    seed=42,
)
# Every trace now has trace.split set to "train", "val", or "test".
```

The splitter reads `patient_id` from each trace's `trace_metadata`. If
your custom `bank_type` doesn't follow that convention, populate the
key before calling.

### Build training DataLoaders

```python
from myocard_egm_data.datasets import build_dataloaders

bundle = build_dataloaders(
    hybrid,
    input_length=512,
    batch_size=64,
    num_workers=4,
    binary=True,
    znorm=True,
    znorm_eps=1e-6,
    augment_train=True,
    max_gain=0.0,
    max_shift_frac=0.1,
    split_fractions=(0.8, 0.1, 0.1),
    split_seed=42,
    dataset_seed=0,
    pin_memory=True,
)

for signal, label in bundle.train:
    ...
```

Every kwarg is required — the library ships no defaults for training
policy. The classifier code (`egm-classifier`) holds these in its own
config layer.

### Save and load a ClassifierBank

```python
from myocard_egm_data.banks import write_classifier_bank, load_classifier_bank

write_classifier_bank(hybrid, "out/hybrid_eval.cbank.h5")
again = load_classifier_bank("out/hybrid_eval.cbank.h5")
```

The on-disk format is documented in `project/classifier_bank_format.md`
(internal); the salient points are HDF5, JSON-encoded metadata dicts,
and sentinel-mask columns so `label_truth=None` round-trips distinct
from `label_truth=0`.

## Module map

| Module | What's in it |
|---|---|
| `myocard_egm_data.banks` | `ClassifierBank` + per-trace types, converters from Pydantic `SyntheticBank` / `IafdbBank`, `read_*_hdf5` Pydantic readers (synthetic / iafdb / noise), `write_*` Pydantic writers (synthetic / iafdb / noise), ClassifierBank HDF5 I/O |
| `myocard_egm_data.records` | One per-file module per schema, mirroring the per-schema layout in `myocard-egm-contracts._generated.python`: `training_run_record` (run.json), `training_metrics` (metrics.csv), `hybrid_eval_metrics` (mixed eval summary), `egm_class_model_metadata` (1-D EGM-classifier inference sidecar), `noise_bank_run_record` (noise-bank provenance sidecar). Each module owns `build_*` (where applicable) + `write_*` + `load_*` and re-exports its Pydantic models |
| `myocard_egm_data.splits` | `patient_aware_split` (numpy-array level) and `split_classifier_bank` / `apply_split_indices` (ClassifierBank-level) |
| `myocard_egm_data.augmentation` | `TraceTransform` — per-trace normalize + pad + augment, used in the DataLoader pipeline |
| `myocard_egm_data.datasets` | PyTorch `Dataset` wrappers and `build_dataloaders` (requires `[torch]` extra) |

## Where to read more

- For the on-disk ClassifierBank format details:
  `project/classifier_bank_format.md`.
- For schema definitions — `synthetic_bank`, `iafdb_bank`,
  `noise_bank`, `noise_bank_run_record`, `training_run_record`,
  `training_metrics`, `hybrid_eval_metrics`,
  `egm_class_model_metadata` — see the JSON Schema files in
  `myocard-egm-contracts`, with prose companions under
  `docs/schemas/`.
- For the noise_bank ↔ noise_bank_run_record pairing convention and
  the reason the bank schema is intentionally minimal: the
  `noise_bank` and `noise_bank_run_record` entries under
  `myocard-egm-contracts/docs/schemas/`.
- For training-time pre-processing rationale:
  `project/trace_transform_review.md`.
