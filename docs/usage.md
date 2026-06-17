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
| `myocard_egm_data.banks` | `ClassifierBank` + per-trace types, converters from Pydantic SyntheticBank / IafdbBank, `read_*_hdf5` Pydantic readers, `write_*` Pydantic writers, ClassifierBank HDF5 I/O |
| `myocard_egm_data.records` | run.json / metrics.csv / hybrid_eval_metrics.json / model_metadata.json writers and readers |
| `myocard_egm_data.splits` | `patient_aware_split` (numpy-array level) and `split_classifier_bank` / `apply_split_indices` (ClassifierBank-level) |
| `myocard_egm_data.augmentation` | `TraceTransform` — per-trace normalize + pad + augment, used in the DataLoader pipeline |
| `myocard_egm_data.datasets` | PyTorch `Dataset` wrappers and `build_dataloaders` (requires `[torch]` extra) |

## Where to read more

- For the on-disk ClassifierBank format details:
  `project/classifier_bank_format.md`.
- For schema definitions (synthetic_bank, iafdb_bank,
  run_record, metrics, hybrid_eval_metrics, model_metadata):
  the JSON Schema files in `myocard-egm-contracts`.
- For training-time pre-processing rationale:
  `project/trace_transform_review.md`.
