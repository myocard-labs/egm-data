# myocard-egm-data

> Pure I/O for the intracardiac-EGM stack: readers and writers for every on-disk format defined by [`myocard-egm-contracts`](https://github.com/myocard-labs/egm-contracts) — HDF5 banks, run records, predictions, and phase-manifest JSON.

Part of the [myocard-labs](https://github.com/myocard-labs) cardiac signal-processing toolkit.

---

## Why

The on-disk formats (HDF5 banks, JSON run records, CSV metrics, etc.) are owned by [`myocard-egm-contracts`](https://github.com/myocard-labs/egm-contracts) as JSON Schemas. `myocard-egm-data` is the matching I/O layer: every writer produces a file that the contracts' file-level validators accept, and every reader returns a Python-friendly view of the same.

What this package contains:

- **Bank readers and writers** for the HDF5 schemas — `synthetic_bank` (clean or noise-mixed EGMs), `iafdb_bank` (calibrated + band-passed segments), and `noise_bank` (low-amplitude windows used as additive noise by the synthetic mixer). Source-specific Pydantic models from `egm-contracts` get converted into the unified in-memory `ClassifierBank`. All banks carry stable cross-artifact ids (egm-contracts v0.5.0): readers surface them, and producer-bank writers require them on new files.
- **Record readers and writers** for the JSON / CSV training and evaluation artifacts — `training_run_record` (run.json), `training_metrics` (metrics.csv), `egm_class_model_metadata` (inference-side model metadata for the 1-D EGM-classifier family), and `noise_bank_run_record` (extraction provenance sidecar for a noise bank). Every `build_*` returns a typed Pydantic model; every `write_*` accepts one; every `load_*` returns one.
- **Phase-artifact readers and writers** (`myocard_egm_data.phases`) for the cross-artifact-linkage JSON formats added in egm-contracts v0.5.0 — the per-phase `manifest.json`, observations, and figure specs that organize a project phase's artifacts. Typed `load_*` / `write_*` over each schema.
Schema versioning lives in `myocard-egm-contracts`; this package is the thin I/O layer over those schemas. The torch-based training-data layer (`Dataset` wrappers, patient-aware split, per-trace augmentation) lives in its sole consumer, [`myocard-egm-classifier`](https://github.com/myocard-labs/egm-classifier).

---

## Install

From source during pre-1.0 iteration:

```bash
pip install git+https://github.com/myocard-labs/egm-data.git
```

Editable install for development:

```bash
git clone https://github.com/myocard-labs/egm-data.git
cd egm-data
pip install -e ".[dev]"
pre-commit install
```

---

## Programmatic usage

```python
from myocard_egm_data.banks import load_synthetic_bank_as_classifier
from myocard_egm_data.records import build_training_run_record, write_training_run_record

# Load a synthetic bank and convert it to the unified ClassifierBank.
cb = load_synthetic_bank_as_classifier("data/noise_mixed_v1.h5")
print(cb.n_traces, cb.n_samples_first)

# At end of training, write the run record as a typed Pydantic model.
record = build_training_run_record(
    config=..., run_meta=..., epoch_records=[...], select_metric="auroc",
)
write_training_run_record("out/run.json", record)
```

See [docs/usage.md](docs/usage.md) for full walkthroughs of the per-schema build/write/load helpers.

This is a pure I/O package — it does not import torch, so a notebook, viewer, or analysis script can install it without pulling in torch. The torch-based training-data layer lives in [`myocard-egm-classifier`](https://github.com/myocard-labs/egm-classifier).

---

## Tests

```bash
pytest                  # full suite
pytest --cov            # with coverage
ruff check .            # lint
ruff format --check .   # format check
mypy                    # type check
```

The round-trip tests construct fixtures, write them through this package's writers, validate them with the contracts' file-level validators, then read them back and assert structure. If a writer drifts from a schema the validator will catch it.

CI runs the same checks on Python 3.10, 3.11, and 3.12 — see `.github/workflows/ci.yml`.

---

## Project status

Part of the in-progress [myocard-labs](https://github.com/myocard-labs) refactor. Pre-1.0 — expect breaking changes across minor versions until the schema/API stabilizes. See `intracardiac-platform/project/project_plan.md` for the roadmap.

---

## Citation

```bibtex
@software{klein_myocard_egm_data_2026,
  author  = {Klein, Daniel},
  title   = {myocard-egm-data: I/O and dataset ergonomics for intracardiac EGM formats},
  year    = {2026},
  url     = {https://github.com/myocard-labs/egm-data},
}
```

---

## License

MIT — see [LICENSE](LICENSE). Attribution requirements for any data sources surfaced through this package are listed in [NOTICE](NOTICE).
