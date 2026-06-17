# myocard-egm-data

> I/O and dataset ergonomics for the intracardiac-EGM stack: readers and writers for every on-disk format defined by [`myocard-egm-contracts`](https://github.com/myocard-labs/egm-contracts), plus PyTorch `Dataset` wrappers, patient-aware splits, and the per-trace augmentation/normalization transform.

Part of the [myocard-labs](https://github.com/myocard-labs) cardiac signal-processing toolkit.

---

## Why

The on-disk formats (HDF5 banks, JSON run records, CSV metrics, etc.) are owned by [`myocard-egm-contracts`](https://github.com/myocard-labs/egm-contracts) as JSON Schemas. `myocard-egm-data` is the matching I/O layer: every writer produces a file that the contracts' file-level validators accept, and every reader returns a Python-friendly view of the same.

What this package contains:

- **Bank readers and writers** for the two HDF5 schemas — `synthetic_bank` (clean or hybrid clean+noise EGMs) and `iafdb_healthy_bank` (calibrated + band-passed segments). The reader auto-detects which schema a file conforms to.
- **Record readers and writers** for the JSON / CSV training artifacts — `run.json`, `metrics.csv`, `predictions_<eval_name>.{json,csv}`, `hybrid_eval_metrics.json`, and `model_metadata.json`.
- **PyTorch `Dataset` wrappers** including the `TraceTransform` per-trace normalize+pad+augment pipeline, the patient-aware split, and a `build_dataloaders` convenience that ties banks + splits + datasets together.

Schema versioning lives in `myocard-egm-contracts`; this package is the thin I/O layer over those schemas.

---

## Install

From source during pre-1.0 iteration:

```bash
pip install git+https://github.com/myocard-labs/egm-data.git
```

With the optional torch extras (needed for `myocard_egm_data.datasets`):

```bash
pip install "myocard-egm-data[torch] @ git+https://github.com/myocard-labs/egm-data.git"
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
from myocard_egm_data.banks import load_bank, write_synthetic_bank
from myocard_egm_data.records import write_run_record
from myocard_egm_data.datasets import build_dataloaders

# Auto-detects synthetic vs IAFDB schema.
bank = load_bank("data/hybrid_v1.h5")
print(bank.n_traces, bank.n_samples, bank.schema)

# Patient-aware split + per-trace transform + DataLoader, all in one call.
bundle = build_dataloaders("data/hybrid_v1.h5", batch_size=64)
for x, y in bundle.train:
    ...
```

The bank, record, splits, and augmentation packages do not import torch. Only `myocard_egm_data.datasets` does — a notebook or viewer can install without it.

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
