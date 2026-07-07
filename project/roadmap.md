# egm-data — roadmap

What's planned for future releases. Internal doc — public users see
the README and `docs/usage.md`. egm-data sits between egm-contracts
(the schema layer, no I/O) and downstream producers + consumers (which
shouldn't touch HDF5 / CSV / JSON directly); roadmap items here are
mostly driven by either (a) cascading egm-contracts bumps or (b)
consumer-side ergonomics requests.

Items scheduled into cross-cutting Phase work in the meta repo's
`project_plan.md` carry a `→ tracked at intracardiac-platform Phase X`
annotation; the rest are component-internal — driven by what consumers
actually need.

## v0.5.0 (shipped) — training-data layer removed

The torch-based training-data layer moved to its sole consumer,
**egm-classifier** (`myocard_egm_classifier.data`), per the Refactor Step 8
code-placement audit. egm-data is now a **pure I/O** library:

- **Removed** `datasets/` (`EGMTraceDataset`, `build_dataloaders`,
  `LoaderBundle`), `splits/` (`patient_aware_split` + strategies), and
  `augmentation/` (`TraceTransform`) — plus their tests.
- **Dropped** the `[torch]` optional extra and the `torch.*` mypy override;
  egm-data no longer imports torch anywhere.
- **Consumers** re-pin `myocard-egm-data[torch] v0.4.x` to `myocard-egm-data v0.5.0`
  (no extra).

Future work on the training-data layer now lives in egm-classifier's
roadmap (variable-length / multi-beat datasets, the `normalize` mode,
splitting-strategy choices, streaming loaders, K-fold).

## v0.3.x (shipped)

Scope (recap; see `project/classifier_bank_format.md` and the README
for the per-subpackage contract):

- **`banks/`** — readers + writers for every contracts-defined HDF5
  bank format. `ClassifierBank` is the unified consumer-side bank
  shape (produced by both iafdb-pipeline and synthetic-egm-pipeline);
  `IafdbBank`, `SyntheticBank`, `NoiseBank` are the producer-side
  bank shapes. `BankBase` + per-schema subclasses + the bank
  `Protocol` (added in the v0.2.x bank-type redesign).
- **`records/`** — readers + writers for the JSON / CSV records:
  `training_run_record` (`run.json`), `training_metrics`
  (`metrics.csv` per-epoch rows), `egm_class_model_metadata`.
  Records are Pydantic-model-in /
  Pydantic-model-out — readers return typed contracts models;
  writers take typed contracts models.
- Every cross-module boundary uses typed contracts Pydantic models
  per [[feedback-use-contracts-at-boundaries]] — no dicts, no mirror
  dataclasses.

Pinned dependency:

```
myocard-egm-contracts @ git+...@v0.5.2
```

## Shipped after v0.4.0

- **v0.4.1** — `ClassifierBank.uniform_fs_hz()` accessor for single-rate banks
  (+ test).
- **v0.4.2** — re-pin egm-contracts `v0.5.1 → v0.5.2` (the generated role
  vocabulary) and add `phases.load_phase_dir(phase_dir)`, which reads
  `<phase_dir>/manifest.json` through `load_phase_manifest` so a consumer points
  at a phase folder and gets the typed `PhaseManifest` (the phase-folder layout
  lives with the reader, not each caller). Additive.

## v0.4.0 — current release (shipped)

Cascades the egm-contracts v0.5.0 / v0.5.1 cross-artifact linkage into
egm-data. See
`intracardiac-platform/project/cross_artifact_linkage_design.md`.

- **Re-pinned** myocard-egm-contracts to `v0.5.1`.
- **Producer-bank ids surfaced + enforced.** Bank readers surface the new
  optional `bank_id` (None on legacy banks); `write_synthetic_bank` /
  `write_iafdb_bank` require it on new writes. Record `build_*` helpers
  accept the new id / pointer fields (`run_id` / `trained_on_bank_id` /
  `produced_model_id` on the run record, `model_id` on the model sidecar,
  `bank_id` on the noise-bank sidecar).
- **ClassifierBank → 0.2.** Added the bank's own optional stable `id`
  (validated via `common.ArtifactId`), and reworked the per-source /
  per-trace `bank_id` from an integer index into the source bank's stable
  ArtifactId — collapsing the dual id concepts and simplifying `concat`
  (dedup by stable id, no remap). Clean break: 0.1 banks are not read.
- **New `phases/` subpackage** — typed `load_*` / `write_*` for the three
  cross-artifact-linkage JSON formats (`phase_manifest`, `observation`,
  `figure_spec`).
- **Promoted** the shared Pydantic <-> JSON helpers to a package-level
  `_serialization` module, shared by `records/` and `phases/`.

## v0.5.0+ — cascading from egm-contracts schema bumps

When egm-contracts ships a schema change, egm-data needs the
corresponding reader/writer update before any producer or consumer can
adopt. See egm-contracts roadmap "Schema bumps to coordinate" for the
cascade order (egm-contracts → egm-data → producers → consumers).

The known upcoming egm-contracts bumps and the egm-data work each
requires:

### Polymorphic `stimulation` (egm-contracts v0.6.0) — Phase 2

`synthetic_bank` writer needs to encode the polymorphic `stimulation`
object (type discriminator + per-type params); reader needs to decode.
The Pydantic model from contracts handles validation; egm-data's
writer just needs to serialize the model to HDF5 attrs in a way that
round-trips through the reader.

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 2.

### `iafdb_bank` 1.3 + audit-report sidecar (egm-contracts) — Phase 1.5

When egm-contracts adds the `run_record_path` field to `iafdb_bank`
attrs (and possibly a new `iafdb_bank_run_record` schema for the
sidecar itself), egm-data ships:

- Updated `iafdb_bank` writer that accepts an optional sidecar path
  argument and stamps it into attrs.
- New reader + writer for the sidecar schema (whether it's a fresh
  schema or a reuse of `noise_bank_run_record`'s structure).

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 1.5.

### `noise_bank` 1.1 + `noise_bank_run_record` 1.2 (egm-contracts) — Phase 1.5

When egm-contracts adds the `calibration_scalar` per-trace column and
the `per_trace_provenance.lead` field, egm-data updates the noise-bank
reader + writer to round-trip the new fields.

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 1.5.

### `egm_features_bank` (egm-contracts) — Refactor Step 4

When egm-contracts adds the features-bank schema, egm-data ships
the reader + writer. Any torch `Dataset` wrapper over features lives
in the consumer (egm-classifier), consistent with the Step 8 placement
of the training-data layer.

> → Tracked at `intracardiac-platform/project/refactor_checklist.md` Phase 4.

### 3D geometry schema (egm-contracts) — Phase 7

If egm-contracts adds a `synthetic_bank` schema field referencing a 3D
mesh, egm-data needs to handle the mesh reference (probably a relative
path + content hash, not the mesh data itself — mesh files are too big
to embed in the bank). Possibly also a separate `MeshReader` if we
decide egm-data should own 3D mesh I/O.

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 7.

## Won't-do (out of scope, but documented to save the question)

- **No DSP primitives in this repo.** Filtering, calibration,
  threshold extraction live in `myocard-egm-signal`. egm-data
  consumes signal-processing primitives; it doesn't implement them.
- **No feature engineering.** Per-trace feature computation lives in
  `myocard-egm-features` (Refactor Step 4, pending). egm-data
  reads/writes feature banks once the schema is defined but doesn't
  compute the features.
- **No model code.** egm-data is pure I/O. The training-data layer
  (datasets + loaders + augmentation, Step 8), model architecture,
  training loops, and ONNX export all live in egm-classifier.
- **No producer-specific logic.** Bank readers + writers are
  symmetric — they don't know whether a bank came from iafdb-pipeline
  or synthetic-egm-pipeline. Producer-specific transformations
  (e.g., IAFDB's R-wave-anchored calibration) belong in the producer
  repo, not here. egm-data takes already-calibrated, already-segmented
  signal and writes / reads the bank format.

## Open architectural questions for later

- **Should `records/` grow a versioned-write helper** that
  automatically writes the schema_version attr alongside the record?
  Today the contracts Pydantic model carries the version; the writer
  serializes it transparently. A helper that emits the schema version
  to a sidecar `.version` file could simplify cross-language
  consumption (C++ deployment doesn't want to parse a full JSON to
  know if it can read the file).
