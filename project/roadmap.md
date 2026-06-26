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

## v0.3.3 — current release (shipped)

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
  (`metrics.csv` per-epoch rows), `hybrid_eval_metrics`,
  `egm_class_model_metadata`. Records are Pydantic-model-in /
  Pydantic-model-out — readers return typed contracts models;
  writers take typed contracts models.
- **`datasets/`** — `EGMTraceDataset` (PyTorch `Dataset` over a
  `ClassifierBank`), `LoaderBundle` (train + val + test loaders
  wrapped with consistent collate + worker config).
- **`splits/`** — `patient_aware_split` (pluggable stratification
  strategies: `AnyPositive`, `BinnedDensity`).
- **`augmentation/`** — `TraceTransform` (per-call train-time
  augmentation: pad/crop, per-trace z-score, optional gain + time-shift
  augmentations).
- Every cross-module boundary uses typed contracts Pydantic models
  per [[feedback-use-contracts-at-boundaries]] — no dicts, no mirror
  dataclasses.

Pinned dependency:

```
myocard-egm-contracts @ git+...@v0.4.0
```

`[torch]` extra pulls torch + torchmetrics for the dataset wrappers.

## v0.3.x — planned additive bumps

### Variable-length / longer-T `EGMTraceDataset` support — Phase 4

Today's `EGMTraceDataset` assumes every trace is exactly the trained
`input_length` samples (default 512 at 1 kHz). Phase 4's multi-beat
sequence classification work will want longer traces (multi-cycle
windows) and possibly variable-length traces (different beat counts
per window). Two questions to settle:

- Does the dataset emit fixed-length multi-beat windows (e.g. 4
  consecutive beats padded to a uniform length), or variable-length
  (each window is exactly as long as N beats took)?
- Does the loader collate need a custom `collate_fn` to pad variable-
  length batches, or does it stay simple at the cost of imposing
  fixed-length windows on the producer?

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 4 (multi-beat sequence classification). Kickoff discussion will pick the shape.

### `TraceTransform` `normalize` mode (`zscore` / `zero2one` / `none`) — Phase 1.5

`TraceTransform` currently exposes a `znorm: bool` knob (true = per-trace
z-score, false = no normalization). The deployment-side metadata schema
already supports `zscore` / `zero2one` / `none` (egm-contracts v0.4.0);
training + eval need to match. Plan:

- Replace `znorm: bool` with `normalize: Literal["zscore", "zero2one", "none"]`.
- Add a `normalize_eps: float` knob for the per-trace divisor floor
  (already used by zscore).
- Backwards-compat shim: `znorm=True` translates to `normalize="zscore"`
  for one minor-version window, then removed.

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 1.5. Pairs with egm-classifier's Phase 1.5 zero2one work (the classifier plumbs `normalize` through its YAML configs; egm-data implements the per-trace math).

### `TraceTransform` relocation question — Refactor Step 8 (code-placement audit)

`TraceTransform` is currently in `egm-data/src/myocard_egm_data/augmentation/`
but its only consumer is egm-classifier (training + eval data loaders,
plus the export-side calibration loop). If no other consumer
materializes by the Refactor Step 8 audit time, the right move is
relocate it to egm-classifier (where its single consumer lives).

> → Tracked at `intracardiac-platform/project/refactor_checklist.md` Phase 8 (cleanup + verification — explicit code-placement audit task lists TraceTransform as a known candidate). Also tracked as task #287. The audit happens at Phase 8 rather than during egm-features scaffolding (Phase 4) so it can be comprehensive across all repos instead of piecemeal.

## v0.4.0+ — cascading from egm-contracts schema bumps

When egm-contracts ships a schema change, egm-data needs the
corresponding reader/writer update before any producer or consumer can
adopt. See egm-contracts roadmap "Schema bumps to coordinate" for the
cascade order (egm-contracts → egm-data → producers → consumers).

The known upcoming egm-contracts bumps and the egm-data work each
requires:

### Polymorphic `stimulation` (egm-contracts v0.5.0) — Phase 2

`synthetic_bank` writer needs to encode the polymorphic `stimulation`
object (type discriminator + per-type params); reader needs to decode.
The Pydantic model from contracts handles validation; egm-data's
writer just needs to serialize the model to HDF5 attrs in a way that
round-trips through the reader.

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 2.

### `iafdb_bank` 1.2 + audit-report sidecar (egm-contracts) — Phase 1.5

When egm-contracts adds the `run_record_path` field to `iafdb_bank`
attrs (and possibly a new `iafdb_bank_run_record` schema for the
sidecar itself), egm-data ships:

- Updated `iafdb_bank` writer that accepts an optional sidecar path
  argument and stamps it into attrs.
- New reader + writer for the sidecar schema (whether it's a fresh
  schema or a reuse of `noise_bank_run_record`'s structure).

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 1.5.

### `noise_bank` 1.1 + `noise_bank_run_record` 1.1 (egm-contracts) — Phase 1.5

When egm-contracts adds the `calibration_scalar` per-trace column and
the `per_trace_provenance.lead` field, egm-data updates the noise-bank
reader + writer to round-trip the new fields.

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 1.5.

### `egm_features_bank` (egm-contracts) — Refactor Step 4

When egm-contracts adds the features-bank schema, egm-data ships
reader + writer + a `FeaturesBankDataset` for consumers that want to
load features the same way they load classifier traces.

> → Tracked at `intracardiac-platform/project/refactor_checklist.md` Phase 4.

### 3D geometry schema (egm-contracts) — Phase 7

If egm-contracts adds a `synthetic_bank` schema field referencing a 3D
mesh, egm-data needs to handle the mesh reference (probably a relative
path + content hash, not the mesh data itself — mesh files are too big
to embed in the bank). Possibly also a separate `MeshReader` if we
decide egm-data should own 3D mesh I/O.

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 7.

## Splitting strategies deep dive — open (component-internal)

The current `patient_aware_split` ships `AnyPositive` and `BinnedDensity`
stratification strategies. The strategy Protocol is extensible — see
`splits/` — but the trade-off space hasn't been fully mapped:

- Stratification target choice (any-positive vs density binning vs
  local-density-per-pair binning vs fibrosis-volume-fraction).
- Bin-count tuning for `BinnedDensity` (currently a YAML knob; whether
  there's a principled default).
- Edge-case handling for small patient counts (< 10 patients) where
  stratified splitting can fail outright — currently raises a clear
  error; a K-fold fallback might be the right answer.

> → Tracked as task #278. Component-internal; no specific phase home until a consumer asks for a new strategy.

## Won't-do (out of scope, but documented to save the question)

- **No DSP primitives in this repo.** Filtering, calibration,
  threshold extraction live in `myocard-egm-signal`. egm-data
  consumes signal-processing primitives; it doesn't implement them.
- **No feature engineering.** Per-trace feature computation lives in
  `myocard-egm-features` (Refactor Step 4, pending). egm-data
  reads/writes feature banks once the schema is defined but doesn't
  compute the features.
- **No model code.** Datasets + loaders + augmentation — yes; model
  architecture, training loops, ONNX export — no. egm-classifier is
  where the model code lives.
- **No producer-specific logic.** Bank readers + writers are
  symmetric — they don't know whether a bank came from iafdb-pipeline
  or synthetic-egm-pipeline. Producer-specific transformations
  (e.g., IAFDB's R-wave-anchored calibration) belong in the producer
  repo, not here. egm-data takes already-calibrated, already-segmented
  signal and writes / reads the bank format.

## Open architectural questions for later

- **Should `EGMTraceDataset` grow a streaming mode** for banks too
  large to fit in RAM? Today it loads the whole HDF5 signal array
  into memory at construction time. As bank sizes grow (Phase 1.5
  pushes synthetic to 300–500 sims; Phase 7 multi-beat with 3D
  geometries could be much larger), a chunked / mmap-based loader
  might be needed. Revisit when bank load time becomes painful.
- **Should `records/` grow a versioned-write helper** that
  automatically writes the schema_version attr alongside the record?
  Today the contracts Pydantic model carries the version; the writer
  serializes it transparently. A helper that emits the schema version
  to a sidecar `.version` file could simplify cross-language
  consumption (C++ deployment doesn't want to parse a full JSON to
  know if it can read the file).
- **Should `splits/` ship a K-fold cross-validation helper** for the
  small-patient-count regime? Today the egm-classifier CLI doesn't
  have a K-fold mode; if it ever grows one, the per-fold patient
  assignment logic belongs here.
