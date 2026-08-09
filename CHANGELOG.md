# Changelog

All notable changes to `myocard-egm-data` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries are per-version from `v0.4.0` on; the pre-linkage beta (`v0.1.0`–`v0.3.4`) is
summarized under [Earlier versions](#earlier-versions).

## [0.6.2] — 2026-08-09

Re-pin so iafdb-pipeline can export an uncalibrated bank. **No source change** — the enum is enforced
entirely through the re-pinned Pydantic model, and both version-stamping sites already call
`current_version()`, so new banks stamp `iafdb_bank` **1.4** on their own.

Released ahead of iafdb's first `calibration_method: "none"` bank: without this tag that bank would be
unreadable by the library that writes it.

### Changed

- Re-pin `egm-contracts v0.6.0 → v0.6.1` — `iafdb_bank` **1.4**: `calibration_method` admits `"none"`
  alongside `"r_wave_anchoring"`, and `schema_version` accepts both `"1.3"` and `"1.4"`. Existing 1.3
  banks stay readable — the enum widened, nothing was removed.

### Added

- A round-trip test for the uncalibrated path: a `none` / 1.4 bank through writer → contracts
  validator → reader → converter → ClassifierBank → write → read. It pins three things that the enum
  widening does not make obvious — that the writer stamps 1.4 unaided (via `current_version()`, whose
  result is the **last** enum entry, so the order in the schema is load-bearing); that `+inf`, the
  `none`-mode sentinel for `calibration_target_qrs_pp_mv`, survives the ClassifierBank JSON round trip
  *because* that dump does not pass `allow_nan=False`; and that a `none` value reaches
  `bank_metadata` untouched.

## [0.6.1] — 2026-08-06

Patch release for a live artifact-correctness bug, cut so iafdb-pipeline can re-pin for its Wave-1
work. No API or schema change; **v0.6.0 consumers can re-pin without touching any call site.**

### Fixed

- **RootModel reprs no longer leak into a ClassifierBank** (CL-136). `iafdb_bank_to_classifier`
  called `str()` / `list()` directly on values taken out of the contracts model, so *constrained*
  fields — which codegen wraps in a `RootModel` — came out as their repr rather than their value:
  `patient_id` read as `"root='iaf1'"` and `band_hz` as `['root=30.0', 'root=300.0']`. Both reached a
  real artifact. The values stayed distinct per patient, so patient-aware splitting was unaffected,
  but anything that displays, joins on, or parses them was polluted.

  Every value crossing out of a contracts model in that converter now goes through `_unwrap`,
  including the fields that read clean today (`source_record`, `source_channel`,
  `calibration_scalar`). Those are unwrapped only because they carry no schema constraint — adding a
  `pattern:` or `minimum:` to any of them would have silently started polluting the artifact, so the
  fix removes the latent class rather than the two known instances. The regression test asserts over
  *every* key in `bank_metadata` and `trace_metadata` for the same reason.

## [0.6.0] — 2026-07-31

Phase-1.5 Wave 1 — the I/O half of the coordinated schema bump, re-pinned to
**egm-contracts v0.6.0**. Seven schema groups land together; the breaking one is the
`synthetic_bank` restructure. Consumers (synthetic-egm-pipeline, egm-studio, iafdb-pipeline,
egm-classifier) re-pin to this release.

**Minor, not patch**, because `synthetic_bank` 2.0 breaks *egm-data's own* public API:
`read_synthetic_bank_hdf5` returns a differently-shaped model, `write_synthetic_bank` takes one,
and `synthetic_bank_to_classifier`'s label semantics change.

### Changed — breaking

- **`synthetic_bank` 2.0 I/O.** The reader and writer move to the restructured layout: bank-scoped
  root attrs with the θ-spec JSON-encoded as `generation_params_json`, a new **`simulations/`**
  group holding one typed polymorphic config object per simulation (nine `<name>_json` columns),
  and a collapsed **`traces/`** of signal + two foreign keys + int label + noise provenance, plus
  the optional `activation_position`. A pre-2.0 bank is **refused, not partially read** — its
  generation parameters live in columns 2.0 does not have, so a best-effort read would silently
  discard the provenance the bank exists to record.

- **`synthetic_bank_to_classifier` takes labels from the bank.** The 2.0 bank carries a plain int
  label per trace plus a per-simulation `{int: name}` map, so `label_fn` demotes from "the only way
  to get labels" to an optional **re-labeling override**. This inverts what omission means:
  *omitting* `label_fn` now uses the bank's labels, and asking for an unlabeled bank means passing a
  `label_fn` that returns `None`.

- **The ClassifierBank stays source-agnostic.** The five generation columns a 1.1 bank flattened
  onto every trace (`fibrosis_density`, `fibrosis_density_realized`, `electrode_row`,
  `electrode_height_mm`, `stim_edge`) are **dropped, not relocated**, along with `seed`. They are
  per-simulation facts, reachable through `simulation_id`. Only the label-policy *identity* rides
  along in `bank_metadata`, because it defines what the classification task is.

### Added

- **`banks.joins`** — `simulation_configs()` transposes the column-oriented `Simulations` model
  into one `SimulationConfig` row per simulation; `join_traces_with_simulations()` pairs each
  ClassifierTrace with the config that generated it. Read-time only, so nothing persists onto the
  ClassifierBank and its format version is unchanged. Three refusals guard against *silent wrong
  answers* rather than crashes: mismatched banks (simulation ids restart at 0 in every bank, so a
  mismatch yields a full set of confident wrong pairings), unknown `simulation_id`, and duplicate
  simulation ids.

- **`banks.id_consistency`** — the content half of the stable-id contract (B16). egm-contracts
  validates that an id's prefix is a known role; these checks validate that the artifact *is* what
  the prefix claims, across two exact axes (labels present/absent × predictions present/absent), and
  that a `noise_bank` agrees with its sibling run record about which bank it is. Both are wired into
  the writers, so a mislabelled bank cannot be written and the failed write leaves no file.

- **`iafdb_bank` 1.3** — optional `run_record_path` sidecar pointer and the per-trace
  `activation_position` `[0, 1]` column, both absent by default. Position is IAFDB provenance and
  deliberately does **not** propagate into the ClassifierBank.

- **`noise_bank` 1.1** — the stable `bank_id` is stamped into the HDF5 root attrs and surfaced on
  read, so consumers stop having to open the sibling run record just to learn which bank they hold.
  Optional-in-schema, required-on-write.

- **`training_run_record` 1.2** — `make_epoch_record` gains a keyword-optional `train_metrics`, so
  the record shows train-vs-val divergence (with validation metrics alone, an overfitting run and a
  genuinely hard task look identical). Optional by design: a producer can adopt 1.2 before it emits
  the field. A `"reliability"` key inside `train_metrics` is dropped rather than stored —
  `train_reliability` is out of scope (FB-10).

- **`training_metrics`** — the six nullable `train_*` CSV columns, with the column order **pairing**
  the splits (`train_loss, train_*, val_loss, val_*`) so divergence reads off adjacent columns.

### Fixed

- **Held-out test reliability bins are coerced like the val bins** (B18). They were passed through
  raw while val bins went through the coercion path, so a plain mapping or namedtuple that worked
  for val silently failed for test.

### Changed

- `phase_manifest` entries round-trip absent `produced_by_package` / `produced_by_version` cleanly
  (B19) — omitted from the JSON, never written as `null`, so a curator can record "producer unknown"
  instead of stamping `"unknown"` / `"0"` sentinels that read like real provenance.
- `host` is no longer a documented well-known key on the run record (B15); artifact paths in
  `config` are documented as repo-relative (B14). Both conventions; the schema stores what it is
  given.

### Dependencies

Re-pins `egm-contracts v0.5.3 → v0.6.0`.

## [0.5.0] — 2026-07-07

egm-data is now a **pure I/O** library — schemas in, banks/records/manifests on disk, typed
models out; no torch, no training loop.

### Removed

- **Training-data layer** — `datasets/` (`EGMTraceDataset`, `build_dataloaders`,
  `LoaderBundle`), `splits/` (`patient_aware_split` + strategies), `augmentation/`
  (`TraceTransform`), and their tests — relocated to their sole consumer, **egm-classifier**
  (`myocard_egm_classifier.data`), per the Refactor Step 8 code-placement audit.
- The `[torch]` optional extra and the `torch.*` mypy override; egm-data no longer imports
  torch anywhere.

### Changed

- Re-pin `egm-contracts v0.5.2 → v0.5.3` (optional `ArtifactId` date suffix; coordinated
  cascade).

### Dependencies

Consumers migrate `egm-data[torch] v0.4.x` → `egm-data v0.5.0` (no extra).

## [0.4.2] — 2026-07-01

### Added

- `phases.load_phase_dir(phase_dir)` — reads `<phase_dir>/manifest.json` through
  `load_phase_manifest`, so a consumer points at a phase folder and gets a typed
  `PhaseManifest` (the folder layout lives with the reader, not each caller). Additive.

### Changed

- Re-pin `egm-contracts v0.5.1 → v0.5.2` (the generated artifact-role vocabulary).

## [0.4.1] — 2026-06-28

### Added

- `ClassifierBank.uniform_fs_hz()` accessor for single-sample-rate banks (pushed down from
  egm-studio's view-model builder), with a test.

## [0.4.0] — 2026-06-27

Cascades the egm-contracts v0.5.0 / v0.5.1 cross-artifact linkage into the I/O layer. See
`intracardiac-platform/project/cross_artifact_linkage_design.md`.

### Added

- **`phases/` subpackage** — typed `load_*` / `write_*` for the three linkage JSON formats
  (`phase_manifest`, `observation`, `figure_spec`).
- **Producer-bank ids** surfaced on readers (`None` on legacy banks) and required by
  `write_synthetic_bank` / `write_iafdb_bank` on new writes; record `build_*` helpers accept
  the new id / pointer fields (`run_id` / `trained_on_bank_id` / `produced_model_id` on the
  run record, `model_id` on the model sidecar, `bank_id` on the noise-bank sidecar).

### Changed

- **`ClassifierBank` → 0.2** — added the bank's own optional stable `id` (validated via
  `common.ArtifactId`) and reworked the per-source / per-trace `bank_id` from an integer
  index into the source bank's stable `ArtifactId`, collapsing the two id concepts and
  simplifying `concat` (dedup by id, no remap). **Clean break — 0.1 banks are not read.**
- Promoted the shared Pydantic ⇄ JSON helpers to a package-level `_serialization` module,
  shared by `records/` and `phases/`.

### Dependencies

Re-pins `egm-contracts v0.5.1`.

## Earlier versions

Pre-linkage beta (`v0.1.0` – `v0.3.4`, 2026-06-17 → 2026-06-26), tracked here in summary —
this is where the `banks/` + `records/` I/O layer took shape:

- **v0.1.0 – v0.2.0** — first bank readers/writers, then the bank-type redesign (`BankBase`
  + per-schema subclasses + the bank `Protocol`).
- **v0.3.0 – v0.3.4** — the `banks/` + `records/` structure settled: `ClassifierBank` as the
  unified consumer-side bank shape, `IafdbBank` / `SyntheticBank` / `NoiseBank` on the
  producer side, and typed Pydantic-in / Pydantic-out record readers + writers — tracking
  egm-contracts through its v0.3.0 schema renames and the v0.4.1 `hybrid_eval_metrics`
  removal (egm-data v0.3.4).

[0.6.2]: https://github.com/myocard-labs/egm-data/releases/tag/v0.6.2
[0.6.1]: https://github.com/myocard-labs/egm-data/releases/tag/v0.6.1
[0.6.0]: https://github.com/myocard-labs/egm-data/releases/tag/v0.6.0
[0.5.0]: https://github.com/myocard-labs/egm-data/releases/tag/v0.5.0
[0.4.2]: https://github.com/myocard-labs/egm-data/releases/tag/v0.4.2
[0.4.1]: https://github.com/myocard-labs/egm-data/releases/tag/v0.4.1
[0.4.0]: https://github.com/myocard-labs/egm-data/releases/tag/v0.4.0
