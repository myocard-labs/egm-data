# Changelog

All notable changes to `myocard-egm-data` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries are per-version from `v0.4.0` on; the pre-linkage beta (`v0.1.0`–`v0.3.4`) is
summarized under [Earlier versions](#earlier-versions).

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

[0.5.0]: https://github.com/myocard-labs/egm-data/releases/tag/v0.5.0
[0.4.2]: https://github.com/myocard-labs/egm-data/releases/tag/v0.4.2
[0.4.1]: https://github.com/myocard-labs/egm-data/releases/tag/v0.4.1
[0.4.0]: https://github.com/myocard-labs/egm-data/releases/tag/v0.4.0
