# egm-data — roadmap

Future work only — shipped history lives in [`CHANGELOG.md`](../CHANGELOG.md). Internal
doc; public users read the README + `docs/usage.md`.

egm-data sits between egm-contracts (the schema layer, no I/O) and the downstream producers +
consumers (which never touch HDF5 / CSV / JSON directly). Future work here is almost entirely
**reactive**: an egm-contracts schema bump cascades a matching reader/writer update, or a
consumer asks for an I/O ergonomics helper. The canonical cascade order + the master schema-bump
list live in [`egm-contracts/project/roadmap.md`](../../egm-contracts/project/roadmap.md); the
items below are the egm-data half of each.

Work lands here as it's identified, sits in the **Backlog** until a phase-planning session
promotes it, then moves to the CHANGELOG once shipped. Items scheduled into cross-cutting Phase
work carry a `→ tracked at intracardiac-platform Phase X` annotation.

## Phase 1.5 — sim-realism

Shipped in **v0.6.0** — see the CHANGELOG. DAT1 (`synthetic_bank` 2.0 I/O, the θ-free ClassifierBank
conversion, the bank ⋈ bank join), DAT3 (`training_run_record` 1.2 + the `training_metrics` train
columns), and the backlog schema halves B11 / B14 / B15 / B16 / B18 / B19 / B20 are all done and
removed from this roadmap. The **polymorphic `stimulation` encode/decode** item that used to sit
under Phase 2 shipped with them, absorbed into the `activation` facet of the per-simulation config.

Only the deferred entry below remains from this phase.

### `noise_bank` 1.2 + `noise_bank_run_record` 1.2 — noise-side calibration fields

When egm-contracts adds the `calibration_scalar` per-trace column and the
`per_trace_provenance.lead` field, egm-data updates the noise-bank reader + writer to
round-trip them.

**Deferred out of Phase 1.5** (design §4 keeps noise-side calibration in the backlog; only the
`bank_id` attr rides along, as B20). Note the version: B20 takes `noise_bank` **1.1**, so this work
lands at **1.2** — the "1.1" this entry previously claimed is already spoken for.

> → Pairs with iafdb-pipeline's opt-in noise-side calibration. Revisit when that is scheduled.

## Phase 2 — multiclass severity

### Move `synthetic_bank` `seed` from `simulations/` to a root attr

`seed` is the master seed of a generation **run** — bank-scoped, like `generation_params` — but
`synthetic_bank` 2.0 placed it as a required per-simulation column, so today it is duplicated
identically across every row of `simulations/seed`. When egm-contracts moves it, egm-data's change is
one column shifting from the `simulations/` reader/writer (`_SIMULATION_INT_COLUMNS`) to the root
attrs — small and symmetric.

Backlogged 2026-07-31 (Daniel) rather than fixed in 1.5: nothing reads `seed`, and the producer
replicating one value across the rows round-trips correctly in the meantime. Worth doing before many
banks exist, because the duplication is *ambiguous* (a reader cannot tell "one master seed,
replicated" from "simulations that happened to share a seed"), and because per-simulation seeding
arriving later would silently reinterpret the same column.

> Cross-repo: egm-contracts owns the schema move, synthetic-egm-pipeline writes it once. Raised as
> coordination-log CL-096; best batched with the Phase-2 codegen-asymmetry fix below.

### Retire the local config-model unwrap helpers

`tests/test_synthetic_bank_2_0.py` carries `_unwrap` / `_type_name` helpers that paper over an
asymmetry in the generated `simulation_config` models: a union with a **single** variant codegens as
a `RootModel` (needs `.root`) with a `const` discriminator (a plain `str`), while a multi-variant
union codegens as a direct discriminated union (no `.root`) with an `Enum` discriminator (needs
`.value`). Both shapes **flip when a variant is added**, with no schema change or version bump to
signal it — so a purely additive extension silently breaks every consumer that hard-coded today's
shape.

Scheduled for a **Phase-2 fix in egm-contracts** (Daniel, 2026-07-31), either by making every `oneOf`
codegen identically regardless of member count or by shipping the accessor once beside the generated
models. When that lands, egm-data drops its local helpers and re-pins. Deliberately *not* fixed in
1.5: no union gains a variant this phase, since patchy / interstitial fibrosis is out of scope, so
nothing triggers it.

> Cross-repo: egm-studio (STU1 / STU4 / STU6) and the parameter estimator read these models directly
> and will each need the same treatment. Raised as coordination-log CL-093 / CL-094.

## Phase 4 — feature banks (egm-features / Refactor Step 4)

### `egm_features_bank` reader + writer

When egm-contracts defines the features-bank schema (once egm-features' feature list
stabilizes), egm-data ships the symmetric reader + writer. Any torch `Dataset` wrapper *over*
features lives in the consumer (egm-classifier), consistent with the Step 8 placement of the
training-data layer.

> → Tracked at `intracardiac-platform/project/refactor_checklist.md` Phase 4 (egm-features).

## Phase 7 — 3D geometry

### 3D mesh reference handling

If egm-contracts adds a `synthetic_bank` field referencing a 3D mesh, egm-data handles the
reference — probably a relative path + content hash, **not** the mesh data itself (mesh files
are too big to embed in a bank). Open question: whether egm-data should also own a `MeshReader`
for the `.pts/.elem/.lon` triple, or leave mesh I/O to the producer / an external mesh repo.

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 7.

## Backlog (unscheduled — promoted into a phase at a planning session)

### Drop the redundant noise-mixing provenance from the ClassifierBank

The synthetic→ClassifierBank converter copies `snr_db`, `noise_record`, `noise_channel`, and `seed`
into each trace's `trace_metadata`, but the same values already live on the `synthetic_bank`'s
`traces/` group and are reachable through the `simulation_id` / `pair_index` join. Phase 1.5 settled
that generation parameters do not belong on the ClassifierBank (it is a source-agnostic ML
compression — see `intracardiac-platform/project/investigations/synthetic_bank_source_of_truth.md`
§12) and this provenance is arguably the same category, kept only because it was already there.
Removing it would shrink the ClassifierBank and leave exactly one copy of each per-trace fact.

Not done in 1.5 by explicit decision (Daniel, 2026-07-28) — the data structures had already been
re-cut twice that day and the churn wasn't worth the gain. Do it when something else opens the
converter, and check first whether egm-studio's noise views or any SNR-stratified error analysis read
these keys off the ClassifierBank rather than joining. Natural sibling of the platform's **FB-11**
(the same parallel-bank duplication, for the signal arrays), and best done alongside the deferred
correlation-join helper below, which is the join those consumers would switch to.

> Cross-repo: touches egm-studio + egm-classifier if either reads the keys. Pairs with FB-11.

### Correlation join — predictions ⋈ ClassifierBank ⋈ `synthetic_bank` θ

egm-data owns the cross-artifact read/join that answers "which generation parameters produced the
traces this model gets wrong" — e.g. FN-rate against fibrosis-density bins. The join key already
exists on both sides (`simulation_id`, with `pair_index` for the electrode pair), so this is a
relational join returning a typed view, not a denormalization; egm-studio renders it as a diagnostics
view. Synthetic-only by nature — IAFDB traces have no `simulation_id`, so the view is simply empty
for real data.

Scoped with the view that consumes it rather than up front: Phase 1.5 only *guarantees the key* is
reachable from a prediction row. Design rationale in
`intracardiac-platform/project/investigations/synthetic_bank_source_of_truth.md` §12. Note the
`TunedParam.path` root vocabulary needs settling in egm-contracts before the θ side can be walked
generically — paths resolve against three different roots (per-sim config, per-trace `mixer.*`,
per-pair `electrodes.pairs.*`).

> → Build when the FN-vs-θ view is scoped as an egm-studio issue.

## Won't-do (out of scope, but documented to save the question)

- **No DSP primitives.** Filtering, calibration, threshold extraction live in
  `myocard-egm-signal`. egm-data consumes signal-processing outputs; it doesn't implement them.
- **No feature engineering.** Per-trace feature computation lives in `myocard-egm-features`;
  egm-data reads/writes feature banks once the schema is defined but never computes features.
- **No model code.** egm-data is pure I/O — the training-data layer (datasets + loaders +
  augmentation, moved out in v0.5.0), model architecture, training, and ONNX export all live in
  egm-classifier. Future training-data work (variable-length / multi-beat datasets, the
  `normalize` mode, splitting strategies, streaming loaders) is in
  [`egm-classifier/project/roadmap.md`](../../egm-classifier/project/roadmap.md).
- **No producer-specific logic.** Bank readers + writers are symmetric — they don't know whether
  a bank came from iafdb-pipeline or synthetic-egm-pipeline. Producer-specific transforms (e.g.
  IAFDB's R-wave-anchored calibration) belong in the producer repo.

## Open architectural questions for later

- **Should `records/` grow a versioned-write helper** that emits the `schema_version` to a
  sidecar `.version` file alongside the record? Today the contracts Pydantic model carries the
  version and the writer serializes it transparently; a `.version` sidecar could simplify
  cross-language consumption (the C++ deployment runtime doesn't want to parse a full JSON just
  to check whether it can read the file). Add if the C++ chat asks for it.
