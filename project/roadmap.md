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

## Phase 1.5 — sim-realism (scheduled — see the phase plan)

**In flight.** egm-data's Phase-1.5 slice is scoped and stepped in
[`phase_1_5_plan.md`](phase_1_5_plan.md), not here: the `synthetic_bank` **2.0** I/O + θ-free
ClassifierBank conversion + the bank ⋈ bank T4 join (**DAT1**), `training_run_record` 1.2 and the
`training_metrics` CSV train columns (**DAT3**), and the backlog-driven schema halves **B11 · B14 ·
B15 · B16 · B18 · B19 · B20**. All ship in one coordinated PR at **v0.6.0**, re-pinned to
egm-contracts v0.6.0. Those items are removed from this roadmap when the plan lands in the
CHANGELOG; only the entries below stay future-only.

The **polymorphic `stimulation` encode/decode** item that sat under Phase 2 was pulled forward into
1.5 and absorbed by the `synthetic_bank` 2.0 restructure — it is now the `activation` facet of the
per-simulation config, covered by DAT1. Removed from Phase 2 accordingly.

### `noise_bank` 1.2 + `noise_bank_run_record` 1.2 — noise-side calibration fields

When egm-contracts adds the `calibration_scalar` per-trace column and the
`per_trace_provenance.lead` field, egm-data updates the noise-bank reader + writer to
round-trip them.

**Deferred out of Phase 1.5** (design §4 keeps noise-side calibration in the backlog; only the
`bank_id` attr rides along, as B20). Note the version: B20 takes `noise_bank` **1.1**, so this work
lands at **1.2** — the "1.1" this entry previously claimed is already spoken for.

> → Pairs with iafdb-pipeline's opt-in noise-side calibration. Revisit when that is scheduled.

## Phase 2 — multiclass severity

*(No egm-data items currently scheduled — the polymorphic `stimulation` work moved to Phase 1.5, see
above.)*

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

### Stamp the noise bank's `bank_id` into the `.h5`

Today a noise bank's stable id rides only on its sibling `<stem>_run_record.json`; egm-studio's
Noise view has to read the id from that sidecar because the `.h5` carries none. Once egm-contracts
adds a `bank_id` (+ a link to the run record) to the `noise_bank` HDF5 attrs, egm-data stamps it
on write and surfaces it on read, and the sibling-record lookup can retire. Cross-repo with
iafdb-pipeline (the producer) + egm-studio (drops the lookup). Surfaced by the egm-studio B10g
review.

> → **Scheduled into Phase 1.5 as B20** (`noise_bank` 1.1); stepped as S2 in
> [`phase_1_5_plan.md`](phase_1_5_plan.md). Removed from this roadmap when it lands in the CHANGELOG.

### Tolerate optional `produced_by_package` / `produced_by_version`

egm-studio indexes manually-added producer artifacts with sentinel provenance
(`produced_by_package="unknown"`, `version="0"`) because the manifest-entry fields are required.
When egm-contracts makes them optional, egm-data's manifest reader/writer should round-trip their
absence cleanly. Cross-repo.

> → **Scheduled into Phase 1.5 as B19** (`phase_manifest`); stepped as S4 in
> [`phase_1_5_plan.md`](phase_1_5_plan.md). Removed from this roadmap when it lands in the CHANGELOG.

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
