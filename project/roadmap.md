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

### `iafdb_bank` 1.3 audit-report sidecar

When egm-contracts adds `run_record_path` to `iafdb_bank` attrs (+ possibly a new
`iafdb_bank_run_record` schema), egm-data ships an updated `iafdb_bank` writer that accepts an
optional sidecar-path argument and stamps it into attrs, plus a reader + writer for the sidecar
schema (fresh schema, or a reuse of `noise_bank_run_record`'s structure — decided in
egm-contracts).

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 1.5. Pairs with
> iafdb-pipeline's per-record audit reports.

### `noise_bank` 1.1 + `noise_bank_run_record` 1.2 fields

When egm-contracts adds the `calibration_scalar` per-trace column and the
`per_trace_provenance.lead` field, egm-data updates the noise-bank reader + writer to
round-trip them.

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 1.5. Pairs with
> iafdb-pipeline's opt-in noise-side calibration.

## Phase 2 — multiclass severity

### Polymorphic `stimulation` encode/decode

When egm-contracts replaces `stim_edge` with a polymorphic `stimulation` object (type
discriminator + per-type params), the `synthetic_bank` writer needs to serialize the model to
HDF5 attrs and the reader to decode it back. The contracts Pydantic model handles validation;
egm-data just has to make the attr encoding round-trip.

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 2 (coordinated
> egm-contracts v0.6.0 + egm-data + synthetic-egm-pipeline release).

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

### Stamp the noise bank's `bank_id` into the `.h5`

Today a noise bank's stable id rides only on its sibling `<stem>_run_record.json`; egm-studio's
Noise view has to read the id from that sidecar because the `.h5` carries none. Once egm-contracts
adds a `bank_id` (+ a link to the run record) to the `noise_bank` HDF5 attrs, egm-data stamps it
on write and surfaces it on read, and the sibling-record lookup can retire. Cross-repo with
iafdb-pipeline (the producer) + egm-studio (drops the lookup). Surfaced by the egm-studio B10g
review; batched with the refactor-cleanup contracts changes.

### Tolerate optional `produced_by_package` / `produced_by_version`

egm-studio indexes manually-added producer artifacts with sentinel provenance
(`produced_by_package="unknown"`, `version="0"`) because the manifest-entry fields are required.
When egm-contracts makes them optional, egm-data's manifest reader/writer should round-trip their
absence cleanly. Cross-repo; batched with the refactor-cleanup contracts changes.

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
