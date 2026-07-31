# egm-data — Phase 1.5 implementation plan

**Repo:** egm-data · **Phase:** 1.5
**Phase design doc:** `intracardiac-platform/phases/phase_1_5/design.md`
**Status:** in progress · **Progress:** 11/12 steps done (S1–S11 ✅) — all code complete; **97 passed,
ruff + mypy clean**. Remaining: S12 (docs / CHANGELOG / roadmap / v0.6.0 bump / pre-PR run).
**Repo estimate:** **18 points · 17.5–42 h** (cold-start ranges — see [Estimate basis](#estimate-basis))

egm-data is **step 2 of the Wave-1 re-pin cascade**: egm-contracts v0.6.0 tags → this repo ships every
matching reader/writer → synthetic-egm-pipeline (SEP12), egm-studio (STU6), iafdb-pipeline, and
egm-classifier all unblock in parallel. Nothing in this plan can start before the contracts tag, and
four repos wait on its release. Delivery shape (Daniel, 2026-07-28): **one `development` branch,
staged per-schema commits, one PR, one tag.**

---

## Scope — what this plan covers

Per the coverage note in `intracardiac-platform/project/cross_artifact_linkage_design.md`
(Proposed changes): the estimate must cover **all six** schema groups P1–P6, not just the two core §3
rows — the backlog-driven touches land in the same coordinated PR.

| Phase item | What it needs from this repo | Steps |
|---|---|---|
| DAT1 (P2) | `synthetic_bank` **2.0** reader + writer (`simulations/` group, collapsed `traces/`, θ-spec) — the **typed `SyntheticBank` is the exposed artifact** egm-studio's T4 views read θ from | S5–S6 |
| DAT1 (P2) | synthetic→`ClassifierBank` conversion — **θ-free**; carries signal + label + the `LabelPolicy` identity + the `simulation_id` / `pair_index` join key *(project-lead, 2026-07-28 — supersedes the θ-materialization scope)* | S7 |
| DAT1 (P2) — **T4 join** | **Bank ⋈ bank joined view** — each ClassifierBank trace paired with its typed `SimulationConfig` on `simulation_id`; STU1/STU4/STU5 read θ off the typed object *(CL-024 §2 — option **1b**, in 1.5)* | S8 |
| *(deferred)* | **Generic θ resolver + predictions leg** — `TunedParam.path` resolution and the FN-vs-θ join (predictions ⋈ ClassifierBank ⋈ θ). Deferred with the path grammar *(CL-024 §2)*; S8 is the subset the phase needs. → `roadmap.md` at S12 | — |
| DAT3 (P1) | `training_run_record` 1.2 I/O — `make_epoch_record` gains **optional `train_metrics=None`** *(CL-022)* | S9 |
| DAT3 (P1′) | `training_metrics` **CSV** — write + read the six `train_*` scalar columns *(Daniel, 2026-07-28 — see design note 5)* | S10 |
| B18 (P1) | Emit `HeldOutTest.metrics` / `.reliability` in parity with the val bundle | S9 |
| B15 (P1) | Stop documenting `host` as a well-known `run` key (convention; no schema field) | S9 |
| B14 (P1) | Relative artifact paths — **egm-classifier-side convention**; egm-data change is docs only | S9 |
| B16 (P3) | `ArtifactId` **content** check — role prefix ↔ bank content (`tbank_` has labels, `upred_` doesn't) | S11 |
| *(CL-024 §3)* | **Join-key writer check** — `write_classifier_bank` rejects `sim_id`-keyed traces; the only place the convention can be enforced (no JSON Schema — CL-012) | S7 |
| B19 (P4) | Round-trip absent `produced_by_package` / `produced_by_version` cleanly | S4 |
| B17 (P4) | Relative in-phase manifest paths — **egm-studio-side convention**; **no egm-data change** | — |
| B11 (P6) | `iafdb_bank` 1.3 — stamp + surface the optional `run_record_path` sidecar pointer | S3 |
| *(CL-053 / IAF3)* | `iafdb_bank` 1.3 — carry the per-trace **`activation_position`** `[0,1]` column on read + write; **must not** reach the ClassifierBank | S3 |
| *(CL-062 / SEP2)* | `synthetic_bank` 2.0 — same **`activation_position`** column on the typed `SyntheticBank` read + write; same negative test on the converter | S5 · S6 · S7 |
| B20 (P5) | `noise_bank` 1.1 — stamp + surface the `bank_id` root attr | S2 |

**Not in scope.** The `iafdb_bank_run_record` *schema* (P6 flags it deferred — B11 needs only the
pointer). Everything on the far side of a boundary: producer behavior, GUI adoption, classifier emit.

## Design notes

Local decisions taken before coding. Notes 1–3 and 5 were escalations; **all four came back resolved
in [CL-037](../../intracardiac-platform/phases/phase_1_5/coordination_log.md) on 2026-07-29** and are
kept here as the record of what was decided, not as open questions.

1. **✓ Release version — v0.6.0, not v0.5.x** *(CL-037 item 2, accepted)*. Design §7 originally targeted
   "egm-data v0.5.x", but P2 breaks *egm-data's own* public API: `read_synthetic_bank_hdf5` returns a
   differently-shaped model, `write_synthetic_bank` takes one, and `synthetic_bank_to_classifier`'s
   signature changes. Pre-1.0, a minor bump is the honest signal to the four re-pinning consumers.
   **The §7 target now reads v0.6.0**; S12 bumps to it.

2. **✓ `SyntheticLabelFn` becomes an override, not the source of truth** *(CL-037 item 3 → routed as
   **CL-038** to egm-classifier + egm-studio)*. Today the synthetic converter derives labels
   consumer-side from `traces.fibrosis_density` via a `label_fn`. Under P2 the bank *carries* the label
   — a plain int per trace plus a per-sim `LabelPolicy` and a `{int: name}` map — so the converter
   defaults to the bank's own label + names and `label_fn` survives only as an explicit re-labeling
   override. **Non-breaking**: existing calls passing `label_fn` still work, but consumers should omit
   it on the v0.6.0 re-pin and take the bank's label.

3. **✓ P3's egm-data half was already shipped** *(CL-037 item 4 — **P3 amended**)*. P3 listed "add
   per-trace `split` + `prediction` columns" to `ClassifierBank` as outstanding. Both already exist on
   `ClassifierTrace` and already round-trip through `classifier_bank_io` (sentinel-mask pattern: `split`
   as `''`-for-`None`, `has_prediction` / `pred_label_pred` / `pred_label_prob` / `pred_logits_json`).
   **P3's egm-data half is therefore just the B16 content check** — scored and stepped accordingly.

4. **`simulations/` FK-join encoding is ours to validate.** `synthetic_bank_schema_design.md` §8
   explicitly leaves egm-data to "confirm the `simulations/` group + FK-by-`simulation_id` join
   performs for egm-studio's table view, or pick a better encoding." Note this got *lighter* under the
   2026-07-28 correction: the converter no longer joins per-sim config onto every trace, only
   `label_names` + the `LabelPolicy` identity, so the hot path is now egm-studio reading the typed
   `SyntheticBank` directly rather than anything in the conversion. S7 still records the timing on a
   realistic bank; if the join is slow the fix is a vectorized lookup on our side, **not** a schema
   change (that would reopen Wave 1).

5. **✓ `metrics.csv` gains train columns too — and it is now scoped into v0.6.0** *(Daniel 2026-07-28;
   CL-037 item 1, accepted 2026-07-29)*. P1 originally specified `train_metrics` only on `EpochRecord`
   (the JSON run record) and was silent on the separate `training_metrics` CSV, whose `_epoch_to_row`
   pulls well-known scalars from `val_metrics` alone. Because that schema is
   `additionalProperties: false`, an un-bumped validator would have **rejected** the new CSV — so this
   was a genuine gap, not a nicety. **Resolution:** `training_metrics.schema.json` gains six nullable
   properties — `train_auroc` / `train_accuracy` / `train_precision` / `train_recall` / `train_f1` /
   `train_ece` — folded into **P1 + CON3** in the egm-contracts **v0.6.0** Wave-1 bump, so it cannot
   miss the release. Both sub-decisions went the way I recommended:
   - **Paired `x-csv-column-order`** — `epoch, lr, train_loss, train_auroc…train_ece, val_loss,
     val_auroc…val_ece, epoch_seconds`. T5 exists to make train-vs-val divergence visible and the
     paired layout reads far better in a spreadsheet; consumers read by header name, so reordering is
     safe.
   - **No `train_reliability` in the CSV** — consistent with P1's FB-10 deferral; the CSV carries only
     scalars and ECE is already the scalar summary of the bins. The train path mirrors the val path:
     six scalars in, nested `confusion` dropped.

6. **B14 / B15 / B17 are conventions, not egm-data code.** B15 (drop `host`) and B14 (relative paths)
   are producer-side writes; egm-data's part is removing `host` from the documented well-known keys in
   `build_training_run_record`'s docstring and `docs/usage.md`. B17 is egm-studio's curation
   convention — `path` stays `type: string` and the manifest I/O is a Pydantic pass-through, so there
   is nothing here to change.

7. **No back-compat path for `synthetic_bank` 1.1.** Per the investigation's decision 2 (new bank type
   only, nothing load-bearing yet), `read_synthetic_bank_hdf5` simply stops accepting 1.1 — the
   existing `_check_version` / `supported_versions` guard already produces the right error. No legacy
   reader, no migration shim.

8. **`ClassifierBank` stays at 0.2 — no second breaking change in Wave 1.** *(Revised 2026-07-28 by the
   §12 reframe.)* The earlier θ-materialization scope would have put a typed `generation_params` field
   on `ClassifierBankMetaData`, adding a dataset to the `banks/` group and forcing
   `CLASSIFIER_BANK_VERSION` 0.2 → 0.3 — a second breaking change rippling to egm-classifier and
   egm-studio via the predictions bank. **That is now retired.** θ never lands on the ClassifierBank, so
   the dataclass and the HDF5 layout are untouched and Wave 1 keeps exactly one breaking change (the
   `synthetic_bank` restructure). The `LabelPolicy` **identity** rides in the existing generic
   `bank_metadata` dict as a string — deliberately not a typed field, precisely to avoid reintroducing
   the bump for something that is an identifier rather than a structure any consumer walks.

9. **θ-path resolution roots — deferral confirmed** *(CL-024 §2: "no `TunedParam.path` grammar into
   v0.6.0")*. The generic resolver will eventually walk each `TunedParam.path`, and the example paths in
   `synthetic_bank_schema_design.md` §4.1 do **not** all share a root: `substrate.density`,
   `cell_model.courtemanche.params.g_CaL_scale`, and `backend.diffusion` resolve against the
   **per-sim** objects; `mixer.snr_db` does not — there is no `mixer` object in the per-sim config;
   and an electrode knob would want `electrodes.pairs[…].height_mm`, resolving **per-pair** via the
   `pair_index` FK. Three roots, disambiguated by the first path segment. Nothing in 1.5 depends on
   this any more, so it is **not blocking** — but the legal root vocabulary belongs in the
   `TunedParam.path` field description while the schema is being written, not rediscovered later by
   whoever builds the join. **→ egm-contracts, low priority, this phase if convenient.**

10. **`noise_bank` version-number collision in our own roadmap.** P5 takes `noise_bank` 1.0 → **1.1**
   for `bank_id`, but this repo's `roadmap.md` already reserves "1.1" for the *deferred* noise-side
   calibration work (`calibration_scalar` + `per_trace_provenance.lead`), which design §4 explicitly
   keeps out of this phase. S10 renumbers that roadmap entry to **1.2** so the two don't collide.

**Cross-repo alignment (read from `egm-contracts/project/phase_1_5_plan.md`, 2026-07-28).** The
contracts chat flowed down in parallel today and reached the same scope conclusion (all six groups in
one PR + tag) and the same anchor sizing (CON1 = L, CON3 = S). Two of its decisions shape S5–S6 here:
the seven per-function polymorphic objects + `TunedParam` + `generation_params` land as `$defs` in a
**single new `simulation_config.schema.json`**, codegen'd to **one importable module** — so this repo
imports one module, not eight; and `x-hdf5-mapping` grows a `simulations_group`, with the HDF5
validator decoding each `*_json` column **per row**, which is exactly what S6's round-trip test must
satisfy. Their plan still lists the `ClassifierBank` `split` / `prediction` columns as outstanding
egm-data work — design note 3 above says otherwise; the project-lead should reconcile.

## Steps

Each step is one focused commit that ends green (its own tests + `ruff format` + `ruff check` +
`mypy`). Order is deliberate: **S2–S4 are the cheap additive trio**, run first so they prove the
v0.6.0 codegen and the re-pin are sound before S5–S8 sink real time into the breaking restructure —
Daniel's migration-wave de-risking logic applied one level down. Status: ☐ todo · 🔨 wip · ✅ done.

### S1 — Re-pin egm-contracts v0.6.0 ✅ (actual: ~15 min)
- **Change:** `pyproject.toml` dependency pin `v0.5.3 → v0.6.0`. **No behavior change**, no source edits.
- **Verify:** ~~full `pytest` suite green unchanged~~ → measured after the re-pin: **30 passed /
  10 errored**, ruff `check` + `format --check` **clean**, mypy **33 errors in 3 files**.
  **⚠ Correction (found at S3):** I first reported the blast radius as "exactly the synthetic surface".
  That is right for the **tests** — all 10 errors are the 1.1-shaped fixture — but **wrong for mypy**.
  The 33 split **27 synthetic** (`converters.py` 17 + `writers.py` 10) **+ 6 in
  `records/training_metrics.py`**, which the re-pin caused directly: the regenerated
  `TrainingMetricsRow` gained the six `train_*` columns and mypy reads them as required constructor
  arguments. **No runtime break** — they default to `None`, the writer still works, and the CSV already
  emits the new paired column order with empty train cells. Those 6 clear at **S10**.
- **Depends on:** egm-contracts v0.6.0 merged + tagged — **done, `v0.6.0` @ `9595f09`, 2026-07-30**.

> **Plan assumption that didn't survive contact — a declared red window (S1 → S7).** This step was
> written as "suite green unchanged", which is **not achievable for a breaking schema change**: the
> conftest fixture builds a `synthetic_bank` 1.1 model that 2.0 refuses, so the 10 synthetic tests and
> the 33 mypy errors cannot go green until the reader, writer and converter are rewritten (S5–S7). Two
> ways to handle it — `xfail`-mark them so every commit is green, or declare the window. **Declared**,
> because those tests are rewritten wholesale in S5–S7 (the fixture itself must become 2.0-shaped), so
> `xfail` markers would be added and deleted three steps later purely for appearances. The window is
> **contained inside the single PR** — nothing red reaches `release`, and `development` is not a
> bisect surface mid-wave. **Closes at S7**; S12's pre-PR run is the real gate.

### S2 — P5 `noise_bank` 1.1 `bank_id` (B20) ✅
- **Change:** `banks/noise_bank.py` — `write_noise_bank` stamps `f.attrs["bank_id"]` (optional-in-schema,
  **required-on-write**, matching the existing `write_synthetic_bank` / `write_iafdb_bank` guard);
  `read_noise_bank_hdf5` surfaces it via a new `_opt_str_attr` helper (`None` on legacy banks, never
  `""`). Fixture + module docstrings updated 1.0 → 1.1.
- **Verify:** ✅ 7 noise tests pass (5 existing + 2 new: legacy-bank-reads-as-`None`,
  write-without-id-raises). Full suite **32 passed / 10 errored** — up 2 passes, the 10 synthetic errors
  unchanged. ruff clean; mypy still exactly 33 errors with **0 in `noise_bank.py`**, so the step added
  no new type debt. Contracts validator accepts the stamped bank.
- **Depends on:** S1.

> **Found while implementing — a contract obligation my plan missed.** The shipped `noise_bank` 1.1
> `bank_id` description assigns egm-data a **cross-file check**: "The sibling `noise_bank_run_record`
> carries the same id …; when both are present they **MUST agree** — egm-data checks that, since JSON
> Schema cannot compare across two files." That is a third piece beyond stamp-and-surface, and it isn't
> in any of my steps. **Routed to S11**, not here: S11 is already the id-consistency step (B16's role
> prefix ↔ bank content), so all egm-data-enforced id invariants land in one module rather than being
> scattered across the bank writers. Raised to the project-lead in **CL-091** — it's a contracts-authored
> expectation, so they should know it's tracked and where. **Confirmed by Daniel (2026-07-31):** the
> underlying `bank_id`-duplicated-across-bank-and-record issue is already **known and backlog-tracked**
> from the egm-contracts work, and S11 is the right home for the validation half. So this is enforcement
> of a known duplication, not a new discovery — the backlog item may eventually remove the duplication
> and with it the need for the check.

### S3 — P6 `iafdb_bank` 1.3 — `run_record_path` (B11) + `activation_position` (CL-053) ✅
- **Change:** two additive pieces on the same 1.3 bump. **(a)** `banks/writers.py` `write_iafdb_bank`
  stamps the optional relative sidecar pointer; `banks/readers.py` `read_iafdb_bank_hdf5` reads it
  through `_opt_str_attr`. **(b)** *(CL-053)* both carry the new per-trace `traces/activation_position`
  column — a `[0,1]` float, optional-in-schema / required-on-write in activation mode. Field lands here
  in Wave 1 **unpopulated** (IAF3); iafdb-pipeline fills it in Wave 2 (IAF1).
- **Verify:** round-trip with and without the pointer; round-trip of `activation_position` including the
  absent case; the contracts file-level validator accepts every combination. **Plus a negative test:**
  `iafdb_bank_to_classifier` does **not** propagate `activation_position` into `trace_metadata` — it is
  IAFDB provenance for STU5, and the ClassifierBank stays source-agnostic (same rule that keeps θ off
  it). Worth pinning by test precisely because the converter is where it would leak.
- **Depends on:** S1.
- **Result:** ✅ suite **35 passed / 10 errored** (+3 passes, the ten synthetic errors unchanged); ruff
  clean; mypy unchanged at 33 with **0 in `readers.py`**. Four new tests: both-optionals round-trip,
  both-absent-read-as-`None`, and the converter negative test. The base fixture deliberately keeps both
  fields **absent** — that is the real shape of a Wave-1 sliding-window bank, so every existing test
  exercises the absent path for free; a second fixture carries them populated, with positions at the
  `[0, 1]` **endpoints** (0.0 is what a zero-filling reader would produce, 1.0 catches an exclusive
  upper bound).

> **Two obligations swept out of the v0.6.0 schema prose** (the check I promised in CL-091). Grepping
> the shipped schemas for "egm-data …" turns up exactly two *enforcement* clauses, both now placed:
> the `noise_bank` ↔ run-record id agreement (→ S11), and **`synthetic_bank`'s cross-group FK** —
> "egm-data checks it since JSON Schema cannot express a cross-group reference", i.e.
> `traces/simulation_id` must resolve into `simulations/`. CL-089 flagged the latter too ("a writer
> should refuse to emit an orphan rather than rely on validation after the fact") — **added to S6**.
> The remaining hits are all the "stamps it on every new bank" required-on-write guards, which are
> already implemented for all three banks.

### S4 — P4 `phase_manifest` optional `produced_by_*` (B19) ✅
- **Change:** **no production code**, as predicted — `phases/phase_manifest.py` is a Pydantic
  pass-through, the regenerated entry models carry `produced_by_* = None` defaults, and
  `_write_pydantic_json`'s `exclude_defaults=True` already omits them. The deliverable is the three
  tests that pin the behavior.
- **Verify:** ✅ 6 tests in `test_load_phase_dir.py` (3 existing + 3 new); suite **38 passed /
  10 errored** (+3); ruff clean; mypy `src` unchanged at 33 and the touched test file clean.
- **Depends on:** S1.
- **Why tests and not just a changelog line.** The omit-don't-null behavior is a property of *our
  serializer*, not of the schema: these fields are typed as plain strings, so an explicit `null` would
  fail validation on the next read. Nothing in the schema stops a future refactor from switching
  `exclude_defaults` to `exclude_none` or dropping the flag, and the failure would surface as an
  unreadable manifest in egm-studio rather than here. The three tests cover absent-round-trips,
  omitted-not-null (asserting on the raw JSON), and **partial** provenance — package known, version not
  — since the two fields are independently optional and a curator that knows only half shouldn't be
  pushed back into a sentinel.

### S5 — DAT1a · `synthetic_bank` 2.0 **reader** ✅
- **Change:** `banks/readers.py` — replace the flat-root-attr read with: root attrs
  (`schema_version` / `created_utc` / `bank_id` / `description` / `fs_hz` / `trace_duration_ms` /
  `noise_bank_source` / `generation_params_json` = the θ-spec) + a new `simulations/` group decoding
  nine `*_json` columns (`geometry` / `cell_model` / `substrate` / `activation` / `electrodes` /
  `backend` / `label_policy` / `label_names` / `substrate_summary`) into the typed polymorphic
  contracts models + the collapsed `traces/` (`signal`, `simulation_id`, `pair_index`, `label` int,
  `snr_db`, `noise_record`, `noise_channel`, and — *CL-062* — the optional `activation_position`
  `[0,1]` float, absent in Wave 1 and populated by SEP2 in Wave 2). Drop the removed columns and the old
  `fibrosis_params_json` / `electrode_config_json` / `mixer_config_json` / `experiment_config_json`
  root attrs.
- **Verify:** ✅ 10 tests in the new `tests/test_synthetic_bank_2_0.py`; suite **48 passed / 10
  errored** (+10, the remaining errors are the 1.1-shaped fixture that S6/S7 replace); ruff clean; mypy
  unchanged at 33 with **0 in `readers.py`**.
- **Depends on:** S1.
- **The fixture is hand-built, deliberately.** `conftest.write_synthetic_bank_2_0_by_hand` lays the
  HDF5 out from the schema's `x-hdf5-mapping` with raw h5py rather than going through our writer.
  Reader and writer are two halves of one restructure: testing the reader against writer output would
  let a *shared* misreading of the schema round-trip perfectly and still be wrong. The first test
  asserts the hand-built file passes the contracts validator, so the reference itself is guarded. The
  two simulations use **different variants** on the `cell_model` and `activation` unions, which fails
  if the reader ever resolves a union once per column instead of per row.

> **Finding — the generated config models are asymmetric, and the asymmetry flips with variant count.**
> A union with **one** variant today (`Geometry`, `Substrate`, `Electrodes`, `Backend`) codegens as a
> `RootModel`, so reaching the object needs `.root`; a union with **several** (`CellModel`,
> `Activation`, `LabelPolicy`) codegens as a direct discriminated union with no wrapper. The
> discriminator has the same split: a sole variant gets a `const` → plain `str`, a multi-variant union
> gets an `Enum` → needs `.value`. So `geometry.root.size_mm` but `cell_model.ap_time_unit_ms`;
> `cell_model.type.value` but `geometry.type`. **The trap:** both shapes *change when a variant is
> added* — the day Phase 7 adds a second `Geometry`, every `geometry.root…` in every consumer breaks,
> and nothing in the schema changed. My tests unwrap defensively via `_unwrap` / `_type_name` helpers
> so they survive that. **egm-studio (STU6/STU1/STU4) and the estimator will hit this**, so it's raised
> in **CL-093** rather than solved locally — a shared accessor in contracts would be the real fix.
>
> **✓ Settled (Daniel, 2026-07-31):** patchy / interstitial fibrosis is **not** implemented in Phase 1.5,
> so `Substrate` stays single-variant and **no trigger fires this phase** — the in-phase urgency
> CL-094 raised is withdrawn. The asymmetry goes to the **backlog, fixed in Phase 2** (project-lead to
> assign the FB id; the fix is contracts-side). egm-data needs nothing further now: the local
> `_unwrap` / `_type_name` helpers are the interim insulation and are written to survive the flip
> either way. Retiring them once contracts ships the fix is logged in `roadmap.md`.

### S6 — DAT1b · `synthetic_bank` 2.0 **writer** + round-trip ✅
- **Change:** `banks/writers.py` — mirror S5. `simulations/` datasets sized `(M,)`; `traces/` sized
  `(N,)`; `label` as `int64`; θ-spec serialized to `generation_params_json` with `sort_keys=True` for
  deterministic bytes; `activation_position` written when present and omitted when not *(CL-062 — the
  writer half, which CL-062's scope note doesn't name but SEP12 needs, since the producer writes
  through this function)*. **Plus the cross-group FK guard** *(swept out of the schema prose at S3;
  also CL-089)*: refuse to write a bank whose `traces/simulation_id` doesn't resolve into
  `simulations/` — an orphan trace must fail at write time, not be left for the validator afterwards.
- **Verify:** ✅ 15 tests in `test_synthetic_bank_2_0.py` (10 reader + 5 writer). Suite **57 passed /
  6 failed** — up from 48/10, and the 10 *collection errors* are gone entirely now the shared fixture
  builds a 2.0 model. ruff clean. **mypy 33 → 23**, with `writers.py` now **clean**; the remaining 23
  are `converters.py` (17 → S7) and `training_metrics.py` (6 → S10).
- **Depends on:** S5.
- **A structural check, not just a round-trip.** `test_writer_matches_the_hand_built_layout` compares
  group and dataset *names* between our writer's output and the hand-built reference. A round-trip
  alone cannot catch a column written under the wrong name, a missing `_json` suffix, or a stray 1.1
  leftover — reader and writer would agree with each other and both be wrong. This is the payoff for
  having kept the hand-built fixture independent at S5.
- **The old fixture became the 2.0 fixture.** `synthetic_bank_path` now builds through
  `build_synthetic_bank_2_0_model`, which shares `SIM_2_0` / `GENERATION_PARAMS_2_0` with the
  hand-built file — so writer output and the independent reference describe *the same bank*, and any
  divergence surfaces as a failure rather than as two tests quietly asserting different things.
- **FK guard shipped** (`_check_simulation_fk`): a trace pointing at an unrecorded `simulation_id` is
  refused at the write call, with the offending ids and the known set in the message. Wired ahead of
  the file being opened, so a rejected bank leaves no partial file behind.

### S7 — DAT1c · θ-free `synthetic_bank_to_classifier` ✅
- **Change:** `banks/converters.py` — take `label_truth` from the bank's int `label` and
  `ClassifierBank.labels` from `label_names` (per design note 2, `label_fn` demotes to an override);
  record the `LabelPolicy` **identity** in `bank_metadata` (design note 8 — a string, not a typed
  field). Per-trace, keep `patient_id` (`str(simulation_id)`, which egm-classifier's patient-aware split
  depends on), `simulation_id` + `pair_index` as the **join key**, and — **unchanged from today** — the
  noise-mixing provenance (`snr_db`, `noise_record`, `noise_channel`, `seed`). The five
  generation-config fields the old converter flattened (`fibrosis_density`,
  `fibrosis_density_realized`, `electrode_row`, `electrode_height_mm`, `stim_edge`) are **dropped, not
  relocated** — they live on the `synthetic_bank`, which S5 already exposes typed. Update
  `load_synthetic_bank_as_classifier`.
- **Deliberately not touched:** the noise-mixing provenance is redundant with the `synthetic_bank`
  (reachable via the same `simulation_id` join) and is a candidate for removal later, but 1.5 does not
  re-litigate it *(Daniel, 2026-07-28)*. Logged in `roadmap.md` so it outlives this plan.
- **Verify:** conversion test asserting no θ / generation-config key reaches `trace_metadata` —
  **including `activation_position`** *(CL-062, the synthetic twin of S3's iafdb negative test)*; labels
  come from the bank with no `label_fn` supplied and an explicit `label_fn` still overrides;
  `patient_id` unchanged. Plus the **key guarantee** the §12 reframe asks for — assert `simulation_id`
  is reachable from a *prediction* row end-to-end (bank → predictions → back), since that is what makes
  the deferred correlation join possible. Plus the conversion-cost check from design note 4 (~2000
  traces), recorded here.
- **Also here — the join-key writer check** *(CL-024 §3)*: `write_classifier_bank` rejects a bank whose
  traces key the join as `sim_id` rather than `simulation_id`. The producer renames at SEP12; contracts
  cannot pin this (the ClassifierBank is numpy-backed with no JSON Schema — CL-012), so egm-data's
  writer is the only place the convention can be enforced. Test: a `sim_id`-keyed bank raises, with both
  names in the message.
- **Depends on:** S6.
- **Note:** this step got *smaller* under the 2026-07-28 correction — dropping fields is less work than
  joining them. The estimate holds because the `LabelPolicy` id, the key-guarantee test, and the writer
  check replace it.
- **Result:** ✅ **67 passed, 0 failed** — the suite is fully green for the first time since S1. ruff
  clean. **mypy 23 → 6**, `converters.py` now clean; the 6 left are `training_metrics.py` (S10).
  **The declared red window is closed**, one step later than S1 predicted and on the step it named.
- **Conversion cost** (design note 4, closed): on a realistic bank — **2000 traces / 25 simulations at
  T = 192 ms** — write **0.12 s**, read + convert **0.09 s**. No vectorized-join work needed; the naive
  per-trace path is nowhere near mattering, and egm-studio's table view reads the typed `SyntheticBank`
  directly anyway.
- **`label_fn` semantics inverted, deliberately.** Under 1.1, *omitting* `label_fn` produced an
  unlabeled bank. Under 2.0 omitting it takes the bank's own labels — the normal path — and asking for
  an unlabeled bank means passing a `label_fn` that returns `None`. Both spellings are tested so the
  inversion is pinned rather than implied. **egm-classifier / egm-studio see this** (already routed as
  CL-038).
- **Deviation from the plan text:** `seed` is *not* kept in `trace_metadata`. The plan listed it with
  the noise-mixing provenance, but 2.0 moved `seed` into `simulations/`, so keeping it per-trace would
  have been exactly the leak this step exists to prevent. `snr_db` / `noise_record` / `noise_channel`
  stay, per Daniel's 2026-07-29 call. **Follow-up (Daniel, 2026-07-31):** dropping it was right, but my
  reason was wrong — `seed` is the master seed of a generation *run*, i.e. bank-scoped, not
  per-simulation. 2.0 put it in the wrong group. Interim: the producer replicates the same master seed
  into every `simulations/seed` row; moving it to a root attr is backlogged via **CL-096**. No egm-data
  change this phase — we round-trip the column without interpreting it.

### S8 — DAT1d · bank ⋈ bank joined view (T4) ✅
- **Change:** new read-time joined view in `banks/` — given a synthetic-sourced ClassifierBank plus its
  `SyntheticBank`, return each trace paired with its typed `SimulationConfig`, joined on
  `simulation_id`. **Typed object out, no θ flattening** — STU1/STU4/STU5 read θ off the config the same
  way they read any other field. Per CL-012 θ is stored **per-simulation**, so this join needs only
  `simulation_id`; `pair_index` is required solely for per-trace correspondence *between* the two banks
  and is out of scope here. Read-time only — **nothing persists onto the ClassifierBank**, so
  `CLASSIFIER_BANK_VERSION` stays 0.2 and Wave 1 keeps exactly one breaking change.
- **Verify:** join test over a multi-sim bank asserting every trace resolves to the right
  `SimulationConfig`; a trace whose `simulation_id` is absent from `simulations/` raises rather than
  silently yielding `None`; an IAFDB-sourced ClassifierBank is rejected (the join is synthetic-only by
  nature). Timing on a realistic bank (~2000 traces) recorded here, closing out design note 4.
- **Depends on:** S5 (the typed `SyntheticBank`) + S7 (the ClassifierBank carrying the key).
- **Not in scope:** generic `TunedParam.path` resolution and the predictions leg — both deferred with
  the path grammar per CL-024 §2; logged to `roadmap.md` at S12.
- **Result:** ✅ new `banks/joins.py`; suite **75 passed / 0 failed** (+8); ruff clean; mypy still the
  6 `training_metrics.py` errors only. Join cost at 2000 traces / 25 simulations: **0.0018 s** — a dict
  lookup per trace, nowhere near mattering.
- **API shape.** `simulation_configs()` transposes the column-oriented contracts `Simulations` model
  into one `SimulationConfig` **row** per simulation; `join_traces_with_simulations()` pairs each trace
  with its own. The per-function objects pass through **as the contracts models they already are** —
  a reshape, not a reinterpretation — so consumers read θ off them exactly as they'd read any other
  field, and the Phase-2 codegen fix (CL-095) lands on them unchanged.
- **Three guards, all against *silent wrong answers* rather than crashes.** Worth naming because none
  of them protects against an exception — each protects against a confident, plausible, wrong result:
  1. **Mismatched banks.** `simulation_id` restarts at 0 in every bank, so joining a ClassifierBank
     against a *different* synthetic bank yields a complete set of wrong pairings and no error at all.
     Stable bank ids exist precisely so this is checkable, so it's a hard stop.
  2. **Unknown `simulation_id`.** Raising beats returning `None`, which would push the decision onto
     every consumer — and the natural consumer response, skipping the row, quietly drops traces from a
     comparison.
  3. **Duplicate simulation ids.** Ambiguous join; keeping the last would attach half the traces to
     the wrong config.
- **A test I rewrote rather than kept.** The IAFDB-refusal test initially used `object.__setattr__` to
  force execution past the bank-correspondence guard and reach the missing-key message. That's the test
  fighting the code: for a real IAFDB bank, "this synthetic bank isn't among your source banks" *is* the
  correct and more useful error. Now it asserts the refusal plainly, and the missing-key path has its
  own honest test.

### S9 — DAT3a · `training_run_record` 1.2 JSON (P1 · B18 · B15 · B14) ✅
- **Change:** `records/training_run_record.py` — `make_epoch_record` gains a **`train_metrics=None`
  keyword-optional** parameter *(CL-022)*, given the same treatment as `val_metrics` (flat dict in,
  non-finite floats sanitized). **Optional by design:** egm-classifier's CLF5 Wave-1 migration writes
  1.2 records *without* train metrics, which is what keeps the migration wave separate from the feature
  wave. Per P1, `train_reliability` bins are out of scope (→ FB-10), so a `"reliability"` key inside
  `train_metrics` is **dropped, not split into a second field** — deliberately asymmetric with the val
  path. Also: `build_training_run_record` emits `HeldOutTest.metrics` / `.reliability` in parity with
  the val bundle (B18); drop `host` from the documented well-known `run` keys (B15) and note the
  repo-relative path convention (B14).
- **Verify:** round-trip of a record carrying train + val + test bundles through the contracts
  validator; a record built with `train_metrics` omitted still validates (the CLF5 migration case); a
  `"reliability"` key inside `train_metrics` is dropped and produces no second field; `best_epoch` still
  selects on `val_metrics` only (train metrics must not leak into selection).
- **Depends on:** S1. Parallel with S5–S8 (different module).
- **Result:** ✅ 5 new tests; suite **80 passed / 0 failed** (+5); ruff clean. mypy unchanged at the 6
  `training_metrics.py` errors — S10 clears those.
- **B18 turned out to be a real bug, not a description tightening.** The linkage doc frames B18 as
  aligning `HeldOutTest.metrics` to the val bundle, which reads like a validator/wording change. In our
  code the asymmetry was in *behavior*: `build_training_run_record` ran val bins through
  `_coerce_reliability_bin` but passed **test** bins through as `list(...)` raw — so a plain mapping or
  a namedtuple that worked for val silently failed for test. Test bins now take the same path, which is
  what "parity" should have meant.
- **The dropped-`reliability` asymmetry is deliberate and pinned by test.** A `"reliability"` key
  inside `train_metrics` is discarded rather than split into a second field, since `train_reliability`
  is out of scope for 1.2 (FB-10) and there is nowhere to put it. Dropping beats raising: a producer
  computing one metrics dict per split will naturally pass reliability on both, and failing would force
  it to special-case a field it cannot store. Tested so the next reader doesn't "fix" it.
- **B15 / B14 are docstring-only here, as scoped.** `host` left the documented well-known `run` keys
  and the `config` docstring now states the repo-relative-path convention. The `run` object still
  allows extra keys, so a producer wanting `host` isn't blocked — it just isn't in the documented set.

### S10 — DAT3b · `training_metrics` CSV train columns ✅
- **Change:** `records/training_metrics.py` — `_epoch_to_row` pulls the six well-known scalars from
  `EpochRecord.train_metrics` as well as `val_metrics` (dropping nested `confusion`, exactly as the val
  path does); `write_training_metrics` follows the schema's updated `x-csv-column-order`;
  `_coerce_row` / `load_training_metrics` parse the new columns back, with the same empty-cell → `None`
  handling the nullable `val_*` columns already use.
- **Verify:** epochs → CSV → rows round-trip with train and val populated; a non-finite train metric
  writes an empty cell and reads back as `None`; the contracts row validator accepts the output.
- **Depends on:** S9, and on egm-contracts' `training_metrics` column addition — **now scoped into P1 +
  CON3 for v0.6.0** (design note 5, CL-037 item 1), so this arrives with the same tag as everything
  else and is no longer conditional.
- **Result:** ✅ 3 new tests; suite **83 passed / 0 failed**; ruff clean; **mypy fully clean — "no
  issues found in 19 source files"**. That closes the last of the 33 errors the v0.6.0 re-pin
  introduced at S1, and the repo is green on all three checks for the first time in the wave.
- **Smaller than estimated, because the design was already schema-driven.** Serialization and read
  coercion both key off `csv_column_order("training_metrics")`, so the six new columns needed **no
  change to the writer or the reader** — only the `EpochRecord` → row projection had to learn about
  them. This is the payoff for having pulled column order from the contracts package instead of
  hardcoding it, and worth noting in the effort roll-up as a case where prior structure ate the cost.
- **Both splits go through one projection** (`_split_metrics(metrics, prefix)`) rather than two
  hand-written blocks. Carrying train beside val exists to make divergence readable; projecting them
  through different code is how the two quietly stop being comparable.
- **A record with no `train_metrics` writes empty cells, not absent columns.** That is the
  CLF5-migration shape — 1.2 adopted a wave before the emit — and the header has to stay stable across
  that gap or a consumer plotting the file sees the schema change mid-phase.
- **Column order verified against the raw header**, not the typed model, since the model cannot express
  order. Confirmed the shipped order pairs the splits as CL-037 requested:
  `epoch, lr, train_loss, train_auroc…train_ece, val_loss, val_auroc…val_ece, epoch_seconds`.

### S11 — B16 · `ArtifactId` role ↔ content consistency check ✅
- **Also here — the `noise_bank` ↔ `noise_bank_run_record` id-agreement check** (found at S2; the
  contracts 1.1 `bank_id` description explicitly assigns it to egm-data). When both files are present
  their `bank_id`s must match; JSON Schema can't compare across files, so this is ours. Lands here with
  the other id invariants rather than in the noise-bank writer.
- **Change:** new content check in `banks/` — given a bank and its stable id, assert the role prefix
  matches what the bank actually holds (a `tbank_` has `label_truth`; a `upred_` has none; a `lpred_`
  has both labels and predictions; an `nbank_` is a noise bank). Wired into the writers so a
  mislabelled bank can't be written. The **prefix-is-a-known-role** validation itself is
  egm-contracts' half.
- **Verify:** ✅ 14 tests in the new `test_id_consistency.py`; suite **97 passed / 0 failed**; ruff
  clean; mypy clean.
- **Depends on:** S1. Parallel with S5–S10.
- **The check found a real mislabelling on its first run — in our own test data.**
  `test_classifier_bank_id_round_trips` built a bank carrying `label_truth` and no prediction and gave
  it a **`upred_`** id, which claims exactly the opposite. Corrected to `tbank_`. That is the case for
  B16 in miniature: the id validated fine against the contracts pattern, the bank round-tripped
  correctly, every test passed — the id was simply lying, and nothing in the stack could tell.
- **Two independent, *exact* claims per role** — the four roles are the product of a labels axis and a
  predictions axis, and both are checked in both directions:

  | role | labels | predictions |
  |---|---|---|
  | `tbank_` training | present | **absent** |
  | `lpred_` labeled prediction | present | present |
  | `ptbank_` pretraining | absent | **absent** |
  | `upred_` unlabeled prediction | absent | present |

  **Corrected after review (Daniel, 2026-07-31).** My first version let `tbank_` / `ptbank_` carry
  predictions on the reasoning that those roles "say nothing about" them. That was my own invention and
  contradicts the linkage doc's role table: a labeled bank that has been evaluated **is** an
  `lpred_` ("Predictions + truth labels — full metric suite available"), and the unlabeled equivalent
  is `upred_`. Now enforced, with the error naming the role the artifact should have used. In practice
  it barely arises today because eval writes a **new** bank rather than mutating the source — CLF4b
  settled that explicitly ("the predictions artifact gains a per-trace split + prediction … *not* a
  mutation of the source bank") — but encoding the exact rule means that if predictions are ever
  attached in place, the id is forced to keep up instead of going quietly stale.
- **The asymmetry worth naming:** a `upred_` bank that *does* carry labels is the more damaging
  direction. It doesn't produce a wrong number — a consumer seeing `upred_` skips ground-truth
  comparison entirely, so it produces a **silently missing evaluation** the data could have supported.
  Harder to notice than a bad value.
- **Wired into the writer, not offered to callers.** A mislabelled bank cannot be written, and the
  failed write leaves no file behind — so the producer learns while it can still fix it, and the bad
  artifact never reaches a phase manifest where other artifacts start pointing at it.

### S12 — Docs + phase-exit ☐ (1–3 h)
- **Change:** `docs/usage.md` (the 2.0 bank example, the converter's new label behavior, the fact that
  θ is read from the typed `SyntheticBank` and **not** from a ClassifierBank, the dropped `host` key),
  `project/classifier_bank_format.md` if S10 touches the format, `CHANGELOG.md` `[Unreleased]` → the
  release entry, `roadmap.md` — remove the now-shipped `iafdb_bank` 1.3 and `noise_bank` `bank_id` /
  `produced_by`-optional items, retire the Phase-2 "Polymorphic `stimulation` encode/decode" entry
  (pulled forward into this phase and shipped as part of P2), renumber the deferred
  now-shipped Phase-1.5 pointer block (the roadmap was aligned at planning on 2026-07-29 — the
  renumbering, the Phase-2 removal, and the deferred correlation-join + noise-provenance entries are
  already in). Version bump to **v0.6.0** per design note 1.
- **Verify:** the full pre-PR run in `intracardiac-platform/project/pr_checklist.md` — `ruff format src
  tests` **and** `ruff check`, `mypy`, `pytest`, docs sync, CHANGELOG.
- **Depends on:** all prior steps.

**Parallelism:** S2–S4 are mutually independent. The DAT1 chain is S5→S6→S7→S8 (S8 also needs S5's
typed reader). S9→S10 is a second chain that runs alongside it, and S11 is independent of everything.
Only S1 and S12 are serializing.

## Effort tracking

Method: `intracardiac-platform/project/investigations/estimate_vs_actual_tracking.md`.
Daniel speaks the markers (`start` / `switch` / `break` / `resume` / `stop`); this chat stamps the time
from `date` and computes active = marked span − breaks. **Backstop:** unmarked silence > **2h** = away.
Rolls up at cleanup to design §6, §11, and `estimation_ledger.csv`.

> **Not tracked this phase** *(Daniel, 2026-07-29)*. Flow-down tracking, organization, and effort
> tracking were **missed for Phase 1.5** across the project. Daniel + the project-lead chat will design
> the methodology and it lands in a **later phase**, not retrofitted here — so this repo's session log
> stays empty, no reconstruction from transcripts is attempted, and `Actual` / `Elapsed` below are
> **not filled at cleanup**. egm-data contributes **estimates only** to §6 and appends **no row** to
> `estimation_ledger.csv` for 1.5. This supersedes CL-036's `measured=reconstructed` instruction and
> CL-024 §5b's flow-down-counts-as-issue-work rule *for this phase* — both stand as method for whenever
> tracking actually starts. Consequence to accept knowingly: 1.5 yields **no calibration data**, so the
> ledger stays cold and Phase-2 estimates are still by-analogy.

The method below is retained for the phase that adopts it.

### Estimate basis

The ledger is **empty** (header only), so per the §8 cold-start rule these are **estimates by analogy
with deliberately wide ranges**, not `points × rate`. Complexity is scored against the published
anchors: **S** = the `noise_bank` `bank_id` add (P5) — which is literally S2 of this plan — and **L** =
the SEP12 `synthetic_bank` 2.0 migration, of which DAT1 is the I/O half. Implied rate lands at roughly
1–2.8 h/point; the first cleanup replaces all of this with measured velocity.

| Issue | Task-type | Cx | Estimate | Steps | Active | Elapsed | Sessions |
|---|---|---|---|---|---|---|---|
| DAT1 (incl. the 1b T4 join) | schema | **XL = 8** | 8.5–17 h | S5–S8 | | | |
| DAT3 (+B18/B15/B14) | schema | **M = 3** | 3–7 h | S9–S10 | | | |
| B16 | schema | **S = 2** | 2–5 h | S11 | | | |
| B20 | schema | **S = 2** | 1–3 h | S2 | | | |
| B11 + CL-053 `activation_position` | schema | **S = 2** | 1–3 h | S3 | | | |
| B19 | schema | **XS = 1** | 0.5–2 h | S4 | | | |
| *re-pin + docs + PR* | overhead | — | 1.5–5 h | S1, S12 | | | |
| **Repo total** | | **18 pts** | **17.5–42 h** | | | | |

Sizing rationale: **DAT1 = XL**, settled after three revisions on 2026-07-28. The reader / writer /
θ-free conversion core is a clean **L**, matched to the published SEP12 anchor (breaking restructure,
mechanical serialization, no new algorithm). CL-024 §2 then added the **1b T4 join** — a fourth module
and a new capability, but the *narrow* one: a typed-object join on a single key, with no
`TunedParam.path` grammar and nothing persisted. That is +3 points to **XL = 8**, and it is a **low XL**
— well short of the STU4 anchor (a sub-package with a GP emulator), and cheaper than the XL this row
briefly carried mid-day, which also included a generic path resolver and a `ClassifierBank` 0.3 bump.
Worth flagging for the ledger so the XL reference class isn't skewed by a row that only just crossed
the threshold. **DAT3 = M**, raised from S on 2026-07-28 when the
CSV was folded in (design note 5): still purely additive with zero novelty, but now two modules and two
on-disk formats to round-trip instead of one — which sits it right at the published M anchor (CLF2
train-split metrics), whose producer side this is the I/O half of. **B16 = S** — single module, new
check, ordinary test burden. **B19 = XS** — a test that pins existing behavior. **B20 = S**
rather than XS only to stay consistent with the published anchor. **B11 raised XS → S on 2026-07-29**
when CL-053 added `activation_position`: the step now carries a per-trace *column* through both reader
and writer, not just an optional root attr, which is the same shape of work as B20's — plus the
negative test that keeps it out of the ClassifierBank.

### Session log

<!-- one row per marker; Daniel speaks the marker, the chat stamps the time from `date` -->

| Timestamp (local) | Event | Focus (issue) | Note |
|---|---|---|---|
| | | | |

## Notes / decisions log

- **2026-07-28** — Plan created at flow-down. Scope confirmed with Daniel as **all six schema groups**
  (P1–P6), not just the §3 core rows DAT1/DAT3, per the coverage note in
  `cross_artifact_linkage_design.md`. Delivery = one branch, staged commits, one PR.
- **2026-07-28** — Found P3's egm-data half (per-trace `split` + `prediction`) **already shipped** in
  `ClassifierTrace` + `classifier_bank_io`; P3 reduces to the B16 content check here. Design note 3.
- **2026-07-28** — Three items escalated to the project-lead: the **v0.6.0 vs v0.5.x** release-version
  question (note 1), the **`SyntheticLabelFn` signature change** visible to egm-classifier + egm-studio
  (note 2), and whether **`metrics.csv` gains train columns** alongside the run record (note 5).
- **2026-07-28** — **Note 5 answered — yes, train metrics go to the CSV too** (Daniel). Split S8 into
  S8 (run-record JSON) + **new S9** (metrics CSV); renumbered B16 → S10 and docs → S11. **DAT3 rescored
  S → M**; repo total 13 pts / 14–36 h → **14 pts / 15–38 h**. The escalation *sharpens* rather than
  closes: `training_metrics.schema.json` is `additionalProperties: false`, so this is a genuine
  contracts change absent from P1, and S11 is blocked until egm-contracts adds it. Recommended column
  ordering + the no-`train_reliability` call are recorded in note 5 for the project-lead.
- **2026-07-28** — **DAT1 scope change (project-lead):** the egm-studio escalation on who joins per-sim
  config → per-trace θ resolved to **egm-data owns the join** (option a). Verified against the
  authoritative spec — design §3 DAT1 and `cross_artifact_linkage_design.md` P2's consumer-access
  bullet, both of which now name this repo, with STU6 explicitly reduced to a small migration as a
  result. Added **S8** (pure θ resolver) + **S9** (materialization + `ClassifierBank` 0.3); renumbered
  DAT3 → S10–S11, B16 → S12, docs → S13. **DAT1 rescored L → XL**; repo total 14 pts / 15–38 h →
  **17 pts / 18.5–46 h**. Two new escalations fall out: the **θ-path root vocabulary** is
  under-specified (note 8) and the typed bank-level θ-spec forces a **second breaking change in Wave 1**,
  `ClassifierBank` 0.2 → 0.3, which the design doc doesn't currently name (note 9).
- **2026-07-28** — **DAT1 corrected — θ comes back off the ClassifierBank** (project-lead, per
  `investigations/synthetic_bank_source_of_truth.md` §12; supersedes the θ-join note above). The
  purpose lens — the ClassifierBank is a *source-agnostic ML compression*, so generation params are not
  its business — flips Escalation-1 to option (b): egm-data exposes the typed `SyntheticBank` (already
  S5) and egm-studio's T4 views read θ from it; the ClassifierBank carries only signal, label, the
  `LabelPolicy` identity, and the `simulation_id` join key. Removed S8/S9 (θ resolver +
  materialization); renumbered back to 11 steps. **DAT1 rescored XL → L**; repo total 17 pts /
  18.5–46 h → **14 pts / 15–37 h**. **Both escalations from the previous note close:** the
  `ClassifierBank` 0.2 → 0.3 bump is retired (note 8 — Wave 1 is back to exactly one breaking change),
  and the θ-path root question is deferred with the join helper (note 9 — now low-priority, not
  blocking). The correlation join (predictions ⋈ ClassifierBank ⋈ `synthetic_bank`) stays egm-data's to
  own but is scoped with its view, not in 1.5; S11 logs it to `roadmap.md` and S7 verifies the join key
  is reachable from a prediction row so it stays possible. Signal duplication across the parallel banks
  is accepted for 1.5 and tracked platform-side as FB-11 — not an egm-data item.
- **2026-07-28** — **CL-024 batch adjudication applied.** §2: the T4 bank ⋈ bank join comes into DAT1
  this phase as **option 1b** (typed `SimulationConfig` per trace, no path grammar, read-time only) →
  new **S8**; DAT1 **L → XL**, repo 14 pts / 15–37 h → **17 pts / 17–41 h**. *(My CL-010 quoted "16 pts"
  for 1b — arithmetic slip: 5 → 8 is +3, so the repo total is 17. Hours were right. Flagged to the
  project-lead in CL-036 in case §6 copied the wrong figure.)* §3: producer renames `sim_id` →
  `simulation_id`, and egm-data adds a **writer key-name check** (folded into S7) — the only
  enforcement point, since the ClassifierBank has no JSON Schema (CL-012). §5b: flow-down time counts
  toward issue `Actual`, recorded in the Effort section. §4 (`T` = 192 ms on the 64-grid) touches no
  egm-data code. Design note 9's θ-path deferral is now ratified rather than proposed.
- **2026-07-28** — **CL-022 folded into S9** (egm-classifier): `make_epoch_record` gains
  `train_metrics=None` **keyword-optional**, so CLF5's Wave-1 migration can write 1.2 records without
  train metrics — that optionality is what keeps the migration wave separate from the feature wave. A
  `"reliability"` key inside `train_metrics` is **dropped rather than split out**, deliberately
  asymmetric with the val path, per P1's FB-10 deferral. Cannot self-resolve CL-022 under the CL-025
  write rules; replied via CL-036.
- **2026-07-29** — **CL-037 came back resolved on all four items**, so design notes 1, 2, 3, 5 are now
  records rather than open questions. The blocking one landed: the `training_metrics` train columns are
  folded into **P1 + CON3 for v0.6.0** with the **paired** column order and no `train_reliability`, so
  **S10 is unblocked** and its "may drop out" caveat is gone. §7's target is corrected to **egm-data
  v0.6.0**; the `SyntheticLabelFn`→override change was routed to egm-classifier + egm-studio as
  **CL-038**; **P3 was amended** to drop the already-shipped `split`/`prediction` columns.
- **2026-07-29** — **Effort tracking is not done for Phase 1.5** (Daniel): flow-down tracking,
  organization, and effort tracking were missed project-wide this phase; the methodology is being
  designed by Daniel + the project-lead for a later phase and is **not** retrofitted here. No transcript
  reconstruction, no `Actual`/`Elapsed` at cleanup, no `estimation_ledger.csv` row — estimates only.
  Supersedes CL-036's `measured=reconstructed` instruction for this repo; posted as CL-043 so the
  project-lead isn't waiting on a row that will never come.
- **2026-07-29** — **`roadmap.md` aligned to the phase** (the Planning-gate requirement): the Phase-1.5
  section now points at this plan rather than duplicating it; the deferred noise-side-calibration entry
  is renumbered `noise_bank` 1.1 → **1.2** (B20 owns 1.1); and the stale Phase-2 "polymorphic
  `stimulation` encode/decode" entry is removed, since the 2.0 restructure absorbed it into DAT1. This
  was previously scheduled for S12 — done now because the gate reads "roadmap updated" at planning.
- **2026-07-29** — **CL-053 folded into S3** (project-lead, flowing down CL-052): `iafdb_bank` 1.3 gains
  a per-trace `activation_position` `[0,1]` column, riding B11's already-happening bump. egm-data
  carries it on read + write only; it **stays on the `IafdbBank` model and must not reach the
  ClassifierBank** via `iafdb_bank_to_classifier` — same source-agnostic rule that keeps θ off it, and
  pinned by a negative test since the converter is exactly where it would leak. Field lands Wave 1
  unpopulated (IAF3); IAF1 fills it Wave 2. **B11 rescored XS → S**; repo total 17 pts / 17–41 h →
  **18 pts / 17.5–42 h**.
- **2026-07-31** — **S1 done; red window declared.** egm-contracts v0.6.0 landed everything asked for,
  including the CL-037 items (the six `train_*` CSV columns with the **paired** order, and
  `common.ActivationPosition` `$ref`'d by both banks). Re-pin measured: 30 pass / 10 error, ruff clean,
  33 mypy errors — **all** confined to the synthetic reader / writer / converter and the 1.1-shaped test
  fixture. S1's "suite green unchanged" was unachievable by construction for a breaking schema change;
  the window is declared rather than papered over with `xfail`, and closes at S7. See the note under S1.
- **2026-07-31** — **Erratum spotted in CL-089** (not mine to edit): its summary table says
  `phase_manifest` "`produced_by_*` left `required`", but the shipped schema has them **optional** in
  all seven entry `$defs`, and the CHANGELOG says so correctly. The table is the quick-reference people
  skim and egm-studio's curation change depends on it. Raised for correction.
- **2026-07-29** — **CL-062 folded into DAT1** — the synthetic twin of CL-053. `synthetic_bank` 2.0's
  `traces/` gains the same optional `activation_position`; S5 reads it, S6 writes it, S7 gets the
  matching negative test keeping it off the ClassifierBank. **No estimate change:** one optional float
  column inside steps that are already rewriting the entire `traces/` group is absorbed noise, and DAT1
  is already XL — inflating it would misreport the driver. CL-062's scope note names only the reader;
  flagged in CL-063 that the **writer** needs it too, since SEP12 writes through
  `write_synthetic_bank`.
- **2026-07-28** — **Noise-mixing provenance stays on the ClassifierBank for 1.5** (Daniel): no further
  data-structure churn this session. `snr_db` / `noise_record` / `noise_channel` / `seed` are redundant
  with the `synthetic_bank` (same `simulation_id` join) and are a candidate for removal later, so the
  optimization is written into `roadmap.md` — which outlives this ephemeral plan — as a sibling of
  FB-11, alongside the deferred correlation-join entry. S7 keeps today's behavior; nothing to decide.
