"""ClassifierBank — the unified, generic data structure consumed by every
ML classification task in this stack.

ClassifierBank is the *in-memory* contract every consumer (training,
eval, viewer, paper figures) agrees on. Source-specific banks (synthetic,
IAFDB, hypothetical future formats) get converted *into* ClassifierBank
via the converters in ``banks/converters.py``; downstream code only
knows ClassifierBank exists.

Why this shape:

- **Source-agnostic.** Trace-level provenance lives in
  ``trace_metadata`` (a generic dict). Bank-level provenance lives in
  ``ClassifierBankMetaData.bank_metadata`` (also a generic dict). New
  source types add their own keys without changing the dataclass.
- **Multi-bank in a single ClassifierBank.** The `banks` list records
  the provenance of every source bank that contributed to this
  ClassifierBank; each trace references its source via ``bank_id``.
  This is how a "hybrid" ClassifierBank (synthetic + IAFDB) is built —
  see :meth:`ClassifierBank.concat`.
- **Predictions live with the trace.** Eval-time outputs go onto
  ``ClassifierTrace.prediction``. The standalone predictions
  CSV/JSON format that used to ship in egm-contracts is retired.
- **Labels translated by the producer.** ``ClassifierBank.labels``
  maps integer labels (as used by ``label_truth`` / ``label_pred``)
  to human-readable strings. The converter sets this based on
  conversion-time policy.

On-disk format is HDF5. See ``egm-data/project/classifier_bank_format.md``
for the layout. Version is ``CLASSIFIER_BANK_VERSION``.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from myocard_egm_contracts import common as _contracts_common
from pydantic import ValidationError

CLASSIFIER_BANK_VERSION = "0.2"
"""On-disk and in-memory schema version for ClassifierBank.

Bump on any breaking change to the dataclass shape or the HDF5 layout.
Versions start at 0.X during pre-1.0 iteration; X bumps on every
structural change. Lives here rather than in egm-contracts because
ClassifierBank is an egm-data concept (a JSON Schema would force the
on-disk format into a JSON-shaped Pydantic model, which is awful for
numpy arrays — see the project-doc discussion).

0.2 (from 0.1) reworked the cross-artifact linkage: every bank now
carries an optional top-level stable artifact ``id``, and the
per-source / per-trace ``bank_id`` became the source bank's stable
ArtifactId (it was an integer index into ``banks``). This is a clean
break from 0.1 — 0.1 files (integer ``bank_id``) are not read.
"""

SUPPORTED_CLASSIFIER_BANK_VERSIONS: tuple[str, ...] = ("0.2",)
"""On-disk versions the reader accepts. Writers always stamp
:data:`CLASSIFIER_BANK_VERSION`. 0.1 (integer ``bank_id``) is
intentionally unsupported — the project is pre-1.0, so previously
generated 0.1 banks are regenerated rather than migrated."""


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Per-trace prediction record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassifierPrediction:
    """ML prediction outputs for one trace.

    Attributes
    ----------
    label_pred
        Hard label assignment (the argmax over class probabilities).
    label_prob
        The probability of ``label_pred`` (i.e. the post-softmax
        probability of the predicted class, in [0, 1]).
    pred_logits
        Pre-softmax logits keyed by integer label. Storing every logit
        (not just the predicted one) enables confusion analysis later,
        e.g. "which two classes tend to be confused for each other?".
        For a binary model this is ``{0: logit_neg, 1: logit_pos}``.
    """

    label_pred: int
    label_prob: float
    pred_logits: dict[int, float]


# ---------------------------------------------------------------------------
# Per-trace record
# ---------------------------------------------------------------------------


@dataclass
class ClassifierTrace:
    """One per-trace record.

    The signal is what the classifier sees (after any source-side
    preprocessing). Longer source recordings get pre-chunked into one
    trace per slice before they land here.

    ``split``, ``label_truth``, and ``prediction`` are all populated
    at different lifecycle stages and may be ``None`` until they are.

    Per-trace provenance (patient_id, source_record, electrode position,
    etc.) lives in ``trace_metadata`` rather than as first-class fields
    so that adding new source types or split criteria doesn't keep
    growing the dataclass. Consumers that need a specific key
    (``patient_aware_split`` reading ``patient_id``, etc.) just look it
    up by name.

    Attributes
    ----------
    bank_id
        Stable id of the source bank this trace came from — references
        one entry in :attr:`ClassifierBank.banks` by its ``bank_id``
        (an egm-contracts ArtifactId, e.g.
        ``tbank_synthetic_courtemanche_v1_5_2026-06-25``).
    signal
        ``[T]`` float32 array — the EGM waveform. Length may exceed the
        classifier's input length, in which case the dataset wrapper
        decides what slice to feed the model (random crop at train time,
        centered at eval). All traces in a single ClassifierBank are
        required to share the same length (Phase 1 limitation).
    freq_hz
        Sampling rate in Hz. Per-trace because different source banks
        could in principle ship at different rates (current sources are
        all 1 kHz).
    amp_type
        Identifier for the amplitude convention of ``signal``. Known
        values: ``"mv"`` (raw millivolts), ``"z_score"`` (per-trace
        z-scored), ``"normalized"`` (other normalization). New schemes
        add new strings; the consumer is responsible for honoring it.
    split
        Which split this trace belongs to (``"train"``, ``"val"``,
        ``"test"``, ``"eval"``, etc.). ``None`` before splitting.
    label_truth
        Ground-truth integer label. ``None`` if no ground truth is
        known (e.g. unlabeled inference-time data).
    prediction
        Eval-time prediction record. ``None`` before eval.
    trace_metadata
        Generic per-trace provenance dict. Convention: include
        ``patient_id`` for patient-aware splitting; include any other
        source-specific keys (fibrosis_density, peak_to_peak_mv, etc.)
        verbatim from the source bank.
    """

    bank_id: str
    signal: np.ndarray
    freq_hz: float
    amp_type: str
    split: str | None
    label_truth: int | None
    prediction: ClassifierPrediction | None
    trace_metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Per-source-bank metadata record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassifierBankMetaData:
    """Bank-level provenance for one source bank that contributed traces.

    A ClassifierBank can carry traces from multiple source banks
    (for example a synthetic bank combined with an IAFDB bank). Each
    ClassifierTrace references its source via ``bank_id``, the source
    bank's stable cross-artifact id.

    Attributes
    ----------
    bank_id
        Stable cross-artifact id of this source bank (an egm-contracts
        ArtifactId, e.g. ``tbank_synthetic_courtemanche_v1_5_2026-06-25``).
        Traces reference their source by this id. Unique across the
        ClassifierBank's ``banks`` list; ``concat`` dedups by it. Replaces
        the integer index used before ClassifierBank 0.2.
    bank_type
        Identifier for the source format that produced this entry —
        e.g. ``"synthetic"``, ``"iafdb"``. New source types add new
        strings.
    bank_path
        Path the source bank was loaded from (provenance only).
    bank_metadata
        Generic bank-level provenance dict. Synthetic banks carry
        simulator config, fibrosis strategy, electrode grid, etc.;
        IAFDB banks carry calibration method, threshold mode, band-pass
        edges, etc.
    """

    bank_id: str
    bank_type: str
    bank_path: str
    bank_metadata: dict[str, Any]

    def __post_init__(self) -> None:
        # Validate the source bank's stable id against the shared
        # egm-contracts pattern (common.ArtifactId), the same way
        # ClassifierBank.id is validated. Frozen dataclass: read + raise only.
        try:
            _contracts_common.ArtifactId(self.bank_id)
        except ValidationError as exc:
            raise ValueError(
                f"ClassifierBankMetaData.bank_id {self.bank_id!r} is not a "
                "valid stable artifact id (egm-contracts ArtifactId pattern, "
                "e.g. 'tbank_synthetic_courtemanche_v1_5_2026-06-25')."
            ) from exc


# ---------------------------------------------------------------------------
# ClassifierBank — the top-level container
# ---------------------------------------------------------------------------


@dataclass
class ClassifierBank:
    """A unified collection of classifier-ready traces.

    Attributes
    ----------
    schema_version
        On-disk and in-memory schema version
        (:data:`CLASSIFIER_BANK_VERSION` at creation time).
    created_utc
        ISO-8601 UTC timestamp captured at construction.
    id
        Stable cross-artifact identifier for this bank when it is a
        saved, tracked artifact — e.g. a predictions bank
        (``upred_..._<date>`` / ``lpred_..._<date>``). ``None`` for
        intermediate in-memory banks that aren't tracked artifacts (a
        concat result, a freshly converted source bank, etc.). Note this
        is the *bank's own* stable id and is distinct from the integer
        per-source ``bank_id`` carried on traces + source-bank entries.
    banks
        List of source-bank provenance entries. Indexed by
        :attr:`ClassifierTrace.bank_id`.
    traces
        Flat list of all traces from every contributing source bank.
    labels
        Translation from integer labels (as used by
        ``ClassifierTrace.label_truth`` and
        ``ClassifierPrediction.label_pred``) to human-readable strings,
        e.g. ``{0: "healthy", 1: "fibrotic"}``. The producer sets this
        based on the labeling policy applied during conversion.
    """

    schema_version: str = field(default_factory=lambda: CLASSIFIER_BANK_VERSION)
    created_utc: str = field(default_factory=_utc_now)
    id: str | None = None
    banks: list[ClassifierBankMetaData] = field(default_factory=list)
    traces: list[ClassifierTrace] = field(default_factory=list)
    labels: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate the optional stable artifact id against the shared
        # egm-contracts pattern (common.ArtifactId). The field stays a plain
        # str for dataclass + HDF5 ergonomics, but a malformed id is rejected
        # at construction time so an invalid id can never exist on a bank.
        if self.id is not None:
            try:
                _contracts_common.ArtifactId(self.id)
            except ValidationError as exc:
                raise ValueError(
                    f"ClassifierBank.id {self.id!r} is not a valid stable "
                    "artifact id (expected the egm-contracts ArtifactId "
                    "pattern, e.g. 'upred_iafdb_v1_5_2026-06-25')."
                ) from exc

    # ---- derived helpers ------------------------------------------------

    @property
    def n_traces(self) -> int:
        return len(self.traces)

    @property
    def n_samples_first(self) -> int:
        """Length of the first trace, for quick inspection / logging.

        Traces in a ClassifierBank are NOT required to share a length.
        The training-time :class:`~myocard_egm_data.augmentation.TraceTransform`
        handles per-trace pad/crop, so a bank can carry mixed-length
        traces and the dataset wrapper will still produce a fixed-T
        tensor for the model. Use this property only for diagnostics; if
        you need a uniform columnar view (e.g. to feed the dataset
        directly), call :meth:`signal_array` which will raise on a
        non-uniform bank.
        """
        if not self.traces:
            raise ValueError("ClassifierBank has no traces.")
        return int(self.traces[0].signal.shape[0])

    def signal_array(self) -> np.ndarray:
        """Stack every trace's signal into one ``[N, T]`` float32 array.

        Convenience for downstream code that wants the columnar view
        (e.g. the PyTorch dataset wrapper). Requires all traces to share
        the same length and raises a clear error otherwise — for
        mixed-length banks, use :meth:`signal_list` and have the
        consumer build its own pad/crop pipeline.
        """
        if not self.traces:
            return np.empty((0, 0), dtype=np.float32)
        T = self.traces[0].signal.shape[0]
        for i, t in enumerate(self.traces):
            if t.signal.shape[0] != T:
                raise ValueError(
                    f"ClassifierBank.signal_array() requires uniform trace "
                    f"length. Trace 0 has length {T}; trace {i} has length "
                    f"{t.signal.shape[0]}. Use signal_list() if you want to "
                    "iterate mixed-length traces."
                )
        return np.stack([t.signal.astype(np.float32, copy=False) for t in self.traces])

    def signal_list(self) -> list[np.ndarray]:
        """Return ``[N]`` list of per-trace ``[T_i]`` float32 arrays.

        Works for both uniform and mixed-length banks; the consumer
        decides how to handle the per-row length (pad, crop, vlen,
        etc.).
        """
        return [t.signal.astype(np.float32, copy=False) for t in self.traces]

    def label_truth_array(self) -> np.ndarray:
        """Stack every trace's ``label_truth`` into ``[N]`` int64.

        Raises if any trace has no ground-truth label (caller must
        ensure all traces are labeled before asking for the array).
        """
        out: list[int] = []
        for i, t in enumerate(self.traces):
            if t.label_truth is None:
                raise ValueError(
                    f"Trace at index {i} has no label_truth; can't build "
                    "a label array. Either label all traces or use a "
                    "per-trace iteration."
                )
            out.append(int(t.label_truth))
        return np.asarray(out, dtype=np.int64)

    def patient_id_array(self) -> np.ndarray:
        """Stack ``trace_metadata['patient_id']`` into ``[N]`` object array.

        Convention: every trace carries a ``patient_id`` key in
        ``trace_metadata``. The patient-aware splitter reads this. The
        converter is responsible for setting it (synthetic stringifies
        ``simulation_id``; IAFDB uses the IAFDB patient id verbatim).
        """
        out: list[str] = []
        for i, t in enumerate(self.traces):
            pid = t.trace_metadata.get("patient_id")
            if pid is None:
                raise ValueError(
                    f"Trace at index {i} has no 'patient_id' in "
                    "trace_metadata; can't run a patient-aware split."
                )
            out.append(str(pid))
        return np.asarray(out, dtype=object)

    # ---- concat ---------------------------------------------------------

    @classmethod
    def concat(cls, banks: list[ClassifierBank]) -> ClassifierBank:
        """Merge multiple ClassifierBanks into one.

        Raises ``ValueError`` if the ``labels`` dicts disagree across
        banks (per the locked-in design rule: integer labels mean the
        same thing in every contributing bank, or concat is a bug).

        Source-bank entries are deduplicated by their stable ``bank_id``
        (first occurrence wins); traces keep their ``bank_id`` references
        unchanged, since stable artifact ids are globally unique and don't
        collide. The merged bank's own ``id`` is left unset — it's a new
        artifact whose id the caller assigns when saving it.

        Schema versions across all banks must match — concat is an
        in-memory operation, not a format conversion.
        """
        if not banks:
            raise ValueError("ClassifierBank.concat requires at least one bank.")

        # Labels dicts must match — see docstring for rationale.
        head_labels = banks[0].labels
        for i, b in enumerate(banks[1:], start=1):
            if b.labels != head_labels:
                raise ValueError(
                    f"ClassifierBank.concat: labels dict at index {i} "
                    f"({b.labels}) disagrees with index 0 ({head_labels}). "
                    "Concat requires identical label dictionaries."
                )

        # Schema versions must match.
        head_ver = banks[0].schema_version
        for i, b in enumerate(banks[1:], start=1):
            if b.schema_version != head_ver:
                raise ValueError(
                    f"ClassifierBank.concat: schema_version at index {i} "
                    f"({b.schema_version!r}) disagrees with index 0 "
                    f"({head_ver!r})."
                )

        # Merge source-bank entries, deduplicating by stable bank_id
        # (first occurrence wins). No id remap: stable ids are unique.
        new_banks: list[ClassifierBankMetaData] = []
        seen_bank_ids: set[str] = set()
        for b in banks:
            for meta in b.banks:
                if meta.bank_id in seen_bank_ids:
                    continue
                seen_bank_ids.add(meta.bank_id)
                new_banks.append(
                    ClassifierBankMetaData(
                        bank_id=meta.bank_id,
                        bank_type=meta.bank_type,
                        bank_path=meta.bank_path,
                        bank_metadata=dict(meta.bank_metadata),
                    )
                )
        # Concatenate traces verbatim — their stable bank_id references
        # already point at the (deduped) source-bank entries.
        new_traces: list[ClassifierTrace] = []
        for b in banks:
            for tr in b.traces:
                new_traces.append(
                    ClassifierTrace(
                        bank_id=tr.bank_id,
                        signal=tr.signal,
                        freq_hz=tr.freq_hz,
                        amp_type=tr.amp_type,
                        split=tr.split,
                        label_truth=tr.label_truth,
                        prediction=tr.prediction,
                        trace_metadata=dict(tr.trace_metadata),
                    )
                )

        return cls(
            schema_version=head_ver,
            created_utc=_utc_now(),
            banks=new_banks,
            traces=new_traces,
            labels=dict(head_labels),
        )
