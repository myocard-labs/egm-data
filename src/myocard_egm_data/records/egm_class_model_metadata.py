"""Build, write, and read the EGM-classifier ``model_metadata.json`` sidecar.

The schema (``myocard_egm_contracts.egm_class_model_metadata``)
captures everything an inference runtime needs that the ONNX graph
itself does not encode: expected sample rate, expected trace length,
per-trace normalization scheme, decision threshold, model artifact
hash, training provenance.

Cross-language consumer: Python (training-eval parity checks) and
C++ (TensorRT inference). Lives next to the model artifact on disk
(e.g. ``best.onnx`` + ``best.model_metadata.json``).

This schema is intentionally specific to the 1-D EGM-classifier
family — the ``[B, 1, T]`` input (middle ``1`` is the channel axis
required by PyTorch ``Conv1d``, not a multi-channel slot),
``binary_logit`` output semantics, and single-channel bandpass
preprocessing don't generalize to 2-D electrode-grid CNNs or sparse
3-D electrode point-cloud models, which will get their own
per-topology schemas.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from myocard_egm_contracts._generated.python.egm_class_model_metadata import (
    Decision,
    EgmClassModelMetadata,
    Input,
    ModelArtifact,
    Output,
    Preprocessing,
)
from myocard_egm_contracts._generated.python.egm_class_model_metadata import (
    SchemaVersion as EgmClassModelMetadataSchemaVersion,
)
from myocard_egm_contracts.schema_info import current_version

from .._serialization import _load_pydantic_json, _utc_now, _write_pydantic_json

__all__ = [
    "Decision",
    "EgmClassModelMetadata",
    "Input",
    "ModelArtifact",
    "Output",
    "Preprocessing",
    "build_egm_class_model_metadata",
    "load_egm_class_model_metadata",
    "write_egm_class_model_metadata",
]


def build_egm_class_model_metadata(
    *,
    model_artifact: ModelArtifact | Mapping[str, Any],
    input_spec: Input | Mapping[str, Any],
    output_spec: Output | Mapping[str, Any],
    preprocessing: Preprocessing | Mapping[str, Any],
    decision: Decision | Mapping[str, Any],
    training_provenance: Mapping[str, Any],
    model_id: str | None = None,
) -> EgmClassModelMetadata:
    """Assemble an :class:`EgmClassModelMetadata` with version + timestamp stamped.

    Each sub-section may be passed as a typed Pydantic instance OR a
    plain mapping — Pydantic validates dicts into the nested
    sub-models at construction time, so producers can hand-build the
    decision block as a dict while passing a typed
    :class:`ModelArtifact` from a registry.

    The kwargs ``input_spec`` / ``output_spec`` are spelled out
    (rather than ``input`` / ``output``) because ``input`` shadows the
    Python builtin; the underlying schema field names are still
    ``input`` and ``output``.

    ``model_id`` is the optional stable cross-artifact id of this model
    (egm-contracts v0.5.0) — what a run.json's ``produced_model_id``
    points at. Optional in the schema; egm-classifier stamps it at export
    time. The model->run link is intentionally NOT stored here (the run
    record owns it); pass a value matching the ``common.ArtifactId`` shape
    (e.g. ``model_egm_classifier_v1_5_2026-06-25``) or ``None``.
    """
    return EgmClassModelMetadata(
        schema_version=EgmClassModelMetadataSchemaVersion(
            current_version("egm_class_model_metadata"),
        ),
        created_utc=_utc_now(),
        model_id=model_id,
        # ``model_validate`` accepts either a dict or an existing
        # instance and returns the typed sub-model — gives mypy the
        # concrete type it wants while preserving the dict-ergonomic
        # API at the call site.
        model_artifact=ModelArtifact.model_validate(model_artifact),
        input=Input.model_validate(input_spec),
        output=Output.model_validate(output_spec),
        preprocessing=Preprocessing.model_validate(preprocessing),
        decision=Decision.model_validate(decision),
        training_provenance=dict(training_provenance),
    )


def write_egm_class_model_metadata(path: Path | str, record: EgmClassModelMetadata) -> Path:
    """Write an :class:`EgmClassModelMetadata` to ``path`` as strict, deterministic JSON."""
    return _write_pydantic_json(path, record)


def load_egm_class_model_metadata(path: Path | str) -> EgmClassModelMetadata:
    """Load an EGM-classifier ``model_metadata.json`` document into a typed model.

    Raises ``pydantic.ValidationError`` on shape mismatch.
    """
    return _load_pydantic_json(path, EgmClassModelMetadata)
