"""Shared serialization plumbing — Pydantic <-> strict, deterministic JSON.

Package-level helpers used by every subpackage that reads/writes a typed
egm-contracts model as JSON: ``records/`` (run.json, model_metadata.json,
noise_bank_run_record.json) and ``phases/`` (manifest.json, observations,
figure specs). Kept here, rather than under one of those subpackages, so
neither has to reach into the other's internals.

Provides:

- ISO-8601 UTC timestamp helper (every ``build_*`` stamps one).
- NaN/Inf sanitization for numpy-derived metric dicts before Pydantic
  construction (Pydantic v2's ``model_dump_json`` raises on non-finite
  floats by default).
- Pydantic -> strict, deterministic JSON writer (sort_keys, allow_nan=False).
- Pydantic JSON reader that returns a typed model instance.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

_M = TypeVar("_M", bound=BaseModel)


def _utc_now() -> _dt.datetime:
    """Return the current UTC time as a timezone-aware ``datetime``.

    Used by every ``build_*`` helper to stamp the ``created_utc``
    field. We return a ``datetime`` rather than an ISO string so mypy
    is happy at the Pydantic-model construction site — Pydantic
    serializes datetimes to ISO strings via ``model_dump(mode="json")``
    on the way out anyway.
    """
    return _dt.datetime.now(_dt.timezone.utc)


def _sanitize_floats(value: Any) -> Any:
    """Recursively replace NaN/Inf floats with ``None``.

    Numpy-derived metric dicts (notably ``EpochRecord.val_metrics``)
    can legitimately carry NaN — e.g. AUROC on a single-class
    validation split. Calling this on the dict BEFORE constructing
    the Pydantic model keeps the result JSON-serializable downstream:
    Pydantic v2's ``model_dump_json`` and the stdlib ``json.dump`` (with
    ``allow_nan=False``) both raise on non-finite floats.

    Numpy scalars are unwrapped via ``.item()`` if present, then
    re-sanitized so the recursive call catches non-finite numpy floats.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {k: _sanitize_floats(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_floats(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes, BaseModel)):
        try:
            return _sanitize_floats(value.item())
        except (ValueError, TypeError):
            return value
    return value


def _write_pydantic_json(path: Path | str, record: BaseModel) -> Path:
    """Serialize a Pydantic record to a strict, deterministic JSON file.

    - ``model_dump(mode="json")`` converts every nested field to its
      JSON form (``datetime`` → ISO string, ``Path`` → ``str``,
      ``Enum`` → its value, etc.).
    - ``exclude_defaults=True`` drops fields whose value equals their
      declared default. This matches the JSON Schema convention used
      across this package: optional fields (e.g.
      ``per_trace_provenance``, ``model_artifact.size_bytes``) are
      declared in Pydantic as ``X | None = None``; when the producer
      leaves them at their default, the JSON omits the key entirely
      (schema rejects ``null`` for these because they aren't typed as
      nullable). Required-nullable fields (e.g.
      ``calibration.target_qrs_pp_mv``) are declared as ``X | None``
      with NO default, so they're always emitted — including as JSON
      ``null`` when the producer explicitly passes None — which the
      schema accepts via ``type: ["X", "null"]``.
    - ``sort_keys=True`` makes the byte stream deterministic across
      runs, so the same logical record always produces the same bytes
      (useful for content hashes / diffs / cache keys).
    - ``allow_nan=False`` enforces strict JSON: any non-finite float
      that slipped past ``_sanitize_floats`` raises at write time
      rather than producing a non-spec ``NaN`` / ``Infinity`` token.

    Returns the resolved ``Path`` for convenient chaining.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(
            record.model_dump(mode="json", exclude_defaults=True),
            f,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    return p


def _load_pydantic_json(path: Path | str, model_cls: type[_M]) -> _M:
    """Load a JSON file and validate-construct the given Pydantic model.

    Raises ``pydantic.ValidationError`` on shape mismatch — readers
    therefore do not need a separate validation step against
    ``myocard-egm-contracts.validators.*``; the Pydantic model IS the
    validator (the contracts validators add JSON-Schema-level checks
    on top, but the typed read path already enforces the data shape).
    """
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{p}: expected a JSON object at the top level.")
    return model_cls.model_validate(raw)
