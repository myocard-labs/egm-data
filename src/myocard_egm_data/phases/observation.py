"""Read + write an observation ``<id>.json`` (egm-contracts observation).

An observation records a discovery during signal exploration / ML
diagnostics: a required free-text ``description``, plus an optional pinned
trace list and an optional egm-studio ``view_state``. egm-studio's Save
flow writes these; ``validate_manifest.py`` and paper-figure tooling read
them.
"""

from __future__ import annotations

from pathlib import Path

from myocard_egm_contracts._generated.python.observation import (
    Observation,
    References,
    TraceRef,
    ViewState,
)

from .._serialization import _load_pydantic_json, _write_pydantic_json

__all__ = [
    "Observation",
    "References",
    "TraceRef",
    "ViewState",
    "load_observation",
    "write_observation",
]


def load_observation(path: Path | str) -> Observation:
    """Load an observation ``.json`` document into a typed :class:`Observation`.

    Raises ``pydantic.ValidationError`` on shape mismatch (missing
    ``description``, bad id pattern, ...).
    """
    return _load_pydantic_json(path, Observation)


def write_observation(path: Path | str, observation: Observation) -> Path:
    """Write an :class:`Observation` to ``path`` as strict, deterministic JSON."""
    return _write_pydantic_json(path, observation)
