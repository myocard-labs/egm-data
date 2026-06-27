"""Read + write a figure spec ``<id>.json`` (egm-contracts figure_spec).

A figure spec is the git-tracked source of truth for one publication
figure: a required ``description``, a ``recipe`` name (egm-studio owns the
recipe vocabulary), recipe-specific ``inputs`` / ``layout`` / ``styling``,
and the ``output`` target. egm-studio's headless render CLI reads the spec
and regenerates the (gitignored) image.
"""

from __future__ import annotations

from pathlib import Path

from myocard_egm_contracts._generated.python.figure_spec import (
    FigureSpec,
    Group,
    Inputs,
    Output,
)

from .._serialization import _load_pydantic_json, _write_pydantic_json

__all__ = [
    "FigureSpec",
    "Group",
    "Inputs",
    "Output",
    "load_figure_spec",
    "write_figure_spec",
]


def load_figure_spec(path: Path | str) -> FigureSpec:
    """Load a figure-spec ``.json`` document into a typed :class:`FigureSpec`.

    Raises ``pydantic.ValidationError`` on shape mismatch (missing
    ``description``, non-``fig_`` id, bad ``output.format``, ...).
    """
    return _load_pydantic_json(path, FigureSpec)


def write_figure_spec(path: Path | str, spec: FigureSpec) -> Path:
    """Write a :class:`FigureSpec` to ``path`` as strict, deterministic JSON."""
    return _write_pydantic_json(path, spec)
