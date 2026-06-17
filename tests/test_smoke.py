"""Smoke test — confirms the package imports and __version__ is sane.

Keep this test alive for the life of the package; it's the canary for
broken installs and bad packaging metadata.
"""

from __future__ import annotations

import re

import myocard_egm_data


def test_package_imports() -> None:
    """Canary: the package imports cleanly. Catches packaging metadata
    breakage, bad __init__.py changes, and missing dependencies at
    install time."""
    assert myocard_egm_data is not None


def test_version_is_pep440() -> None:
    """The exported __version__ should at least start with a PEP-440-style
    `MAJOR.MINOR.PATCH`. Guards against an empty or garbled version
    metadata getting into a release wheel."""
    # Loose PEP 440 check — accepts X.Y.Z, X.Y.Z.devN, X.Y.ZrcN, etc.
    assert re.match(r"^\d+\.\d+\.\d+", myocard_egm_data.__version__), (
        f"non-PEP440 version: {myocard_egm_data.__version__!r}"
    )
