"""Public codenames for wells (confidential data: real well ids never appear on shared figures).

The mapping lives in ``<data root>/well_codenames.json`` (git-ignored, next to ``raw/``).
RONALDO and MESSI are the codenames used in the 2024 thesis.
Processing always uses real ids; call :func:`codename` only when labelling outputs.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path


def _mapping_path() -> Path:
    default = Path(__file__).resolve().parents[2] / "data" / "raw"
    raw = Path(os.environ.get("MARS_DATA_DIR", default))
    return raw.parent / "well_codenames.json"


@lru_cache(maxsize=1)
def _mapping() -> dict[str, str]:
    p = _mapping_path()
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def codename(well: str, strict: bool = True) -> str:
    """Public codename of ``well``. Unknown wells raise (``strict``) rather than leak real names."""
    m = _mapping()
    if well in m:
        return m[well]
    if strict:
        raise KeyError(f"no codename for well {well!r}: add it to {_mapping_path()}")
    return "WELL-?"


def codenames(wells) -> list[str]:
    return [codename(w) for w in wells]
