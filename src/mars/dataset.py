"""Site-specific facts about the (confidential) data set: well lists and file naming.

The code never names a real well or the field. Both live in ``<data root>/dataset.json``, next to
``raw/`` and ``well_codenames.json`` (git-ignored); ``configs/dataset.example.json`` shows the
layout. Without the file every well set is empty and LAS files are looked up by the bare well id,
so the package and the synthetic tests still work.

Keys:
  las_prefix      prefix of LAS file stems, e.g. ``"<Field>_"`` -> ``<Field>_<well>.las``
  unprefixed      well-id prefixes whose LAS files carry no field prefix
  well_sets       named lists of real well ids; ``all`` is ``first12 + added10``
  test_expectations  values from the real data that the data tests check
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path


def config_path() -> Path:
    default = Path(__file__).resolve().parents[2] / "data" / "raw"
    raw = Path(os.environ.get("MARS_DATA_DIR", default))
    return raw.parent / "dataset.json"


@lru_cache(maxsize=1)
def config() -> dict:
    p = config_path()
    if not p.exists():
        return {"las_prefix": "", "unprefixed": [], "well_sets": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def wells(name: str) -> tuple[str, ...]:
    """Named well set, e.g. ``first12``, ``added10``, ``all``, ``thesis_pair``, ``swt_exclude``."""
    sets = config().get("well_sets", {})
    if name == "all" and "all" not in sets:
        return tuple(sets.get("first12", ())) + tuple(sets.get("added10", ()))
    if name not in sets and sets:
        raise KeyError(f"well set {name!r} not in {config_path()}")
    return tuple(sets.get(name, ()))


def well(name: str) -> str:
    """The single well of a one-element set (e.g. ``measured_vs_reference``)."""
    (w,) = wells(name)
    return w


def las_stem(well: str, prefix: str | None = None) -> str:
    """LAS file stem / tops identifier of ``well`` (prefix + id, unless exempt)."""
    cfg = config()
    prefix = cfg.get("las_prefix", "") if prefix is None else prefix
    if not prefix or well.startswith(prefix) or well.startswith(tuple(cfg.get("unprefixed", []))):
        return well
    return prefix + well


def well_id(stem: str) -> str:
    """Inverse of :func:`las_stem`: the well id from a LAS file stem."""
    return stem.removeprefix(config().get("las_prefix", "") or "\0")


def las_path(data_dir: str | Path, well: str) -> Path:
    """LAS file of ``well`` under ``<data_dir>/wells`` (prefixed name first, then bare id)."""
    f = Path(data_dir) / "wells" / f"{las_stem(well)}.las"
    return f if f.exists() else Path(data_dir) / "wells" / f"{well}.las"


def expected() -> dict:
    """Values read from the real data that the data tests check (``test_expectations``)."""
    return config().get("test_expectations", {})
