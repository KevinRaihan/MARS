"""Well tops (markers) and zone intervals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Listed out of stratigraphic order in some marker files; the legacy code drops it.
LEGACY_DROPPED_MARKER = "Top LBR Lithostrat"


@dataclass(frozen=True)
class Interval:
    """A depth interval [top, base) in MD (ft)."""

    name: str
    top: float
    base: float

    def mask(self, depth: pd.Series) -> pd.Series:
        return (depth >= self.top) & (depth < self.base)


def read_marker_txt(path: str | Path) -> pd.DataFrame:
    """Read a Petrel marker export: tab-separated ``well, formation, MD``, no header."""
    raw = pd.read_csv(path, sep="\t", header=None, encoding="latin-1")
    if raw.shape[1] == 3:
        raw.columns = ["well", "formation", "md"]
    elif raw.shape[1] == 6:  # some exports also carry X, Y, TVDSS: well, fm, x, y, z, md
        raw.columns = ["well", "formation", "x", "y", "tvdss", "md"]
    else:
        raise ValueError(f"{path}: unexpected marker layout with {raw.shape[1]} columns")
    raw["md"] = pd.to_numeric(raw["md"], errors="coerce")
    return raw.dropna(subset=["formation", "md"]).reset_index(drop=True)


ZONES = {
    # name: (top marker, base marker). Robust to marker order; matched by exact formation name.
    "UBR": ("UBR Cycle 5 / Top UBR", "LBR Cycle 4 / BRF Shale"),
    "UBRn": ("UBR Cycle 5 / Top UBR", "LBR Cycle 4 / BRF Shale"),  # named (robust to faults)
    "BRF": ("LBR Cycle 4 / BRF Shale", "LBR Cycle 3"),
    "LBRc": ("LBR Cycle 3", "Talang Akar"),
    "LBR_all": ("LBR Cycle 4 / BRF Shale", "Talang Akar"),
}


def zone_interval(markers: pd.DataFrame, zone: str) -> Interval:
    """Zone between two named markers (see :data:`ZONES`); raises if a marker is missing."""
    top_name, base_name = ZONES[zone]
    md = markers.set_index("formation")["md"]
    if top_name not in md or base_name not in md:
        raise ValueError(f"markers for zone {zone!r} not found ({top_name!r} / {base_name!r})")
    top, base = float(md[top_name]), float(md[base_name])
    if base <= top:
        raise ValueError(f"zone {zone!r}: base {base} above top {top}")
    return Interval(zone, top, base)


def legacy_zone(markers: pd.DataFrame, zone: str, max_depth: float) -> Interval:
    """Zone interval exactly as the 2024 ``segment_log_data`` computed it.

    After dropping ``Top LBR Lithostrat``, each marker's base is the next marker's MD (or
    ``max_depth`` for the last one). The zone runs from the first marker whose name contains
    ``zone`` (e.g. "UBR", "LBR") to the base of the last such marker. Note that "LBR" therefore
    includes "LBR Cycle 4 / BRF Shale".
    """
    mk = markers[markers.formation != LEGACY_DROPPED_MARKER].reset_index(drop=True)
    top = base = None
    for i, row in mk.iterrows():
        nxt = mk.md.iloc[i + 1] if i + 1 < len(mk) else max_depth
        if zone in row.formation:
            if top is None:
                top = row.md
            base = nxt
    if top is None:
        raise ValueError(f"no marker containing {zone!r}")
    return Interval(zone, float(top), float(base))
