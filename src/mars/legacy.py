"""Faithful reconstruction of the 2024 data preparation (MARS_COMBINED.ipynb).

Kept separate from the corrected pipeline so that the parity test pins down exactly what the
thesis did. Defects reproduced on purpose: row-wise ``dropna`` over all curves (B11) and fitting
on raw, non-upscaled logs (B6).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mars.attributes import (
    LEGACY_EXCLUDE_THESIS,
    TRANSFORMS_THESIS,
    Transform,
    elastic_attributes,
    transform_library,
)
from mars.dataset import las_path
from mars.dataset import wells as well_set
from mars.io.las import read_las
from mars.io.tops import legacy_zone, read_marker_txt

THESIS_WELLS = well_set("thesis_pair")  # RONALDO, MESSI (order matters: stacked in order)


@dataclass
class LegacyZoneData:
    X: pd.DataFrame        # transformed attribute library, stacked over wells
    targets: pd.DataFrame  # PHIE, VSH, SWT (+ DEPT, WELL)
    n_per_well: dict[str, int]


def load_legacy_zone(
    data_dir: str | Path,
    zone: str,
    wells: Sequence[str] = THESIS_WELLS,
    transforms: Sequence[Transform] = TRANSFORMS_THESIS,
    exclude: Sequence[str] = LEGACY_EXCLUDE_THESIS,
) -> LegacyZoneData:
    data_dir = Path(data_dir)
    libs, tgts, counts = [], [], {}
    for w in wells:
        df = read_las(las_path(data_dir, w)).dropna()  # B11, as in 2024
        markers = read_marker_txt(data_dir / "tops" / f"{w}_Marker.txt")
        iv = legacy_zone(markers, zone, max_depth=df["DEPT"].max())
        z = df[iv.mask(df["DEPT"])]
        base = elastic_attributes(z["P-VELOCITY"], z["S-VELOCITY_FINAL"], z["RHOB"])
        libs.append(transform_library(base, transforms, exclude))
        t = z[["DEPT", "PHIE", "VSH", "SWT"]].copy()
        t["WELL"] = w
        tgts.append(t)
        counts[w] = len(z)
    return LegacyZoneData(
        X=pd.concat(libs, ignore_index=True),
        targets=pd.concat(tgts, ignore_index=True),
        n_per_well=counts,
    )
