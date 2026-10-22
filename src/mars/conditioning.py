"""Well-log conditioning for the corrected (non-legacy) pipeline.

Fixes legacy defect B11: rows are dropped only when a curve that is actually needed (Vp, Vs,
density, target) is missing, not when *any* of the ~20 exported curves is null. Padded constant
runs in the shear log (Petrel fills short gaps with a constant, e.g. at the top of RONALDO) are
removed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mars.attributes import elastic_attributes
from mars.dataset import las_path
from mars.io.las import read_las
from mars.io.tops import legacy_zone, read_marker_txt, zone_interval

TARGETS = ("PHIE", "VSH", "SWT")


def constant_run_mask(x: pd.Series, min_run: int = 5) -> np.ndarray:
    """True where ``x`` sits in a run of >= ``min_run`` identical consecutive values."""
    v = x.to_numpy()
    if len(v) == 0:
        return np.zeros(0, bool)
    change = np.r_[True, v[1:] != v[:-1]]
    run_id = np.cumsum(change)
    run_len = np.bincount(run_id)[run_id]
    return (run_len >= min_run) & ~np.isnan(v.astype(float))


def load_well_zone(
    data_dir: str | Path, well: str, zone: str, min_const_run: int = 3, require_vs: bool = True
) -> pd.DataFrame:
    """One well's samples inside ``zone`` ("UBR"/"LBR", legacy marker definition).

    Returns columns DEPT, VP, VS, RHOB, PHIE, VSH, SWT, WELL plus the 11 base elastic attributes.
    Rows lacking Vp, Vs or density are dropped; targets may still contain NaN (drop per target).
    With ``require_vs=False`` (Vs-free work) only Vp and density are required; shear-based
    attributes are then NaN where Vs is missing.
    """
    data_dir = Path(data_dir)
    df = read_las(las_path(data_dir, well), canonical=True)
    markers = read_marker_txt(data_dir / "tops" / f"{well}_Marker.txt")
    if zone in ("UBR", "LBR"):  # legacy definitions (thesis-compatible)
        iv = legacy_zone(markers, zone, max_depth=float(df["DEPT"].max()))
    else:
        iv = zone_interval(markers, zone)
    z = df[iv.mask(df["DEPT"])].copy()
    keep = [c for c in ("DEPT", "VP", "VS", "RHOB", *TARGETS) if c in z.columns]
    z = z[keep]
    if "VS" not in z:
        z["VS"] = np.nan
    z = z.dropna(subset=["VP", "VS", "RHOB"] if require_vs else ["VP", "RHOB"])
    has_vs = z["VS"].notna()
    padded = constant_run_mask(z["VS"], min_const_run) | constant_run_mask(z["VP"], min_const_run)
    vs_ok = ~has_vs | ((z["VP"] > z["VS"]) & (z["VS"] > 0))
    z = z[~padded & vs_ok & (z["RHOB"] > 0) & (z["VP"] > 0)]  # physical sanity
    base = elastic_attributes(z["VP"], z["VS"], z["RHOB"])
    out = pd.concat([z.reset_index(drop=True), base.reset_index(drop=True)], axis=1)
    out["WELL"] = well
    return out


def _nan_moving_mean(v: np.ndarray, w: int, min_frac: float) -> np.ndarray:
    # centred boxcar via cumulative sums: output length always equals len(v), even if w > len(v)
    ok = np.isfinite(v)
    half_lo, half_hi = (w - 1) // 2, w // 2

    def window_sum(x: np.ndarray) -> np.ndarray:
        c = np.r_[0.0, np.cumsum(x)]
        i = np.arange(len(x))
        lo = np.clip(i - half_lo, 0, len(x))
        hi = np.clip(i + half_hi + 1, 0, len(x))
        return c[hi] - c[lo]

    s = window_sum(np.where(ok, v, 0.0))
    n = window_sum(ok.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / n
    out[n < min_frac * min(w, len(v))] = np.nan  # thin zones: scale the requirement
    return out


def upscale_well(z: pd.DataFrame, length_ft: float, step_ft: float = 0.5,
                 min_frac: float = 0.5) -> pd.DataFrame:
    """Upscale one well's zone to a vertical resolution of ``length_ft`` (boxcar window).

    Elastic properties use Backus-style averaging: density arithmetic, P- and S-wave moduli
    (rho*V^2) harmonic, then V = sqrt(M_avg / rho_avg). Petrophysical targets are averaged
    arithmetically. Works on a regular depth grid so that gaps (dropped rows) are not bridged;
    output samples with < ``min_frac`` valid input in their window are dropped. The base
    elastic attributes are recomputed from the upscaled Vp, Vs, rho.
    """
    if z.empty:
        return z
    grid = np.round(np.arange(z["DEPT"].min(), z["DEPT"].max() + step_ft / 2, step_ft), 4)
    r = z.set_index(np.round(z["DEPT"], 4)).reindex(grid)
    w = max(1, int(round(length_ft / step_ft)))
    rho = _nan_moving_mean(r["RHOB"].to_numpy(float), w, min_frac)
    out = pd.DataFrame({"DEPT": grid, "RHOB": rho})
    for v in ("VP", "VS"):
        mod = r["RHOB"].to_numpy(float) * r[v].to_numpy(float) ** 2
        inv = _nan_moving_mean(1.0 / mod, w, min_frac)
        out[v] = np.sqrt((1.0 / inv) / rho)
    for t in TARGETS:
        if t in r:
            out[t] = _nan_moving_mean(r[t].to_numpy(float), w, min_frac)
    # keep only grid points that were real samples (no interpolation into gaps)
    out = out[r["VP"].notna().to_numpy()].dropna(subset=["VP", "RHOB"])  # Vs may be absent
    base = elastic_attributes(out["VP"], out["VS"], out["RHOB"])
    out = pd.concat([out.reset_index(drop=True), base.reset_index(drop=True)], axis=1)
    out["WELL"] = z["WELL"].iloc[0]
    return out


def well_group(well: str) -> str:
    """CV group: sidetracks share their parent's group (W-05st -> W-05, W-12p -> W-12)."""
    w = well.split("st")[0] if "st" in well else well
    return w.rstrip("p")
