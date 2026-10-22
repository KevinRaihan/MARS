"""Well-to-cube tie: well trajectory in (X, Y, TWT) and cube extraction along it.

The trajectory comes from the marker rows of ``welltops.csv`` (X/Y in m, MD in ft): X and Y are
interpolated linearly along MD between markers (linear extrapolation beyond the end markers),
which is adequate over the ~1000 ft Baturaja interval. TWT is twice the LAS one-way-time curve
(vault 05-Literature/ggrp-field-study.md section 3). Cube values are sampled bilinearly in
(IL, XL) via the cube affine and linearly in TWT; a sample is NaN if any of its 8 neighbours is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mars.dataset import las_path as _las_path
from mars.dataset import las_stem
from mars.io.segy import xy_to_ilxl


@dataclass
class Trajectory:
    """Samples along a well path: MD (ft), X/Y (m), TWT (ms)."""

    md: np.ndarray
    x: np.ndarray
    y: np.ndarray
    twt: np.ndarray


def _interp_extrap(xq, xp, fp) -> np.ndarray:
    xq, xp, fp = (np.asarray(v, dtype=float) for v in (xq, xp, fp))
    if len(xp) == 1:
        return np.full_like(xq, fp[0])
    out = np.interp(xq, xp, fp)
    lo, hi = xq < xp[0], xq > xp[-1]
    out[lo] = fp[0] + (xq[lo] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
    out[hi] = fp[-1] + (xq[hi] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return out


def read_welltops(path: str | Path) -> pd.DataFrame:
    """``welltops.csv`` (``;``-separated) with columns well, surface, x, y, z, md."""
    df = pd.read_csv(path, sep=";", encoding="latin-1")
    df.columns = ["well", "surface", "x", "y", "z", "md"]
    for c in ("x", "y", "z", "md"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["x", "y", "md"])


def select_well_tops(
    tops: pd.DataFrame, well: str, marker_md: pd.Series | None = None
) -> pd.DataFrame:
    """Rows of ``tops`` for ``well``.

    The identifier is :func:`mars.dataset.las_stem` of the well or a decorated variant such as
    ``<stem> (7st)``.
    With several candidates, ``marker_md`` (the well's marker-file MDs) picks the one whose MDs
    match best.
    """
    name = las_stem(well)
    cand = [w for w in tops["well"].unique() if w == name or w.startswith(name + " ")]
    if not cand:
        raise KeyError(f"{well}: not in welltops")
    if len(cand) > 1 and marker_md is not None:
        def misfit(w):
            md = tops.loc[tops.well == w, "md"].to_numpy()
            return np.median([np.abs(md - m).min() for m in marker_md])
        cand = sorted(cand, key=misfit)
    elif len(cand) > 1:
        cand = [name] if name in cand else cand
    return tops[tops.well == cand[0]].sort_values("md").reset_index(drop=True)


def las_path(data_dir: str | Path, well: str) -> Path:
    """LAS file of ``well`` (see :func:`mars.dataset.las_path`)."""
    return _las_path(data_dir, well)


def well_path_tops(data_dir: str | Path, well: str, welltops: pd.DataFrame | None = None):
    """Marker rows with md, x, y for ``well``: from its 6-column marker file when it has
    coordinates, else from ``welltops.csv`` (matched by marker MD)."""
    from mars.io.tops import read_marker_txt

    mk = read_marker_txt(Path(data_dir) / "tops" / f"{well}_Marker.txt")
    if {"x", "y"} <= set(mk.columns):
        t = mk[["formation", "x", "y", "md"]].rename(columns={"formation": "surface"})
        return t.dropna().sort_values("md").reset_index(drop=True)
    if welltops is None:
        welltops = read_welltops(Path(data_dir) / "tops" / "welltops.csv")
    return select_well_tops(welltops, well, mk["md"])


def well_time_curve(las: pd.DataFrame, well: str) -> str:
    """Name of the LAS one-way-time curve (the upper-cased LAS stem of the well)."""
    stem = las_stem(well).upper()
    if stem not in las.columns:
        raise KeyError(f"{well}: time curve {stem} not in LAS")
    return stem


def xy_along_md(well_tops: pd.DataFrame, md) -> tuple[np.ndarray, np.ndarray]:
    """X/Y at ``md`` by linear interpolation (and extrapolation) between marker rows."""
    t = well_tops.sort_values("md").drop_duplicates("md")
    return _interp_extrap(md, t["md"], t["x"]), _interp_extrap(md, t["md"], t["y"])


def trajectory(well_tops: pd.DataFrame, md, las_md, las_owt_ms) -> Trajectory:
    """Trajectory at ``md``: X/Y from the markers, TWT = 2 x the LAS one-way time."""
    md = np.asarray(md, dtype=float)
    ok = np.isfinite(np.asarray(las_owt_ms, dtype=float))
    owt = np.interp(md, np.asarray(las_md, float)[ok], np.asarray(las_owt_ms, float)[ok],
                    left=np.nan, right=np.nan)
    x, y = xy_along_md(well_tops, md)
    return Trajectory(md=md, x=x, y=y, twt=2.0 * owt)


def sample_cube(values: np.ndarray, il0: float, xl0: float, t0: float, dt: float,
                fil, fxl, t, method: str = "bilinear") -> np.ndarray:
    """Sample ``values[il, xl, t]`` (regular grid, unit IL/XL steps from ``il0``/``xl0``,
    time from ``t0`` with step ``dt``) at fractional (IL, XL) and time ``t``.

    ``method`` is "bilinear" or "nearest" laterally; time is always linear.
    Out-of-grid or NaN-touching samples are NaN.
    """
    fil, fxl, t = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (fil, fxl, t))
    ni, nx, nt = values.shape
    gi, gx, gt = fil - il0, fxl - xl0, (t - t0) / dt
    if method == "nearest":
        gi, gx = np.round(gi), np.round(gx)
    elif method != "bilinear":
        raise ValueError(method)
    i0, x0, k0 = np.floor(gi).astype(int), np.floor(gx).astype(int), np.floor(gt).astype(int)
    wi, wx, wt = gi - i0, gx - x0, gt - k0
    out = np.zeros(len(t))
    bad = ~(np.isfinite(gi) & np.isfinite(gx) & np.isfinite(gt))
    for di in (0, 1):
        for dx in (0, 1):
            for dk in (0, 1):
                ii, xx, kk = i0 + di, x0 + dx, k0 + dk
                w = (wi if di else 1 - wi) * (wx if dx else 1 - wx) * (wt if dk else 1 - wt)
                inside = (ii >= 0) & (ii < ni) & (xx >= 0) & (xx < nx) & (kk >= 0) & (kk < nt)
                need = w > 0
                bad |= need & ~inside
                v = np.where(inside, values[np.clip(ii, 0, ni - 1), np.clip(xx, 0, nx - 1),
                                            np.clip(kk, 0, nt - 1)], 0.0)
                bad |= need & ~np.isfinite(v)
                out += np.where(need, w * np.nan_to_num(v), 0.0)
    out[bad] = np.nan
    return out


class CubeWindow:
    """A small in-memory sub-cube around a trajectory, for fast repeated sampling."""

    def __init__(self, da, affine: dict, traj: Trajectory, margin_traces: int = 3,
                 margin_ms: float = 40.0):
        fil, fxl = xy_to_ilxl(affine, traj.x, traj.y)
        il, xl, tw = da["il"].values, da["xl"].values, da["twt_ms"].values
        ok = np.isfinite(traj.twt)
        i_lo = max(0, int(np.floor(fil.min() - il[0])) - margin_traces)
        i_hi = min(len(il), int(np.ceil(fil.max() - il[0])) + margin_traces + 1)
        x_lo = max(0, int(np.floor(fxl.min() - xl[0])) - margin_traces)
        x_hi = min(len(xl), int(np.ceil(fxl.max() - xl[0])) + margin_traces + 1)
        dt = float(tw[1] - tw[0])
        k_lo = max(0, int(np.floor((traj.twt[ok].min() - margin_ms - tw[0]) / dt)))
        k_hi = min(len(tw), int(np.ceil((traj.twt[ok].max() + margin_ms - tw[0]) / dt)) + 1)
        if i_lo >= i_hi or x_lo >= x_hi:
            raise ValueError("trajectory outside the cube")
        self.values = da.isel(il=slice(i_lo, i_hi), xl=slice(x_lo, x_hi),
                              twt_ms=slice(k_lo, k_hi)).values.astype(float)
        self.il0, self.xl0 = float(il[i_lo]), float(xl[x_lo])
        self.t0, self.dt = float(tw[k_lo]), dt
        self.fil, self.fxl, self.traj = fil, fxl, traj

    def sample(self, shift_ms: float = 0.0, method: str = "bilinear", dil: float = 0.0,
               dxl: float = 0.0) -> np.ndarray:
        """Cube along the path shifted by ``shift_ms`` in time and (dil, dxl) laterally."""
        return sample_cube(self.values, self.il0, self.xl0, self.t0, self.dt,
                           self.fil + dil, self.fxl + dxl, self.traj.twt + shift_ms, method)

    def neighbourhood(self, radius: int = 2, t_range: tuple[float, float] | None = None):
        """Finite cube values within ``radius`` traces of the path (and inside ``t_range``)."""
        ii = np.round(self.fil - self.il0).astype(int)
        xx = np.round(self.fxl - self.xl0).astype(int)
        mask = np.zeros(self.values.shape[:2], bool)
        for a, b in zip(ii, xx, strict=True):
            mask[max(0, a - radius): a + radius + 1, max(0, b - radius): b + radius + 1] = True
        v = self.values[mask]
        if t_range is not None:
            t = self.t0 + self.dt * np.arange(self.values.shape[2])
            v = v[:, (t >= t_range[0]) & (t <= t_range[1])]
        return v[np.isfinite(v)]


def offset_ilxl(affine: dict, dx_m, dy_m) -> tuple[np.ndarray, np.ndarray]:
    """IL/XL increments of a map displacement (dx, dy) in metres (linear part of the affine)."""
    b = np.asarray(affine["A_inv"])
    dx_m, dy_m = np.asarray(dx_m, float), np.asarray(dy_m, float)
    return b[0, 0] * dx_m + b[0, 1] * dy_m, b[1, 0] * dx_m + b[1, 1] * dy_m


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def best_shift(log_values, sampler, max_ms: float = 20.0, step_ms: float = 1.0):
    """Bulk time shift in [-max_ms, max_ms] maximising r(log, sampler(shift)).

    Returns ``(shift_ms, r_at_shift, shifts, r_curve)``.
    """
    shifts = np.arange(-max_ms, max_ms + step_ms / 2, step_ms)
    r = np.array([pearson(log_values, sampler(s)) for s in shifts])
    if np.isnan(r).all():
        return 0.0, float("nan"), shifts, r
    j = int(np.nanargmax(r))
    return float(shifts[j]), float(r[j]), shifts, r


def resample_to_time(twt, values, dt: float = 1.0):
    """Average depth samples into regular TWT bins of width ``dt``.

    The bins depend on the finite ``twt`` only, so curves along the same path share them;
    bins whose samples are all NaN in ``values`` are NaN. Returns (bin centres, means).
    """
    twt, values = np.asarray(twt, float), np.asarray(values, float)
    has_t = np.isfinite(twt)
    if not has_t.any():
        return np.array([]), np.array([])
    edges = np.arange(np.floor(twt[has_t].min() / dt) * dt, twt[has_t].max() + dt, dt)
    idx = np.digitize(twt[has_t], edges) - 1
    v = values[has_t]
    ok = np.isfinite(v)
    s = np.bincount(idx[ok], v[ok], minlength=len(edges))
    n = np.bincount(idx[ok], minlength=len(edges))
    occupied = np.bincount(idx, minlength=len(edges)) > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / n, np.nan)
    return (edges + dt / 2)[occupied], mean[occupied]
