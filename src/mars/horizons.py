"""Time horizons (Petrel ``.xyt`` exports) gridded onto the cube IL/XL grid, and zone labels.

Petrel exports the horizons as scattered ``x y z col row`` nodes (``#`` header lines, z in ms,
possibly negative). The field's horizons sit on the seismic bin centres, so :func:`grid_horizon`
snaps nodes to the nearest (IL, XL) and only interpolates (linear Delaunay, no extrapolation)
where a node is missing inside the covered area. Grids are float arrays (n_il, n_xl), NaN
outside coverage.

Zone boundaries without a horizon are derived as an *isochron proportion* between two mapped
surfaces (:func:`proportional_surface`): the fraction f = (t - t_top) / (t_base - t_top) at the
wells is interpolated laterally by inverse-distance weighting and applied between the surfaces.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates

from mars.io.segy import xy_to_ilxl


def read_xyt(path: str | Path) -> pd.DataFrame:
    """Petrel scattered-data export -> DataFrame x, y, twt (positive ms)."""
    d = pd.read_csv(path, comment="#", sep=r"\s+", header=None, usecols=[0, 1, 2],
                    names=["x", "y", "twt"])
    d["twt"] = d["twt"].abs()
    return d.dropna().reset_index(drop=True)


def grid_horizon(df: pd.DataFrame, affine: dict, ilines, xlines, snap_tol: float = 0.15,
                 fill_holes: bool = True) -> tuple[np.ndarray, dict]:
    """Grid horizon nodes onto the (IL, XL) grid. Returns (grid, info).

    Nodes within ``snap_tol`` bins of a bin centre are snapped (duplicates averaged); the rest
    of the grid is filled by linear interpolation inside the convex hull of the snapped nodes
    when ``fill_holes`` (all nodes are interpolated if none snap).
    """
    ilines, xlines = np.asarray(ilines), np.asarray(xlines)
    fil, fxl = xy_to_ilxl(affine, df["x"].to_numpy(), df["y"].to_numpy())
    gi, gx = fil - ilines[0], fxl - xlines[0]
    ri, rx = np.round(gi).astype(int), np.round(gx).astype(int)
    on_node = (np.abs(gi - ri) <= snap_tol) & (np.abs(gx - rx) <= snap_tol)
    inside = (ri >= 0) & (ri < len(ilines)) & (rx >= 0) & (rx < len(xlines))
    use = on_node & inside
    shape = (len(ilines), len(xlines))
    s = np.zeros(shape)
    n = np.zeros(shape)
    np.add.at(s, (ri[use], rx[use]), df["twt"].to_numpy()[use])
    np.add.at(n, (ri[use], rx[use]), 1)
    with np.errstate(invalid="ignore"):
        grid = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    info = {"nodes": int(len(df)), "snapped": int(use.sum()),
            "max_snap_offset_bins": float(max(np.abs(gi - ri)[use].max(initial=0),
                                              np.abs(gx - rx)[use].max(initial=0)))}
    holes = np.isnan(grid)
    if fill_holes and holes.any():
        if use.any():
            pts, vals = np.argwhere(n > 0), grid[n > 0]
        else:
            ok = np.isfinite(gi) & np.isfinite(gx)
            pts, vals = np.column_stack([gi[ok], gx[ok]]), df["twt"].to_numpy()[ok]
        qi = np.argwhere(holes)
        filled = griddata(pts, vals, qi, method="linear")
        grid[qi[:, 0], qi[:, 1]] = filled
        info["filled"] = int(np.isfinite(filled).sum())
    info["coverage"] = float(np.isfinite(grid).mean())
    return grid, info


def sample_surface(grid: np.ndarray, affine: dict, ilines, xlines, x, y) -> np.ndarray:
    """Bilinear surface value at map coordinates (NaN if any neighbour is NaN or outside)."""
    fil, fxl = xy_to_ilxl(affine, np.atleast_1d(x), np.atleast_1d(y))
    coords = np.vstack([fil - ilines[0], fxl - xlines[0]])
    g = np.nan_to_num(grid, nan=0.0)
    v = map_coordinates(g, coords, order=1, mode="constant", cval=np.nan)
    nan_mask = map_coordinates(np.isnan(grid).astype(float), coords, order=1, mode="constant",
                               cval=1.0)
    v[nan_mask > 0] = np.nan
    return v


def idw(xy_known: np.ndarray, values: np.ndarray, xy_query: np.ndarray, power: float = 2.0,
        eps: float = 1e-6) -> np.ndarray:
    """Inverse-distance-weighted interpolation (exact at the known points)."""
    xy_known, values = np.asarray(xy_known, float), np.asarray(values, float)
    ok = np.isfinite(values) & np.isfinite(xy_known).all(1)
    xy_known, values = xy_known[ok], values[ok]
    q = np.asarray(xy_query, float)
    out = np.empty(len(q))
    for s in range(0, len(q), 200_000):
        d = np.hypot(q[s:s + 200_000, None, 0] - xy_known[None, :, 0],
                     q[s:s + 200_000, None, 1] - xy_known[None, :, 1])
        w = 1.0 / np.maximum(d, eps) ** power
        out[s:s + 200_000] = (w * values).sum(1) / w.sum(1)
    return out


def proportional_surface(top: np.ndarray, base: np.ndarray, fraction: np.ndarray) -> np.ndarray:
    """Surface at ``fraction`` of the isochron between ``top`` and ``base``."""
    return top + np.clip(fraction, 0.0, 1.0) * (base - top)


def zone_labels(twt: np.ndarray, boundaries: list[np.ndarray]) -> np.ndarray:
    """Zone index per sample: k where boundaries[k-1] <= t < boundaries[k] (1-based), 0 outside.

    ``twt`` is the time axis (nt,); ``boundaries`` are K+1 surfaces of shape (n_il, n_xl),
    top to base, for K zones. Returns int8 (n_il, n_xl, nt). Crossing surfaces are honoured in
    order (a sample is assigned to the first zone whose interval contains it).
    """
    t = np.asarray(twt, float)[None, None, :]
    lab = np.zeros(boundaries[0].shape + (t.shape[-1],), dtype=np.int8)
    for k in range(len(boundaries) - 1):
        top, base = boundaries[k][..., None], boundaries[k + 1][..., None]
        with np.errstate(invalid="ignore"):
            m = (t >= top) & (t < base) & (lab == 0)
        lab[m] = k + 1
    return lab


def well_marker_times(data_dir: str | Path, wells, markers, welltops=None) -> pd.DataFrame:
    """Marker TWT (2 x LAS one-way time at the marker MD) and X/Y at each well.

    Returns columns well, marker, md, twt, x, y (rows only for markers the well has and whose
    MD is covered by the time curve).
    """
    from mars.io.las import read_las
    from mars.io.tops import read_marker_txt
    from mars.tie import las_path, read_welltops, well_path_tops, well_time_curve, xy_along_md

    if welltops is None:
        welltops = read_welltops(Path(data_dir) / "tops" / "welltops.csv")
    rows = []
    for w in wells:
        las = read_las(las_path(data_dir, w))
        tc = well_time_curve(las, w)
        ok = las[tc].notna().to_numpy()
        tops = well_path_tops(data_dir, w, welltops)
        mk = read_marker_txt(Path(data_dir) / "tops" / f"{w}_Marker.txt")
        for m in markers:
            hit = mk.loc[mk.formation == m, "md"]
            if hit.empty:
                continue
            md = float(hit.iloc[0])
            owt = np.interp(md, las["DEPT"].to_numpy()[ok], las[tc].to_numpy()[ok],
                            left=np.nan, right=np.nan)
            x, y = xy_along_md(tops, [md])
            rows.append({"well": w, "marker": m, "md": md, "twt": 2 * owt, "x": x[0], "y": y[0]})
    return pd.DataFrame(rows).dropna(subset=["twt"]).reset_index(drop=True)
