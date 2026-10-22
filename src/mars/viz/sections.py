"""Vertical sections through the cubes along arbitrary polylines (e.g. through wells).

Replaces the Petrel "arbitrary line" figures of the thesis (Gambar 4.15-4.23). Cubes are the
xarray/Zarr stores produced by :mod:`mars.io.segy` (dims ``il, xl, twt_ms``; geometry affine in
``attrs["affine_ilxl_to_xy"]``). Well labels must be codenames (:func:`mars.naming.codename`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from mars.io.segy import xy_to_ilxl


@dataclass
class Section:
    dist_m: np.ndarray            # (n,) distance along the polyline
    twt_ms: np.ndarray            # (nt,)
    values: np.ndarray            # (n, nt)
    il: np.ndarray                # (n,) fractional inline of each column
    xl: np.ndarray                # (n,)
    x: np.ndarray
    y: np.ndarray
    vertices_dist_m: np.ndarray   # distance of each polyline vertex (e.g. well positions)
    labels: list[str] = field(default_factory=list)


def polyline_points(x_vertices, y_vertices, spacing_m: float = 12.5):
    """Densify a polyline (points every ``spacing_m`` plus every vertex).

    Returns x, y, dist, vertex_dist.
    """
    xv, yv = np.asarray(x_vertices, float), np.asarray(y_vertices, float)
    seg = np.hypot(np.diff(xv), np.diff(yv))
    vdist = np.r_[0.0, np.cumsum(seg)]
    dist = np.unique(np.r_[np.arange(0.0, vdist[-1], spacing_m), vdist])
    return np.interp(dist, vdist, xv), np.interp(dist, vdist, yv), dist, vdist


def sample_section(da: xr.DataArray, x_vertices, y_vertices, labels=(), spacing_m: float = 12.5,
                   twt_range: tuple[float, float] | None = None) -> Section:
    """Bilinear (IL/XL) sampling of ``da`` along the polyline through the given XY vertices."""
    affine = da.attrs.get("affine_ilxl_to_xy") or da.to_dataset().attrs.get("affine_ilxl_to_xy")
    x, y, dist, vdist = polyline_points(x_vertices, y_vertices, spacing_m)
    il, xl = xy_to_ilxl(affine, x, y)
    il_axis, xl_axis = da["il"].to_numpy(), da["xl"].to_numpy()
    if twt_range is not None:
        da = da.sel(twt_ms=slice(*twt_range))
    t = da["twt_ms"].to_numpy()
    fi = np.clip(il - il_axis[0], 0, len(il_axis) - 1.000001)
    fx = np.clip(xl - xl_axis[0], 0, len(xl_axis) - 1.000001)
    i0, x0 = np.floor(fi).astype(int), np.floor(fx).astype(int)
    wi, wx = fi - i0, fx - x0
    # read the needed traces once (unique corners), then combine
    corners = {(a, b) for a, b in zip(np.r_[i0, i0 + 1, i0, i0 + 1], np.r_[x0, x0, x0 + 1, x0 + 1],
                                      strict=True)}
    keys = sorted(corners)
    ii = xr.DataArray([k[0] for k in keys], dims="pt")
    xx = xr.DataArray([k[1] for k in keys], dims="pt")
    traces = da.isel(il=ii, xl=xx).transpose("pt", "twt_ms").to_numpy()
    lut = {k: n for n, k in enumerate(keys)}

    def tr(a, b):
        return traces[[lut[(p, q)] for p, q in zip(a, b, strict=True)]]

    a, b = wi[:, None], wx[:, None]
    v = ((1 - a) * (1 - b) * tr(i0, x0) + a * (1 - b) * tr(i0 + 1, x0)
         + (1 - a) * b * tr(i0, x0 + 1) + a * b * tr(i0 + 1, x0 + 1))
    outside = (il < il_axis[0]) | (il > il_axis[-1]) | (xl < xl_axis[0]) | (xl > xl_axis[-1])
    v[outside] = np.nan
    return Section(dist, t, v, il, xl, x, y, vdist, list(labels))


def project_to_polyline(section: Section, x, y) -> tuple[np.ndarray, np.ndarray]:
    """Distance along the section and perpendicular offset (m) of points ``x, y`` (e.g. a deviated
    well path), using the nearest densified section column."""
    x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
    d2 = (x[:, None] - section.x[None, :]) ** 2 + (y[:, None] - section.y[None, :]) ** 2
    k = np.argmin(d2, axis=1)
    return section.dist_m[k], np.sqrt(d2[np.arange(len(x)), k])


def surface_along(section: Section, surface: xr.DataArray) -> np.ndarray:
    """Values of a 2-D (il, xl) surface (e.g. horizon TWT) along the section columns (bilinear)."""
    return surface.interp(il=xr.DataArray(section.il, dims="p"),
                          xl=xr.DataArray(section.xl, dims="p")).to_numpy()


def plot_section(ax, section: Section, cmap: str = "viridis", vmin=None, vmax=None,
                 horizons: dict[str, np.ndarray] | None = None, well_logs: list[dict] | None = None,
                 title: str = "", cbar_label: str = ""):
    """Draw a section; ``horizons`` = {name: twt per column}; ``well_logs`` = list of dicts with
    keys dist_m, twt_ms, values, label (codename) - drawn as colour-filled strips on top.
    ``dist_m`` may be a scalar (vertical strip) or one distance per sample (a deviated path,
    see :func:`project_to_polyline`)."""
    import matplotlib.pyplot as plt

    m = ax.pcolormesh(section.dist_m, section.twt_ms, section.values.T, cmap=cmap, vmin=vmin,
                      vmax=vmax, shading="nearest")
    ax.set_facecolor("0.9")
    t_lo, t_hi = section.twt_ms.min(), section.twt_ms.max()
    for name, twt in (horizons or {}).items():
        ax.plot(section.dist_m, twt, lw=1.2, color="k")
        ok = np.flatnonzero(np.isfinite(twt) & (twt > t_lo) & (twt < t_hi))
        if ok.size:  # label at the first column where the horizon is inside the window
            ax.text(section.dist_m[ok[0]], twt[ok[0]], f"{name} ", fontsize=7, va="center",
                    ha="right", clip_on=True)
    width = max(section.dist_m[-1] * 0.02, 40.0)
    for w in well_logs or []:
        t = np.asarray(w["twt_ms"], float)
        val = np.asarray(w["values"], float)
        dist = np.broadcast_to(np.asarray(w["dist_m"], float), t.shape)
        ok = np.isfinite(t) & np.isfinite(val) & np.isfinite(dist)
        t, val, dist = t[ok], val[ok], dist[ok]
        if t.size < 2:
            continue
        order = np.argsort(t)
        t, val, dist = t[order], val[order], dist[order]
        if np.ptp(dist) > width / 2:  # deviated well: draw the log along its projected path
            from matplotlib.collections import LineCollection

            pts = np.c_[dist, t][:, None, :]
            lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=cmap,
                                linewidths=6)
            lc.set_array((val[:-1] + val[1:]) / 2)
            lc.set_clim(vmin, vmax)
            lc.set_capstyle("round")
            ax.plot(dist, t, color="k", lw=8, solid_capstyle="butt", zorder=2)
            lc.set_zorder(3)
            ax.add_collection(lc)
            ax.text(dist[0], t[0] - 3, w["label"], ha="center", va="bottom", fontsize=8,
                    weight="bold")
            continue
        d0 = float(np.mean(dist))
        edges = np.r_[t[0], (t[1:] + t[:-1]) / 2, t[-1]]
        ax.pcolormesh([d0 - width / 2, d0 + width / 2], edges, val[:, None], cmap=cmap,
                      vmin=vmin, vmax=vmax, shading="flat")
        ax.add_patch(plt.Rectangle((d0 - width / 2, t[0]), width, t[-1] - t[0], fill=False,
                                   lw=1.0, ec="k"))
        ax.text(d0, t[0] - 3, w["label"], ha="center", va="bottom", fontsize=8, weight="bold")
    for d in section.vertices_dist_m:
        ax.axvline(d, color="w", lw=0.5, ls=":")
    ax.set_ylim(section.twt_ms.max(), section.twt_ms.min())
    ax.set_xlabel("distance along line (m)")
    ax.set_ylabel("TWT (ms)")
    ax.set_title(title)
    cb = plt.colorbar(m, ax=ax, pad=0.01)
    cb.set_label(cbar_label)
    return m
