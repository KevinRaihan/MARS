"""Interval property maps and structure maps (replaces thesis Gambar 4.1-4.2, 4.18, 4.21, 4.24).

Property cubes are (il, xl, twt_ms) DataArrays; surfaces are (il, xl) DataArrays of TWT (ms).
Maps are drawn in projected X/Y (cube coords ``cdp_x``/``cdp_y``) with codename well posts.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def interval_stat(prop: xr.DataArray, top: xr.DataArray, base: xr.DataArray, stat: str = "mean",
                  chunk_il: int = 32) -> xr.DataArray:
    """Statistic of ``prop`` between ``top`` and ``base`` surfaces (TWT ms), per trace.

    ``stat``: "mean", "p10", "p50", "p90", "max", or "count" (number of finite samples).
    Processed in inline chunks so a full cube never sits in memory.
    """
    t = prop["twt_ms"].to_numpy()
    out = np.full((prop.sizes["il"], prop.sizes["xl"]), np.nan, dtype=np.float32)
    for s in range(0, prop.sizes["il"], chunk_il):
        sl = slice(s, s + chunk_il)
        v = prop.isel(il=sl).to_numpy().astype(np.float64)
        top_c = top.isel(il=sl).to_numpy()[..., None]
        base_c = base.isel(il=sl).to_numpy()[..., None]
        inside = (t[None, None, :] >= top_c) & (t[None, None, :] < base_c)
        v = np.where(inside, v, np.nan)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if stat == "mean":
                    r = np.nanmean(v, axis=2)
                elif stat == "max":
                    r = np.nanmax(v, axis=2)
                elif stat == "count":
                    r = np.isfinite(v).sum(axis=2).astype(float)
                elif stat in ("p10", "p50", "p90"):
                    r = np.nanpercentile(v, float(stat[1:]), axis=2)
                else:
                    raise ValueError(stat)
        out[sl] = r
    return xr.DataArray(out, coords={"il": prop["il"], "xl": prop["xl"]}, dims=("il", "xl"),
                        name=f"{prop.name}_{stat}")


def net_thickness_ms(masks: list[xr.DataArray], top: xr.DataArray, base: xr.DataArray,
                     dt_ms: float = 1.0, chunk_il: int = 32) -> xr.DataArray:
    """Net-reservoir thickness (ms TWT): samples between surfaces where all boolean ``masks`` hold
    (e.g. predicted VSH <= cutoff, PHIE >= cutoff, SWT <= cutoff, the GGRP net criteria)."""
    t = masks[0]["twt_ms"].to_numpy()
    out = np.zeros((masks[0].sizes["il"], masks[0].sizes["xl"]), dtype=np.float32)
    for s in range(0, masks[0].sizes["il"], chunk_il):
        sl = slice(s, s + chunk_il)
        ok = np.ones(masks[0].isel(il=sl).shape, bool)
        for m in masks:
            ok &= m.isel(il=sl).to_numpy().astype(bool)
        inside = (t[None, None, :] >= top.isel(il=sl).to_numpy()[..., None]) & \
                 (t[None, None, :] < base.isel(il=sl).to_numpy()[..., None])
        out[sl] = (ok & inside).sum(axis=2) * dt_ms
    return xr.DataArray(out, coords={"il": masks[0]["il"], "xl": masks[0]["xl"]}, dims=("il", "xl"),
                        name="net_ms")


def plot_map(ax, field: xr.DataArray, cdp_x: xr.DataArray, cdp_y: xr.DataArray, wells=None,
             cmap: str = "viridis", vmin=None, vmax=None, title: str = "", cbar_label: str = "",
             contours: int | None = None, km: bool = True):
    """Map in projected coordinates. ``wells`` = list of dicts {x, y, label (codename), value?}.

    With ``km=True`` coordinates are shown relative to the survey's SW corner in km (avoids
    printing absolute UTM coordinates on shared figures).
    """
    import matplotlib.pyplot as plt

    X, Y = cdp_x.to_numpy(), cdp_y.to_numpy()
    x0, y0 = np.nanmin(X), np.nanmin(Y)
    sx = (X - x0) / 1000 if km else X
    sy = (Y - y0) / 1000 if km else Y
    ax.set_facecolor("0.82")  # no data (outside survey, zero thickness) shows grey
    m = ax.pcolormesh(sx, sy, field.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
    if contours:
        ax.contour(sx, sy, field.to_numpy(), levels=contours, colors="k", linewidths=0.4)
    for w in wells or []:
        wx = (w["x"] - x0) / 1000 if km else w["x"]
        wy = (w["y"] - y0) / 1000 if km else w["y"]
        ax.plot(wx, wy, "o", ms=4, mfc="w", mec="k")
        lab = w["label"] + (f"\n{w['value']:.2f}" if w.get("value") is not None else "")
        ax.annotate(lab, (wx, wy), xytext=(3, 3), textcoords="offset points", fontsize=6)
    ax.set_aspect("equal")
    ax.set_xlabel("x (km, relative)" if km else "x (m)")
    ax.set_ylabel("y (km, relative)" if km else "y (m)")
    ax.set_title(title)
    cb = plt.colorbar(m, ax=ax, shrink=0.8, pad=0.01)
    cb.set_label(cbar_label)
    return m


def interval_sum(prop: xr.DataArray, top: xr.DataArray, base: xr.DataArray,
                 masks: list[xr.DataArray] | None = None, dt_ms: float = 1.0,
                 chunk_il: int = 32) -> xr.DataArray:
    """Sum of ``prop`` x ``dt_ms`` between surfaces where all boolean ``masks`` hold (NaN = 0),
    e.g. hydrocarbon pore thickness in ms with ``prop = PHIE * (1 - SWT)`` and the net masks."""
    t = prop["twt_ms"].to_numpy()
    out = np.zeros((prop.sizes["il"], prop.sizes["xl"]), dtype=np.float32)
    for s in range(0, prop.sizes["il"], chunk_il):
        sl = slice(s, s + chunk_il)
        v = np.nan_to_num(prop.isel(il=sl).to_numpy().astype(np.float64))
        ok = (t[None, None, :] >= top.isel(il=sl).to_numpy()[..., None]) & \
             (t[None, None, :] < base.isel(il=sl).to_numpy()[..., None])
        for m in masks or []:
            ok &= m.isel(il=sl).to_numpy().astype(bool)
        out[sl] = (v * ok).sum(axis=2) * dt_ms
    return xr.DataArray(out, coords={"il": prop["il"], "xl": prop["xl"]}, dims=("il", "xl"),
                        name=f"{prop.name}_sum")
