"""Phase 5 step 5: interpretation figures from the seismic MARS property cubes.

Usage:
    python scripts/make_phase5_figures.py [--props data/interim/props/props_s25.zarr]
        [--sensitivity props_s0.zarr props.zarr] [--out results/figures/phase5] [--no-3d]

Map inputs follow the 2026-10-21 decision (vault decisions log): UBRn PHIE from the full (density)
transform, VSH and SWT from the Zp-only transforms, LBRc PHIE not resolvable (not mapped).
Figures (all labels are codenames, coordinates relative km):
  structure_maps.png        zone-top TWT and isochrons
  property_maps.png         interval means of the recommended properties, log means posted
  net_reservoir_maps.png    net thickness from the GGRP DST cut-offs
  reservoir_property_maps.png  PHIE / VSH / SWT with non-reservoir faded and the area passing
                            all cut-offs outlined, plus hydrocarbon pore thickness (UBRn) or net
                            reservoir (LBRc); also a well-area zoom (_well_area.png)
  variant_difference.png    |full - Zp-only| interval means (model uncertainty)
  sigma_sensitivity.png     UBRn PHIE map at each lateral smoothing
  section_<line>.png        PHIE / VSH / SWT sections with upscaled logs along the well paths
  scene3d.png               pyvista: Top UBR coloured by UBRn PHIE, Top TAF, well paths
Also writes maps_at_wells.csv (codenames): interval mean of the map at each well vs the log mean,
and caches the maps in data/interim/props/maps_<store>.nc.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mars import dataset, paths  # noqa: E402

warnings.filterwarnings("ignore", message="CUDA path could not be detected")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from mars.conditioning import load_well_zone, upscale_well  # noqa: E402
from mars.io.las import read_las  # noqa: E402
from mars.io.segy import ilxl_to_xy, xy_to_ilxl  # noqa: E402
from mars.naming import codename  # noqa: E402
from mars.tie import (  # noqa: E402
    las_path,
    read_welltops,
    resample_to_time,
    trajectory,
    well_path_tops,
    well_time_curve,
)
from mars.viz.maps import interval_stat, interval_sum, net_thickness_ms, plot_map  # noqa: E402
from mars.viz.sections import (  # noqa: E402
    plot_section,
    project_to_polyline,
    sample_section,
    surface_along,
)

DATA = paths.RAW
PROPS = paths.PROPS
WELLS = list(dataset.wells("all"))
# zone -> (top surface, base surface, upscale ft)
ZONES = {"UBRn": ("top_UBR", "top_BRF", 25), "LBRc": ("top_LBRc", "top_TAF", 50)}
# recommended variable per zone/target (None = not resolvable from this inversion)
RECOMMENDED = {
    "UBRn": {"PHIE": "phie", "VSH": "vsh_zponly", "SWT": "swt_zponly"},
    "LBRc": {"PHIE": None, "VSH": "vsh_zponly", "SWT": "swt_zponly"},
}
ALTERNATIVE = {"phie": "phie_zponly", "vsh_zponly": "vsh", "swt_zponly": "swt"}
# GGRP DST cut-offs (vault/05-Literature/ggrp-field-study.md)
CUTOFFS = {"UBRn": {"VSH": 0.60, "PHIE": 0.083, "SWT": 0.82},
           "LBRc": {"VSH": 0.49, "SWT": 0.82}}
CMAPS = {"PHIE": "viridis", "VSH": "YlOrBr", "SWT": "Blues_r"}
RANGES = {"PHIE": (0.0, 0.30), "VSH": (0.0, 0.8), "SWT": (0.2, 1.0)}
LINES = {
    "A": {"wells": list(dataset.wells("section_A")), "extend_m": 1500.0},  # MESSI-RONALDO
    "B": {"wells": list(dataset.wells("section_B")), "extend_m": 500.0},   # west-east
}


# --------------------------------------------------------------------------- inputs
def open_props(path: Path) -> xr.Dataset:
    ds = xr.open_zarr(path)
    affine = json.loads(ds.attrs["affine_ilxl_to_xy"]) if isinstance(
        ds.attrs.get("affine_ilxl_to_xy"), str) else ds.attrs.get("affine_ilxl_to_xy")
    if affine is None:
        affine = xr.open_zarr(paths.CUBES / "zp.zarr").attrs[
            "affine_ilxl_to_xy"]
    ds.attrs["affine_ilxl_to_xy"] = affine
    il, xl = np.meshgrid(ds["il"].to_numpy(), ds["xl"].to_numpy(), indexing="ij")
    x, y = ilxl_to_xy(affine, il, xl)
    return ds.assign_coords(cdp_x=(("il", "xl"), x), cdp_y=(("il", "xl"), y))


def surfaces(ds: xr.Dataset) -> dict[str, xr.DataArray]:
    return {k: ds[k].load() for k in ("top_UBR", "top_BRF", "top_LBRc", "top_TAF")}


def well_data(welltops) -> dict:
    """Per well: full trajectory (for 3D / sections) and per-zone upscaled logs."""
    out = {}
    for w in WELLS:
        try:
            tops = well_path_tops(DATA, w, welltops)
            las = read_las(las_path(DATA, w))
            tcurve = well_time_curve(las, w)
        except (FileNotFoundError, KeyError, ValueError, IndexError):
            continue
        rec = {"zones": {}}
        md = las["DEPT"].to_numpy()
        rec["path"] = trajectory(tops, md, las["DEPT"], las[tcurve])
        for zone, (*_, up_ft) in ZONES.items():
            try:
                z = load_well_zone(DATA, w, zone, require_vs=False)
            except (ValueError, KeyError):
                continue
            up = upscale_well(z, up_ft)
            if up.empty:
                continue
            tr = trajectory(tops, up["DEPT"], las["DEPT"], las[tcurve])
            rec["zones"][zone] = {"up": up, "traj": tr}
        out[w] = rec
    return out


# --------------------------------------------------------------------------- maps
def compute_maps(ds: xr.Dataset, surf: dict, cache: Path) -> xr.Dataset:
    if cache.exists():
        return xr.open_dataset(cache).load()
    maps = {}
    for zone, (top, base, _) in ZONES.items():
        for var in RECOMMENDED[zone].values():
            for v in (var, ALTERNATIVE.get(var)) if var else ():
                key = f"{zone}_{v}"
                print("  interval mean", key, flush=True)
                maps[key] = interval_stat(ds[v], surf[top], surf[base], "mean")
        c = CUTOFFS[zone]
        masks = [ds[RECOMMENDED[zone]["VSH"]] <= c["VSH"], ds[RECOMMENDED[zone]["SWT"]] <= c["SWT"]]
        if "PHIE" in c:
            masks.append(ds[RECOMMENDED[zone]["PHIE"]] >= c["PHIE"])
        print("  net", zone, flush=True)
        maps[f"{zone}_net_ms"] = net_thickness_ms(masks, surf[top], surf[base])
        maps[f"{zone}_gross_ms"] = (surf[base] - surf[top]).where(lambda a: a > 0)
    out = xr.Dataset(maps).assign_coords(cdp_x=ds["cdp_x"], cdp_y=ds["cdp_y"])
    out.to_netcdf(cache)
    return out


def map_value_at(field: xr.DataArray, affine, x, y) -> float:
    il, xl = xy_to_ilxl(affine, np.atleast_1d(x), np.atleast_1d(y))
    return float(field.interp(il=il[0], xl=xl[0]))


def well_posts(wd: dict, zone: str, target: str | None = None) -> list[dict]:
    posts = []
    for w, rec in wd.items():
        z = rec["zones"].get(zone)
        if z is None:
            continue
        val = None
        if target and target in z["up"] and z["up"][target].notna().any():
            val = float(z["up"][target].mean())
        posts.append({"x": float(np.nanmean(z["traj"].x)), "y": float(np.nanmean(z["traj"].y)),
                      "label": codename(w), "value": val, "well": w})
    return posts


def fig_structure(surf, ds, wd, out):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, k in zip(axes.flat[:4], surf, strict=True):
        plot_map(ax, surf[k], ds["cdp_x"], ds["cdp_y"], wells=well_posts(wd, "UBRn"),
                 cmap="terrain_r", title=f"{k} (TWT)", cbar_label="TWT (ms)", contours=12)
    for ax, zone in zip(axes.flat[4:], ZONES, strict=True):
        top, base, _ = ZONES[zone]
        plot_map(ax, (surf[base] - surf[top]).where(lambda a: a > 0), ds["cdp_x"], ds["cdp_y"],
                 wells=well_posts(wd, zone), cmap="magma", title=f"{zone} isochron",
                 cbar_label="ms TWT", contours=8)
    fig.suptitle("Structure (horizons and derived Top LBRc) — Field X, Baturaja Fm.")
    fig.tight_layout()
    fig.savefig(out / "structure_maps.png", dpi=110)
    plt.close(fig)


def fig_properties(maps, ds, wd, out, sigma_m):
    panels = [(z, t, v) for z in ZONES for t, v in RECOMMENDED[z].items()]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, (zone, target, var) in zip(axes.flat, panels, strict=True):
        if var is None:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{zone} {target}: not resolvable\nfrom this inversion\n"
                    "(seismic r at wells −0.18 pooled;\nlog-scale blind r 0.92)",
                    ha="center", va="center", fontsize=12, transform=ax.transAxes)
            continue
        variant = "full" if "zponly" not in var else "Zp-only"
        plot_map(ax, maps[f"{zone}_{var}"], ds["cdp_x"], ds["cdp_y"],
                 wells=well_posts(wd, zone, target), cmap=CMAPS[target],
                 vmin=RANGES[target][0], vmax=RANGES[target][1],
                 title=f"{zone} mean {target} ({variant} transform)", cbar_label=target)
    fig.suptitle(f"Interval-mean properties from seismic MARS (lateral σ = {sigma_m:g} m); "
                 "posts: codename and upscaled-log mean; grey = no data / zero thickness")
    fig.tight_layout()
    fig.savefig(out / "property_maps.png", dpi=110)
    # zoom on the well area (relative km of the posts, 0.8 km margin)
    posts = well_posts(wd, "UBRn")
    x0, y0 = float(ds["cdp_x"].min()), float(ds["cdp_y"].min())
    px = np.array([p["x"] for p in posts]) - x0
    py = np.array([p["y"] for p in posts]) - y0
    for ax in axes.flat:
        if ax.has_data():
            ax.set_xlim(px.min() / 1000 - 0.8, px.max() / 1000 + 0.8)
            ax.set_ylim(py.min() / 1000 - 0.8, py.max() / 1000 + 0.8)
    fig.savefig(out / "property_maps_well_area.png", dpi=130)
    plt.close(fig)


def fig_net(maps, ds, wd, out, vp):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, zone in zip(axes, ZONES, strict=True):
        net_ft = maps[f"{zone}_net_ms"] / 2000.0 * vp[zone]
        c = CUTOFFS[zone]
        rule = ", ".join(f"{k} {'≥' if k == 'PHIE' else '≤'} {v}" for k, v in c.items())
        plot_map(ax, net_ft.where(maps[f"{zone}_gross_ms"] > 0), ds["cdp_x"], ds["cdp_y"],
                 wells=well_posts(wd, zone), cmap="cividis",
                 title=f"{zone} net reservoir ({rule})", cbar_label="net thickness (ft, zone Vp)")
    fig.tight_layout()
    fig.savefig(out / "net_reservoir_maps.png", dpi=110)
    plt.close(fig)


def fig_difference(maps, ds, wd, out):
    pairs = [(z, t, v) for z in ZONES for t, v in RECOMMENDED[z].items() if v]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, (zone, target, var) in zip(axes.flat, pairs, strict=False):
        d = abs(maps[f"{zone}_{var}"] - maps[f"{zone}_{ALTERNATIVE[var]}"])
        plot_map(ax, d, ds["cdp_x"], ds["cdp_y"], wells=well_posts(wd, zone), cmap="Reds",
                 vmin=0, vmax=float(np.nanpercentile(d, 98)),
                 title=f"{zone} {target}: |full − Zp-only|", cbar_label=f"Δ mean {target}")
    for ax in axes.flat[len(pairs):]:
        ax.axis("off")
    fig.suptitle("Transform-choice uncertainty (difference of interval means)")
    fig.tight_layout()
    fig.savefig(out / "variant_difference.png", dpi=110)
    plt.close(fig)


def fig_sigma(stores: list[tuple[float, Path]], surf, wd, out):
    fig, axes = plt.subplots(1, len(stores), figsize=(6.5 * len(stores), 7), squeeze=False)
    top, base, _ = ZONES["UBRn"]
    for ax, (sigma, path) in zip(axes[0], stores, strict=True):
        ds = open_props(path)
        cache = PROPS / f"map_UBRn_phie_{path.stem}.nc"
        if cache.exists():
            m = xr.open_dataarray(cache).load()
        else:
            m = interval_stat(ds["phie"], surf[top], surf[base], "mean")
            m.to_netcdf(cache)
        plot_map(ax, m, ds["cdp_x"], ds["cdp_y"], wells=well_posts(wd, "UBRn", "PHIE"),
                 cmap=CMAPS["PHIE"], vmin=RANGES["PHIE"][0], vmax=RANGES["PHIE"][1],
                 title=f"UBRn mean PHIE, lateral σ = {sigma:g} m", cbar_label="PHIE")
    fig.tight_layout()
    fig.savefig(out / "sigma_sensitivity.png", dpi=100)
    plt.close(fig)


def maps_at_wells(maps, ds, wd, out) -> pd.DataFrame:
    rows = []
    affine = ds.attrs["affine_ilxl_to_xy"]
    for zone in ZONES:
        for target, var in RECOMMENDED[zone].items():
            if var is None:
                continue
            for p in well_posts(wd, zone, target):
                if p["value"] is None or (target == "SWT" and p["well"] in dataset.wells("swt_exclude")):
                    continue
                rows.append({"zone": zone, "target": target, "well": p["label"],
                             "log_mean": p["value"],
                             "map_mean": map_value_at(maps[f"{zone}_{var}"], affine,
                                                      p["x"], p["y"])})
    df = pd.DataFrame(rows)
    df.to_csv(out / "maps_at_wells.csv", index=False, float_format="%.4f")
    summ = (df.dropna().groupby(["zone", "target"])
            .apply(lambda g: pd.Series({"wells": len(g),
                                        "r": g["log_mean"].corr(g["map_mean"]),
                                        "bias": (g["map_mean"] - g["log_mean"]).mean()}),
                   include_groups=False))
    summ.to_csv(out / "maps_at_wells_summary.csv", float_format="%.3f")
    print(summ.round(3).to_string())
    return df


def hc_pore_thickness(ds, surf, cache: Path) -> xr.DataArray:
    """UBRn hydrocarbon pore thickness, sum of PHIE * (1 - SWT) over net-pay samples (ms)."""
    if cache.exists():
        return xr.open_dataarray(cache).load()
    top, base, _ = ZONES["UBRn"]
    rec, c = RECOMMENDED["UBRn"], CUTOFFS["UBRn"]
    phi, sw = ds[rec["PHIE"]], ds[rec["SWT"]]
    masks = [ds[rec["VSH"]] <= c["VSH"], phi >= c["PHIE"], sw <= c["SWT"]]
    print("  HC pore thickness UBRn", flush=True)
    hpt = interval_sum((phi * (1 - sw)).rename("hpt"), surf[top], surf[base], masks)
    hpt.to_netcdf(cache)
    return hpt


def passes(zone: str, target: str, field: xr.DataArray) -> np.ndarray:
    cut = CUTOFFS[zone][target]
    a = field.to_numpy()
    return (a >= cut) if target == "PHIE" else (a <= cut)


def fig_reservoir(maps, hpt, ds, wd, out, vp, sigma_m):
    """Thesis-style target-property maps highlighting the reservoir."""
    x0, y0 = float(ds["cdp_x"].min()), float(ds["cdp_y"].min())
    sx = (ds["cdp_x"].to_numpy() - x0) / 1000
    sy = (ds["cdp_y"].to_numpy() - y0) / 1000
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    for row, zone in zip(axes, ZONES, strict=True):
        rec = RECOMMENDED[zone]
        fields = {t: maps[f"{zone}_{v}"] for t, v in rec.items() if v}
        ok_all = np.ones(next(iter(fields.values())).shape, bool)
        for t, f in fields.items():
            ok_all &= passes(zone, t, f)
        ok_all &= np.isfinite(maps[f"{zone}_gross_ms"].to_numpy())
        live = np.isfinite(maps[f"{zone}_gross_ms"].to_numpy())
        if zone == "UBRn":  # most of the UBR passes on average: outline the best HCPT quartile
            h = hpt.to_numpy()
            ok_all &= h >= np.nanpercentile(h[ok_all], 75)
            note = "outline: passes all cut-offs and top-quartile HCPT (sweet spots)"
        else:
            note = "outline: passes all cut-offs"
        note += f", {ok_all.sum() / live.sum():.0%} of zone area"
        for ax, target in zip(row[:3], ("PHIE", "VSH", "SWT"), strict=True):
            if target not in fields:
                ax.axis("off")
                ax.text(0.5, 0.5, f"{zone} PHIE: not resolvable\nfrom this inversion\n"
                        "(reservoir outline uses\nVSH and SWT only)", ha="center",
                        va="center", fontsize=12, transform=ax.transAxes)
                continue
            f = fields[target]
            cut = CUTOFFS[zone][target]
            plot_map(ax, f, ds["cdp_x"], ds["cdp_y"], wells=well_posts(wd, zone, target),
                     cmap=CMAPS[target], vmin=RANGES[target][0], vmax=RANGES[target][1],
                     title=f"{zone} {target} — reservoir where {target} "
                           f"{'≥' if target == 'PHIE' else '≤'} {cut}",
                     cbar_label=f"interval mean {target}")
            fail = ~passes(zone, target, f) & np.isfinite(f.to_numpy())
            ax.pcolormesh(sx, sy, np.where(fail, 1.0, np.nan), cmap="Greys", vmin=0, vmax=1.5,
                          alpha=0.55, shading="nearest")
            ax.contour(sx, sy, ok_all.astype(float), levels=[0.5], colors="red",
                       linewidths=0.9)
        ax = row[3]
        if zone == "UBRn":
            hpt_ft = (hpt / 2000.0 * vp[zone]).where(maps[f"{zone}_gross_ms"] > 0)
            plot_map(ax, hpt_ft, ds["cdp_x"], ds["cdp_y"], wells=well_posts(wd, zone),
                     cmap="inferno", vmin=0, vmax=float(np.nanpercentile(hpt_ft, 99)),
                     title="UBRn HC pore thickness Σφ(1−Sw)h (net pay)",
                     cbar_label="HCPT (ft)")
        else:
            net_ft = (maps[f"{zone}_net_ms"] / 2000.0 * vp[zone]).where(
                maps[f"{zone}_gross_ms"] > 0)
            plot_map(ax, net_ft, ds["cdp_x"], ds["cdp_y"], wells=well_posts(wd, zone),
                     cmap="inferno", title="LBRc net reservoir (VSH, SWT cut-offs)",
                     cbar_label="net thickness (ft)")
        ax.contour(sx, sy, ok_all.astype(float), levels=[0.5], colors="cyan", linewidths=0.9)
        shown = next(a for a in row if a.has_data())
        shown.text(0.01, 0.01, note, transform=shown.transAxes, fontsize=8, color="red",
                   bbox={"fc": "w", "alpha": 0.8, "lw": 0})
    fig.suptitle("Target property distribution (seismic MARS, interval means, lateral σ = "
                 f"{sigma_m:g} m). Faded: fails that property's GGRP cut-off. "
                 "Outline: potential hydrocarbon reservoir (see note per zone).")
    fig.tight_layout()
    fig.savefig(out / "reservoir_property_maps.png", dpi=110)
    posts = well_posts(wd, "UBRn")
    px = np.array([p["x"] for p in posts]) / 1000 - x0 / 1000
    py = np.array([p["y"] for p in posts]) / 1000 - y0 / 1000
    for ax in axes.flat:
        if ax.has_data():
            ax.set_xlim(px.min() - 0.8, px.max() + 0.8)
            ax.set_ylim(py.min() - 0.8, py.max() + 0.8)
    fig.savefig(out / "reservoir_property_maps_well_area.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- sections
def line_vertices(wd, wells, extend_m):
    pts = []
    for w in wells:
        tr = wd[w]["zones"]["UBRn"]["traj"] if "UBRn" in wd[w]["zones"] else wd[w]["path"]
        pts.append((float(tr.x[0]), float(tr.y[0])))
    xy = np.array(pts)
    d0 = xy[0] - xy[1]
    d1 = xy[-1] - xy[-2]
    xy = np.vstack([xy[0] + d0 / np.hypot(*d0) * extend_m, xy,
                    xy[-1] + d1 / np.hypot(*d1) * extend_m])
    return xy[:, 0], xy[:, 1]


def fig_section(name, spec, ds, surf, wd, out):
    wells = [w for w in spec["wells"] if w in wd]
    xv, yv = line_vertices(wd, wells, spec["extend_m"])
    twts = [s for w in wells for s in (np.nanmin(z["traj"].twt) for z in wd[w]["zones"].values())]
    twtb = [s for w in wells for s in (np.nanmax(z["traj"].twt) for z in wd[w]["zones"].values())]
    t_rng = (min(twts) - 40, max(twtb) + 40)
    labels = ["", *[codename(w) for w in wells], ""]
    secs, hz = {}, None
    for var in {v for z in RECOMMENDED.values() for v in z.values() if v}:
        da = ds[var]
        da.attrs["affine_ilxl_to_xy"] = ds.attrs["affine_ilxl_to_xy"]
        secs[var] = sample_section(da, xv, yv, labels=labels, twt_range=t_rng)
        if hz is None:
            hz = {k: surface_along(secs[var], s) for k, s in surf.items()}
    any_sec = next(iter(secs.values()))
    t = any_sec.twt_ms[None, :]
    in_zone = {z: (t >= hz[a][:, None]) & (t < hz[b][:, None]) for z, (a, b, _) in ZONES.items()}
    fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)
    for ax, target in zip(axes, ("PHIE", "VSH", "SWT"), strict=True):
        vals = np.full(any_sec.values.shape, np.nan, np.float32)
        for zone in ZONES:
            var = RECOMMENDED[zone][target]
            if var:
                vals = np.where(in_zone[zone], secs[var].values, vals)
        sec = secs[RECOMMENDED["UBRn"][target]]
        shown = type(sec)(**{**sec.__dict__, "values": vals})
        logs = []
        for w in wells:
            for zone, z in wd[w]["zones"].items():
                if RECOMMENDED[zone][target] is None or target not in z["up"]:
                    continue
                dist, _ = project_to_polyline(shown, z["traj"].x, z["traj"].y)
                tb, val = resample_to_time(z["traj"].twt, z["up"][target].to_numpy())
                _, db = resample_to_time(z["traj"].twt, dist)
                logs.append({"dist_m": db, "twt_ms": tb, "values": val, "label": codename(w)})
        variants = ", ".join(f"{z} {'full' if 'zponly' not in v else 'Zp-only'}"
                             for z in ZONES if (v := RECOMMENDED[z][target]))
        plot_section(ax, shown, cmap=CMAPS[target], vmin=RANGES[target][0],
                     vmax=RANGES[target][1], horizons=hz, well_logs=logs,
                     title=f"{target} ({variants})"
                     + ("; LBRc not resolvable" if target == "PHIE" else ""),
                     cbar_label=target)
        if ax is not axes[-1]:
            ax.set_xlabel("")
    fig.suptitle(f"Section {name}–{name}′: seismic MARS properties with upscaled "
                 "logs along the well paths")
    fig.tight_layout()
    fig.savefig(out / f"section_{name}.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- 3D
def fig_3d(maps, ds, surf, wd, out, zscale=-8.0):
    import pyvista as pv

    pv.OFF_SCREEN = True
    x = ds["cdp_x"].to_numpy()
    y = ds["cdp_y"].to_numpy()
    x0, y0 = np.nanmin(x), np.nanmin(y)
    step = 4  # decimate crosslines for the scene

    def grid(tw, scal, name):
        tw = tw.to_numpy()[:, ::step]
        g = pv.StructuredGrid((x[:, ::step] - x0) / 1000, (y[:, ::step] - y0) / 1000,
                              tw / 1000 * zscale)
        g[name] = np.asarray(scal)[:, ::step].ravel(order="F")
        return g.threshold(value=(-1e9, 1e9), scalars=name) if np.isnan(g[name]).any() else g

    p = pv.Plotter(off_screen=True, window_size=(1600, 1000))
    p.add_mesh(grid(surf["top_UBR"], maps["UBRn_phie"].to_numpy(), "UBRn PHIE"),
               cmap="viridis", clim=RANGES["PHIE"],
               scalar_bar_args={"title": "UBRn PHIE (on Top UBR)", "vertical": True,
                                "position_x": 0.9, "position_y": 0.55, "height": 0.35})
    p.add_mesh(grid(surf["top_TAF"], maps["LBRc_vsh_zponly"].to_numpy(), "LBRc VSH"),
               cmap="YlOrBr", clim=RANGES["VSH"], opacity=0.85,
               scalar_bar_args={"title": "LBRc VSH (on Top TAF)", "vertical": True,
                                "position_x": 0.9, "position_y": 0.1, "height": 0.35})
    for w, rec in wd.items():
        tr = rec["path"]
        ok = np.isfinite(tr.twt) & (tr.twt > np.nanmin(surf["top_UBR"]) - 150)
        if ok.sum() < 2:
            continue
        pts = np.c_[(tr.x[ok] - x0) / 1000, (tr.y[ok] - y0) / 1000, tr.twt[ok] / 1000 * zscale]
        p.add_mesh(pv.lines_from_points(pts[::5]), color="black", line_width=3)
        p.add_point_labels(pts[:1], [codename(w)], font_size=10, point_size=1,
                           shape_opacity=0.4)
    p.set_scale(zscale=1.0)
    p.show_grid(xtitle="x (km, rel.)", ytitle="y (km, rel.)", ztitle=f"TWT (s) x {-zscale:g}")
    p.camera_position = "iso"
    p.camera.azimuth = 200
    p.camera.elevation = -10
    p.screenshot(str(out / "scene3d.png"))
    p.close()


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--props", type=Path, default=PROPS / "props_s25.zarr")
    ap.add_argument("--sensitivity", type=Path, nargs="*",
                    default=[PROPS / "props_s0.zarr", PROPS / "props_s25.zarr",
                             PROPS / "props.zarr"])
    ap.add_argument("--out", type=Path, default=paths.RESULTS / "figures" / "phase5")
    ap.add_argument("--no-3d", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ds = open_props(a.props)
    sigma_m = float(ds.attrs["lateral_sigma_m"])
    vp = json.loads(ds.attrs["zone_vp_ft_s"]) if isinstance(ds.attrs["zone_vp_ft_s"], str) \
        else ds.attrs["zone_vp_ft_s"]
    surf = surfaces(ds)
    print("wells ...", flush=True)
    wd = well_data(read_welltops(DATA / "tops" / "welltops.csv"))
    print("maps ...", flush=True)
    maps = compute_maps(ds, surf, PROPS / f"maps_{a.props.stem}.nc")
    fig_structure(surf, ds, wd, a.out)
    fig_properties(maps, ds, wd, a.out, sigma_m)
    fig_net(maps, ds, wd, a.out, vp)
    fig_difference(maps, ds, wd, a.out)
    hpt = hc_pore_thickness(ds, surf, PROPS / f"hpt_{a.props.stem}.nc")
    fig_reservoir(maps, hpt, ds, wd, a.out, vp, sigma_m)
    maps_at_wells(maps, ds, wd, a.out)
    print("sections ...", flush=True)
    for name, spec in LINES.items():
        fig_section(name, spec, ds, surf, wd, a.out)
    stores = [(float(xr.open_zarr(p).attrs["lateral_sigma_m"]), p) for p in a.sensitivity
              if p.exists()]
    if len(stores) > 1:
        print("sigma sensitivity ...", flush=True)
        fig_sigma(sorted(stores), surf, wd, a.out)
    if not a.no_3d:
        print("3D ...", flush=True)
        fig_3d(maps, ds, surf, wd, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
