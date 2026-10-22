"""Tie the decimated Zp / Zs / density cubes to the wells over UBR + LBR.

Usage:
    python scripts/tie_wells.py [--data data/raw] [--cubes data/interim/cubes] [--out results/tie]
        [--upscale-ft 50] [--max-shift 20] [--method bilinear]
Logs are conditioned with mars.conditioning.load_well_zone and Backus-upscaled with
upscale_well; cube values are sampled along the trajectory (mars.tie). Statistics are computed
on 1 ms TWT bins. One bulk shift per well maximises r(Zp) within +-max-shift ms. Writes
tie_table.csv, summary.md, tie_summary.png and one track plot per well.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

from mars import dataset, paths  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from mars.conditioning import load_well_zone, upscale_well  # noqa: E402
from mars.io.las import read_las  # noqa: E402
from mars.naming import codename  # noqa: E402
from mars.tie import (  # noqa: E402
    CubeWindow,
    best_shift,
    las_path,
    offset_ilxl,
    pearson,
    read_welltops,
    resample_to_time,
    trajectory,
    well_path_tops,
    well_time_curve,
)

ZONES = ("UBR", "LBR")
RHO_OK = (1.8, 3.0)
OFFSETS_M = (25.0, 50.0, 100.0, 200.0, 400.0)
LOG_COLOR, CUBE_COLOR, RAW_COLOR = "#2a78d6", "#eb6834", "#b8b7b2"
PROPS = (("zp", "ZP", "Zp [ft/s*g/cc]"), ("zs", "ZS", "Zs [ft/s*g/cc]"), ("rho", "RHOB", "ρ [g/cc]"))


def with_impedances(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ZP"] = df["VP"] * df["RHOB"]
    df["ZS"] = df["VS"] * df["RHOB"]
    return df


def tie_well(well, data, cubes, affine, args):
    las = read_las(las_path(data, well))
    owt_col = well_time_curve(las, well)
    tops = well_path_tops(data, well, args.welltops)

    raw, up = [], []
    for zone in ZONES:
        try:
            z = load_well_zone(data, well, zone, require_vs=False)
        except ValueError:  # zone markers missing in this well
            continue
        if z.empty:
            continue
        u = upscale_well(z, args.upscale_ft)
        z["ZONE"] = u["ZONE"] = zone
        raw.append(with_impedances(z))
        up.append(with_impedances(u))
    if not up:
        return [], []
    raw, up = pd.concat(raw, ignore_index=True), pd.concat(up, ignore_index=True)
    traj = trajectory(tops, up["DEPT"], las["DEPT"], las[owt_col])
    up["TWT"] = traj.twt
    raw["TWT"] = 2 * np.interp(raw["DEPT"], las["DEPT"], las[owt_col])
    up = up[np.isfinite(up["TWT"])]
    traj = trajectory(tops, up["DEPT"], las["DEPT"], las[owt_col])
    margin = int(np.ceil(max(OFFSETS_M) / 6.25)) + 2  # crossline bin is 6.25 m
    try:
        win = {c: CubeWindow(cubes[c][c], affine, traj, margin_traces=margin,
                             margin_ms=args.max_shift + 10) for c in cubes}
    except ValueError:  # path outside the cube footprint
        return [], []
    if not np.isfinite(win["zp"].sample(0.0, args.method)).any():
        return [], []  # e.g. a sidetrack whose Baturaja part of the path is north of the survey

    def binned(values, zone=None):
        m = np.ones(len(up), bool) if zone is None else (up["ZONE"] == zone).to_numpy()
        return resample_to_time(up["TWT"].to_numpy()[m], np.asarray(values)[m])

    _, zp_log = binned(up["ZP"])
    shift, r_best, shifts, r_curve = best_shift(
        zp_log, lambda s: binned(win["zp"].sample(s, args.method))[1], args.max_shift)

    rows = []
    for zone in (None, *ZONES):
        if zone is not None and not (up["ZONE"] == zone).any():
            continue
        row = {"well": well, "zone": zone or "UBR+LBR", "shift_ms": shift}
        for cube, col, _ in PROPS:
            _, lg = binned(up[col], zone)
            for tag, s in (("0", 0.0), ("", shift)):
                _, cb = binned(win[cube].sample(s, args.method), zone)
                row[f"r_{cube}{'_s0' if tag else ''}"] = pearson(lg, cb)
            if cube in ("rho", "zp"):
                d = cb - lg
                ok = np.isfinite(d)
                row[f"n{'' if cube == 'rho' else '_zp'}"] = int(ok.sum())
                scale = 1.0 if cube == "rho" else 100.0 / np.nanmean(lg)
                unit = "" if cube == "rho" else "_pct"
                row[f"bias_{cube}{unit}"] = float(np.nanmean(d) * scale)
                row[f"rmse_{cube}{unit}"] = float(np.sqrt(np.nanmean(d**2)) * scale)
        m = np.ones(len(up), bool) if zone is None else (up["ZONE"] == zone).to_numpy()
        t = up["TWT"].to_numpy()[m]
        near = win["rho"].neighbourhood(2, (t.min() + shift, t.max() + shift))
        bad = (near < RHO_OK[0]) | (near > RHO_OK[1])
        row["rho_near_n"] = int(near.size)
        row["rho_near_nonphysical"] = int(bad.sum())
        row["rho_near_nonphysical_pct"] = float(100 * bad.mean()) if near.size else float("nan")
        row["path_drift_m"] = float(np.hypot(traj.x - traj.x[0], traj.y - traj.y[0]).max())
        row["il_xl_top"] = f"{win['zp'].fil[0]:.1f}/{win['zp'].fxl[0]:.1f}"
        rows.append(row)

    # same comparison with the cube sampled beside the well: is the match local to the
    # well (hard-data conditioning of the stochastic inversion) or does it hold away from it?
    offsets = []
    for d in (0.0, *OFFSETS_M):
        for az in np.arange(0, 360, 45) if d else [0.0]:
            dil, dxl = offset_ilxl(affine, d * np.sin(np.deg2rad(az)), d * np.cos(np.deg2rad(az)))
            rec = {"well": well, "offset_m": d, "azimuth": float(az)}
            for cube, col, _ in PROPS:
                _, lg = binned(up[col])
                _, cb = binned(win[cube].sample(shift, args.method, float(dil), float(dxl)))
                rec[f"r_{cube}"] = pearson(lg, cb)
                _, at_well = binned(win[cube].sample(shift, args.method))
                rec[f"r_{cube}_self"] = pearson(at_well, cb)  # cube's own lateral continuity
                if cube == "rho":
                    rec["rmse_rho"] = float(np.sqrt(np.nanmean((cb - lg) ** 2)))
            offsets.append(rec)

    plot_well(well, raw, up, win, shift, r_curve, shifts, rows[0], args)
    return rows, offsets


def plot_well(well, raw, up, win, shift, r_curve, shifts, row, args):
    fig, axes = plt.subplots(1, 4, figsize=(14, 7.5), sharey=False, constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.7]})
    for ax, (cube, col, label) in zip(axes[:3], PROPS, strict=True):
        ax.plot(raw[col], raw["TWT"], color=RAW_COLOR, lw=0.6, label="log 0.5 ft")
        ax.plot(up[col], up["TWT"], color=LOG_COLOR, lw=1.6, label=f"log upscaled {args.upscale_ft:g} ft")
        ax.plot(win[cube].sample(shift, args.method), up["TWT"], color=CUBE_COLOR, lw=1.6,
                label=f"cube (shift {shift:+.0f} ms)")
        for zone in ZONES:
            zt = up.loc[up["ZONE"] == zone, "TWT"]
            if len(zt):
                ax.axhline(zt.min(), color="0.35", lw=0.7, ls=":")
                ax.text(0.01, zt.min(), f" top {zone}", fontsize=8,
                        color="0.3", va="bottom", transform=ax.get_yaxis_transform())
        r = row[f"r_{cube}"]
        ax.set(title=f"{label.split(' ')[0]}   r = {r:.2f}", xlabel=label)
        ax.grid(alpha=0.25, lw=0.5)
        ax.invert_yaxis()
    axes[0].set_ylabel("TWT [ms] (log time; cube sampled at TWT + shift)")
    axes[2].axvspan(0, RHO_OK[0], color="0.9", zorder=0)
    axes[2].axvspan(RHO_OK[1], 4, color="0.9", zorder=0)
    lo = np.nanmin([raw["RHOB"].min(), np.nanmin(win["rho"].sample(shift, args.method))])
    hi = np.nanmax([raw["RHOB"].max(), np.nanmax(win["rho"].sample(shift, args.method))])
    axes[2].set_xlim(lo - 0.05, hi + 0.05)
    axes[0].legend(loc="lower left", fontsize=8)
    ax = axes[3]
    ax.plot(shifts, r_curve, color=LOG_COLOR, lw=1.6)
    ax.axvline(shift, color=CUBE_COLOR, lw=1.2, ls="--")
    ax.set(title="r(Zp) vs bulk shift", xlabel="shift [ms]", ylabel="r")
    ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle(f"{codename(well)}: log vs cube, UBR + LBR (n = {row['n']} × 1 ms)")
    fig.savefig(args.out / f"track_{codename(well)}.png", dpi=100)
    plt.close(fig)


def plot_summary(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, constrained_layout=True)
    for ax, zone in zip(axes, ZONES, strict=True):
        d = df[df.zone == zone]
        x = np.arange(len(d))
        ax.bar(x - 0.2, d["r_zp"], 0.38, color=LOG_COLOR, label="r(Zp)")
        ax.bar(x + 0.2, d["r_rho"], 0.38, color=CUBE_COLOR, label="r(ρ)")
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_xticks(x, [codename(w) for w in d["well"]], rotation=45)
        lo = min(0.0, float(np.nanmin(df[["r_zp", "r_rho"]].to_numpy())) - 0.05)
        ax.set(title=f"{zone}: cube vs upscaled log (best shift)", ylim=(lo, 1))
        ax.grid(axis="y", alpha=0.25, lw=0.5)
    axes[0].set_ylabel("Pearson r")
    axes[0].legend(loc="lower left")
    fig.savefig(out / "tie_summary.png", dpi=110)
    plt.close(fig)


def plot_offsets(off: pd.DataFrame, out: Path):
    per_well = off.groupby(["well", "offset_m"])[["r_zp", "r_rho"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, constrained_layout=True)
    for ax, (col, label) in zip(axes, (("r_zp", "r(Zp)"), ("r_rho", "r(ρ)")), strict=True):
        for i, (_, g) in enumerate(per_well.groupby("well")):
            ax.plot(g["offset_m"], g[col], color=RAW_COLOR, lw=0.9,
                    label="cube vs log, single wells" if i == 0 else None)
        med = per_well.groupby("offset_m")[col].median()
        ax.plot(med.index, med.values, color=LOG_COLOR, lw=2.2, marker="o", ms=7,
                label="cube vs well log (median of 12 wells)")
        self_med = off.groupby(["well", "offset_m"])[col + "_self"].mean().groupby("offset_m").median()
        ax.plot(self_med.index, self_med.values, color=CUBE_COLOR, lw=2.2, ls="--", marker="s",
                ms=7, label="cube vs cube at the well (median)")
        ax.set(title=f"{label}, UBR+LBR, cube sampled beside the well", xlabel="lateral offset [m]")
        ax.set_xscale("symlog", linthresh=25)
        ax.set_xticks([0, 25, 50, 100, 200, 400], ["0", "25", "50", "100", "200", "400"])
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(loc="lower left")
    axes[0].set_ylabel("Pearson r (mean of 8 azimuths)")
    fig.savefig(out / "tie_vs_offset.png", dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=paths.RAW)
    ap.add_argument("--cubes", type=Path, default=paths.CUBES)
    ap.add_argument("--out", type=Path, default=paths.RESULTS / "tie")
    ap.add_argument("--upscale-ft", type=float, default=50.0)
    ap.add_argument("--max-shift", type=float, default=20.0)
    ap.add_argument("--method", choices=("bilinear", "nearest"), default="bilinear")
    ap.add_argument("--wells", nargs="+", default=None,
                    help="default: every LAS in <data>/wells")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.welltops = read_welltops(args.data / "tops" / "welltops.csv")
    cubes = {c: xr.open_zarr(args.cubes / f"{c}.zarr") for c in ("zp", "zs", "rho")}
    affine = cubes["zp"].attrs["affine_ilxl_to_xy"]

    if args.wells is None:
        args.wells = [dataset.well_id(f.stem)
                      for f in sorted((args.data / "wells").glob("*.las"))]
    rows, offs = [], []
    for well in args.wells:
        r, o = tie_well(well, args.data, cubes, affine, args)
        if not r:
            print(well, "skipped: no cube samples along the zone path (outside the survey?)")
            continue
        rows += r
        offs += o
        print(well, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r[0].items()}))
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "tie_table.csv", index=False, float_format="%.4f")
    plot_summary(df, args.out)
    off = pd.DataFrame(offs)
    off.to_csv(args.out / "tie_offsets.csv", index=False, float_format="%.4f")
    plot_offsets(off, args.out)


if __name__ == "__main__":
    main()
