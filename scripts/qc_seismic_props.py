"""QC of the seismic property cubes at the wells (not blind: the inversion used these wells).

Usage:
    python scripts/qc_seismic_props.py [--data data/raw] [--props data/interim/props/props.zarr]
        [--out results/seismic_qc]
For each well and zone (UBRn, LBRc), PHIE/VSH/SWT predicted from the cubes are sampled along the
well path (mars.tie, bilinear, no time shift) and compared, on 1 ms TWT bins, with the logs
upscaled as in the transforms (load_well_zone(require_vs=False) -> upscale_well(z, upscale_ft)).
For reference, the same transform applied to the upscaled *logs* is scored too ("log MARS").
Writes qc_table.csv (real ids) and summary.md (codenames).
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from mars.conditioning import load_well_zone, upscale_well  # noqa: E402
from mars.io.las import read_las  # noqa: E402
from mars.naming import codename  # noqa: E402
from mars.seismic_apply import RHO_CLIP, ZP_VALID  # noqa: E402
from mars.tie import (  # noqa: E402
    CubeWindow,
    Trajectory,
    las_path,
    pearson,
    read_welltops,
    resample_to_time,
    trajectory,
    well_path_tops,
    well_time_curve,
)
from mars.transform import MarsTransform  # noqa: E402

ZONES = ("UBRn", "LBRc")
TARGETS = ("PHIE", "VSH", "SWT")
VARIANTS = ("", "_zponly")


def metrics(log, pred) -> dict:
    ok = np.isfinite(log) & np.isfinite(pred)
    if ok.sum() < 3:
        return {"n": int(ok.sum()), "r": np.nan, "rmse": np.nan, "bias": np.nan}
    d = pred[ok] - log[ok]
    return {"n": int(ok.sum()), "r": pearson(log[ok], pred[ok]),
            "rmse": float(np.sqrt(np.mean(d**2))), "bias": float(np.mean(d))}


def qc_well(well, zone, data, props, affine, transforms, welltops, raw_cubes=None):
    try:
        z = load_well_zone(data, well, zone, require_vs=False)
    except (ValueError, KeyError):
        return [], []
    if len(z) < 10:
        return [], []
    up = upscale_well(z, transforms[("PHIE", "")].meta["upscale_ft"])
    if up.empty:
        return [], []
    las = read_las(las_path(data, well))
    traj = trajectory(well_path_tops(data, well, welltops), up["DEPT"], las["DEPT"],
                      las[well_time_curve(las, well)])
    ok = np.isfinite(traj.twt)
    if ok.sum() < 3:
        return [], []
    up = up[ok].reset_index(drop=True)
    traj = Trajectory(traj.md[ok], traj.x[ok], traj.y[ok], traj.twt[ok])
    near, raw_zp, raw_rho = {}, None, None
    if raw_cubes is not None:  # non-physical inversion values within 2 traces of the path
        t_rng = (np.nanmin(traj.twt), np.nanmax(traj.twt))
        vals = {c: CubeWindow(raw_cubes[c][c], affine, traj, margin_traces=3, margin_ms=5)
                .neighbourhood(2, t_rng) for c in raw_cubes}
        near = {"near_zp_n": int(vals["zp"].size),
                "near_zp_masked": int(((vals["zp"] < ZP_VALID[0]) | (vals["zp"] > ZP_VALID[1])).sum()),
                "near_rho_n": int(vals["rho"].size),
                "near_rho_clipped": int(((vals["rho"] < RHO_CLIP[0]) | (vals["rho"] > RHO_CLIP[1])).sum())}
        # unsmoothed cube along the path (same Zp mask / density clip) for comparison
        raw_zp = CubeWindow(raw_cubes["zp"]["zp"], affine, traj, 2, 5).sample()
        raw_zp[(raw_zp < ZP_VALID[0]) | (raw_zp > ZP_VALID[1])] = np.nan
        raw_rho = np.clip(CubeWindow(raw_cubes["rho"]["rho"], affine, traj, 2, 5).sample(),
                          *RHO_CLIP)
    rows, bins = [], []
    for t in TARGETS:
        if t not in up or up[t].notna().sum() < 3:
            continue
        _, log = resample_to_time(traj.twt, up[t].to_numpy())
        for v in VARIANTS:
            name = t.lower() + v
            win = CubeWindow(props[name], affine, traj, margin_traces=2, margin_ms=5)
            _, pred = resample_to_time(traj.twt, win.sample(0.0))
            m = metrics(log, pred)
            tr = transforms[(t, v)]
            logm = tr.predict_from_logs(up["VP"], None, up["RHOB"])
            cut = tr.meta.get("apply_where_vsh_le")
            if cut is not None:  # two-step at log scale too: reservoir by the log-MARS VSH
                vsh_log = transforms[("VSH", v)].predict_from_logs(up["VP"], None, up["RHOB"])
                logm = np.where(vsh_log <= cut, logm, np.nan)
            _, logm = resample_to_time(traj.twt, logm)
            lm = metrics(log, logm)
            rr = {"r": np.nan}
            if raw_zp is not None:
                pr = np.clip(tr.predict_from_impedance(raw_zp, None, raw_rho), 0, 1)
                if cut is not None:
                    vr = transforms[("VSH", v)].predict_from_impedance(raw_zp, None, raw_rho)
                    pr = np.where(np.clip(vr, 0, 1) <= cut, pr, np.nan)
                rr = metrics(log, resample_to_time(traj.twt, pr)[1])
            bins.append(pd.DataFrame({"well": well, "zone": zone, "target": t,
                                      "variant": "Zp-only" if v else "full",
                                      "log": log, "pred": pred, "logmars": logm}))
            rows.append({"well": well, "zone": zone, "target": t,
                         "variant": "Zp-only" if v else "full", **m,
                         "r_logmars": lm["r"], "rmse_logmars": lm["rmse"],
                         "r_unsmoothed_cube": rr["r"],
                         "n_log_bins": int(np.isfinite(log).sum()), **near})
    return rows, bins


def pooled_metrics(b: pd.DataFrame) -> pd.DataFrame:
    """r / RMSE / bias over all wells' 1 ms bins together (robust when per-well n is small)."""
    out = []
    for (z, t, v), g in b.groupby(["zone", "target", "variant"]):
        m = metrics(g["log"].to_numpy(), g["pred"].to_numpy())
        lm = metrics(g["log"].to_numpy(), g["logmars"].to_numpy())
        used = np.isfinite(g["pred"]) & np.isfinite(g["log"])
        out.append({"zone": z, "target": t, "variant": v, **m, "r_logmars": lm["r"],
                    "wells": int(g.loc[used, "well"].nunique())})
    return pd.DataFrame(out)


def smoothing_sensitivity(data, raw, affine, tr_all, welltops, wells, vsig,
                          lateral_m=(0, 25, 50, 100)) -> pd.DataFrame:
    """r at the wells of the full transforms applied to the cube smoothed with lateral sigma
    in ``lateral_m`` (metres) and the zone's vertical sigma off/on (CPU, sub-cubes)."""
    from mars.seismic_apply import SmoothingPlan, cpu_backend, iter_smoothed
    from mars.tie import sample_cube

    be, rows = cpu_backend(), []
    margin = int(np.ceil(4 * max(lateral_m) / 6.25)) + 2
    for well in wells:
        for zone in ZONES:
            try:
                z = load_well_zone(data, well, zone, require_vs=False)
            except (ValueError, KeyError):
                continue
            if len(z) < 10:
                continue
            tr = {t: tr_all[(zone, t, "")] for t in TARGETS}
            up = upscale_well(z, tr["PHIE"].meta["upscale_ft"])
            if up.empty:
                continue
            las = read_las(las_path(data, well))
            traj = trajectory(well_path_tops(data, well, welltops), up["DEPT"], las["DEPT"],
                              las[well_time_curve(las, well)])
            ok = np.isfinite(traj.twt)
            if ok.sum() < 10:
                continue
            up = up[ok].reset_index(drop=True)
            traj = Trajectory(traj.md[ok], traj.x[ok], traj.y[ok], traj.twt[ok])
            try:
                wz = CubeWindow(raw["zp"]["zp"], affine, traj, margin, 20)
                wr = CubeWindow(raw["rho"]["rho"], affine, traj, margin, 20)
            except ValueError:
                continue
            for lat in lateral_m:
                for vert in (False, True):
                    plan = SmoothingPlan(max(lat, 1e-9), {zone: vsig[zone] if vert else 0.0})
                    smoothed = list(iter_smoothed(be, wz.values.astype(np.float32),
                                                  wr.values.astype(np.float32), plan))
                    zs, rs = smoothed[0][1], smoothed[0][2]
                    zp = sample_cube(zs, wz.il0, wz.xl0, wz.t0, wz.dt, wz.fil, wz.fxl, traj.twt)
                    rho = sample_cube(rs, wz.il0, wz.xl0, wz.t0, wz.dt, wz.fil, wz.fxl, traj.twt)
                    for t in TARGETS:
                        if t not in up or up[t].notna().sum() < 5:
                            continue
                        pred = np.clip(tr[t].predict_from_impedance(zp, None, rho), 0, 1)
                        r = metrics(resample_to_time(traj.twt, up[t].to_numpy())[1],
                                    resample_to_time(traj.twt, pred)[1])["r"]
                        rows.append({"well": well, "zone": zone, "target": t,
                                     "lateral_sigma_m": lat, "vertical": vert, "r": r})
    return pd.DataFrame(rows)


def findings(pk: pd.DataFrame, sens: pd.DataFrame | None) -> list[str]:
    """Plain-language bullets: full vs Zp-only, log-to-seismic loss, bias, smoothing cost."""
    out = []
    for z in ZONES:
        for t in TARGETS:
            if (z, t, "full") not in pk.index or (z, t, "Zp-only") not in pk.index:
                continue
            f, o = pk.loc[(z, t, "full")], pk.loc[(z, t, "Zp-only")]
            better = "full" if f.r > o.r else "Zp-only"
            out.append(f"- **{z} {t}:** pooled seismic r is {fmt(f.r)} with the full transform "
                       f"and {fmt(o.r)} with Zp-only, so **{better}** is better at the wells. "
                       f"The full transform drops from log-MARS r {fmt(f.r_logmars)} to "
                       f"{fmt(f.r)} on the seismic, with bias {f.bias:+.3f}.")
    if sens is not None and len(sens):
        m = sens.groupby(["zone", "target", "lateral_sigma_m", "vertical"])["r"].median()
        ub = [(t, m.get(("UBRn", t, 0, True)), m.get(("UBRn", t, 50, True))) for t in TARGETS]
        out.append("- **Smoothing cost at the wells (UBRn, median r for lateral σ 0 → 50 m):** "
                   + ", ".join(f"{t} {fmt(a)} → {fmt(b)}" for t, a, b in ub) +
                   ". Vertical smoothing changes r by only a few hundredths.")
    out.append("- The full transforms lean on density (exp_mm(RHOB), Vp = Zp/ρ), and the "
               "single-realisation density cube carries ~0.03–0.05 g/cc of error even at the "
               "wells, which the rotation amplifies. The Zp-only transforms lose little between "
               "logs and seismic because the Zp cube is the better-resolved product.")
    return out


def fmt(v, p=2):
    return "" if pd.isna(v) else f"{v:.{p}f}"


def write_summary(df: pd.DataFrame, path: Path, transforms, pooled: pd.DataFrame,
                  sens: pd.DataFrame | None, attrs: dict):
    full = df[df.variant == "full"]
    med = df.groupby(["zone", "target", "variant"]).agg(
        wells=("r", lambda s: int(s.notna().sum())), r_median=("r", "median"),
        rmse_median=("rmse", "median"), bias_median=("bias", "median"),
        r_logmars_median=("r_logmars", "median"),
        r_unsmoothed_median=("r_unsmoothed_cube", "median")).reset_index()
    pk = pooled.set_index(["zone", "target", "variant"])
    vs = attrs.get("vertical_sigma_ms", {})
    lines = [
        "# Seismic MARS properties: QC at the wells (Phase 5 step 4)", "",
        "> ⚠ **Not a blind test.** The Zp/ρ cubes are one realisation of a stochastic inversion "
        "that very probably used these wells as conditioning data. The cubes tie the logs at the "
        "wells (median r(Zp) ≈ 0.96, r(ρ) ≈ 0.89–0.92) but lose that agreement within ~100 m "
        "(`results/tie/summary.md`). The numbers below check the chain (horizons → zones → "
        "smoothing → transforms) and show how the cube errors propagate. They are **not** a "
        "measure of predictive skill between wells.", "",
        "## Method", "",
        f"- **Cube preparation:** Zp outside [{ZP_VALID[0]:,.0f}, {ZP_VALID[1]:,.0f}] ft/s·g/cc "
        f"is set to NaN; the 22 well logs span 11,603–59,948. ρ is clipped to "
        f"[{RHO_CLIP[0]}, {RHO_CLIP[1]}] g/cc. Smoothing is a NaN-aware Gaussian with lateral "
        f"σ = {attrs.get('lateral_sigma_m')} m (2 inline × 8 crossline traces). The vertical σ "
        f"is UBRn {vs.get('UBRn', float('nan')):.2f} ms and LBRc "
        f"{vs.get('LBRc', float('nan')):.2f} ms: the variance of the 25 / 50 ft upscaling boxcar "
        "at the zones' median log Vp.",
        "- **Zones:** from `data/interim/props/surfaces.zarr` (see `horizons.md`).",
        "- **Transforms:** the final Vs-free transforms. LBRc PHIE is two-step: it is NaN where "
        "the predicted LBRc VSH > 0.49, so it is scored only on predicted reservoir. Predictions "
        "are clipped to [0, 1].",
        "- **Comparison:** predictions are sampled along each well path (bilinear, no time "
        "shift) and compared on 1 ms TWT bins with the logs upscaled as in the transforms.",
        "- **Reference columns:** **log MARS** = the transform applied to the upscaled logs (what "
        "a perfect cube would give); **unsmoothed** = the transform applied to the raw cube "
        "along the path; **pooled** = all wells' bins together. Per-well LBRc samples are few "
        "(about 27 bins; PHIE about 10 on reservoir only), so per-well r is noisy.",
        "- **SWT caveat:** the SWT transforms exclude RONALDO (different saturation vintage).",
        "", "## Findings (generated from the tables below)", "", *findings(pk, sens), "",
        "## Summary (median over wells; pooled = all bins together)", "",
        "| zone | target | variant | wells | r seismic (median) | r seismic (pooled) | "
        "RMSE (median) | bias | r unsmoothed cube | r log MARS |",
        "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in med.iterrows():
        pr = pk.loc[(r.zone, r.target, r.variant), "r"] if (r.zone, r.target, r.variant) in pk.index \
            else np.nan
        lines.append(f"| {r.zone} | {r.target} | {r.variant} | {r.wells} | {fmt(r.r_median)} | "
                     f"{fmt(pr)} | {fmt(r.rmse_median, 3)} | {fmt(r.bias_median, 3)} | "
                     f"{fmt(r.r_unsmoothed_median)} | {fmt(r.r_logmars_median)} |")
    lines += ["", "## Full vs Zp-only (seismic, median / pooled r)", "",
              "| zone | target | full | Zp-only | transform blind r at log scale (full / Zp-only) |",
              "|---|---|---|---|---|"]
    for (z, t), g in df.groupby(["zone", "target"]):
        cell = []
        for v in ("full", "Zp-only"):
            pr = pk.loc[(z, t, v), "r"] if (z, t, v) in pk.index else np.nan
            cell.append(f"{fmt(g.loc[g.variant == v, 'r'].median())} / {fmt(pr)}")
        bf = transforms[(z, t, "")].meta.get("blind_pooled", {}).get("r", np.nan)
        bz = transforms[(z, t, "_zponly")].meta.get("blind_pooled", {}).get("r", np.nan)
        lines.append(f"| {z} | {t} | {cell[0]} | {cell[1]} | {fmt(bf)} / {fmt(bz)} |")
    if sens is not None and len(sens):
        piv = sens.pivot_table(index=["zone", "target"], columns=["lateral_sigma_m", "vertical"],
                               values="r", aggfunc="median")
        cols = list(piv.columns)
        lines += ["", "## Smoothing sensitivity at the wells (full transforms, median r)", "",
                  "Lateral σ in m; V = with the zone's vertical σ. At the wells, smoothing can only "
                  "dilute a cube that honours the well locally, so this table cannot pick the best "
                  "σ *between* wells. It shows what the smoothing costs at the wells.", "",
                  "| zone | target | " + " | ".join(f"{int(a)} m{' V' if b else ''}"
                                                   for a, b in cols) + " |",
                  "|---|---|" + "---|" * len(cols)]
        for (z, t), r in piv.iterrows():
            lines.append(f"| {z} | {t} | " + " | ".join(fmt(r[c]) for c in cols) + " |")
    npz = attrs.get("nonphysical_samples_per_zone", {})
    if npz:
        lines += ["", "## Non-physical inversion values", "",
                  "| zone | live samples | Zp masked | ρ clipped |", "|---|---|---|---|"]
        for z, c in npz.items():
            lines.append(f"| {z} | {c['live']:,} | {c['zp_masked']:,} "
                         f"({100 * c['zp_masked'] / max(c['live'], 1):.3f}%) | "
                         f"{c['rho_clipped']:,} ({100 * c['rho_clipped'] / max(c['live'], 1):.3f}%) |")
        if "near_zp_n" in df:
            nw = df.drop_duplicates(["well", "zone"]).groupby("zone")[
                ["near_zp_n", "near_zp_masked", "near_rho_n", "near_rho_clipped"]].sum()
            lines += ["", "Within 2 traces of the well paths (zone time range):", ""]
            for z, c in nw.iterrows():
                lines.append(f"- {z}: Zp masked {int(c.near_zp_masked)} of {int(c.near_zp_n):,}, "
                             f"ρ clipped {int(c.near_rho_clipped)} of {int(c.near_rho_n):,}")
    for zone in ZONES:
        lines += ["", f"## {zone}: per well (full transforms; r / RMSE; n = 1 ms bins)", "",
                  "| well | " + " | ".join(f"{t} r | {t} RMSE" for t in TARGETS) +
                  " | n | PHIE r Zp-only | VSH r Zp-only | SWT r Zp-only |",
                  "|---|" + "---|" * (2 * len(TARGETS) + 4)]
        g = full[full.zone == zone]
        zo = df[(df.zone == zone) & (df.variant == "Zp-only")]
        for w in sorted(g.well.unique(), key=codename):
            cells = []
            for t in TARGETS:
                r = g[(g.well == w) & (g.target == t)]
                cells += [fmt(r.r.iloc[0]) if len(r) else "",
                          fmt(r.rmse.iloc[0], 3) if len(r) else ""]
            n = g[(g.well == w) & (g.target == "VSH")]["n"]
            zp = [zo[(zo.well == w) & (zo.target == t)]["r"] for t in TARGETS]
            lines.append(f"| {codename(w)} | " + " | ".join(cells) +
                         f" | {int(n.iloc[0]) if len(n) else ''} | " +
                         " | ".join(fmt(s.iloc[0]) if len(s) else "" for s in zp) + " |")
    missing = sorted(w for w, g in df.groupby("well") if (g["n"] == 0).all())
    if missing:
        lines += ["", "Wells without cube samples along their zone paths: " +
                  ", ".join(codename(w) for w in missing) +
                  " (the Baturaja part of the path lies north of the survey)."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path,
                    default=paths.RAW)
    ap.add_argument("--props", type=Path, default=paths.PROPS / "props.zarr")
    ap.add_argument("--cube", type=Path, default=paths.CUBES / "zp.zarr")
    ap.add_argument("--transforms", type=Path, default=paths.TRANSFORMS)
    ap.add_argument("--out", type=Path, default=paths.RESULTS / "seismic_qc")
    ap.add_argument("--no-sensitivity", action="store_true",
                    help="skip the lateral/vertical smoothing sensitivity at the wells")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    props = xr.open_zarr(args.props)
    affine = xr.open_zarr(args.cube).attrs["affine_ilxl_to_xy"]
    welltops = read_welltops(args.data / "tops" / "welltops.csv")
    raw = {c: xr.open_zarr(args.cube.parent / f"{c}.zarr") for c in ("zp", "rho")}
    tr_all = {(z, t, v): MarsTransform.from_json(args.transforms / f"{z}_{t}{v}.json")
              for z in ZONES for t in TARGETS for v in VARIANTS}
    wells = [dataset.well_id(f.stem) for f in sorted((args.data / "wells").glob("*.las"))]
    rows, bins = [], []
    for well in wells:
        for zone in ZONES:
            tr = {(t, v): tr_all[(zone, t, v)] for t in TARGETS for v in VARIANTS}
            r, b = qc_well(well, zone, args.data, props, affine, tr, welltops, raw)
            rows += r
            bins += b
            if r:
                print(well, zone, json.dumps({f"{x['target']}{'/zp' if x['variant'] != 'full' else ''}":
                                              round(x["r"], 2) for x in r}), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "qc_table.csv", index=False, float_format="%.4f")
    pooled = pooled_metrics(pd.concat(bins, ignore_index=True))
    pooled.to_csv(args.out / "qc_pooled.csv", index=False, float_format="%.4f")
    sens = None
    if not args.no_sensitivity:
        sens = smoothing_sensitivity(args.data, raw, affine, tr_all, welltops, wells,
                                     dict(props.attrs["vertical_sigma_ms"]))
        sens.to_csv(args.out / "smoothing_sensitivity.csv", index=False, float_format="%.4f")
    write_summary(df, args.out / "summary.md", tr_all, pooled, sens, props.attrs)


if __name__ == "__main__":
    main()
