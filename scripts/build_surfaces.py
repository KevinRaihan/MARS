"""Grid the Petrel time horizons onto the cube grid and build the Baturaja zone boundaries.

Usage:
    python scripts/build_surfaces.py [--data data/raw] [--cube data/interim/cubes/zp.zarr]
        [--out data/interim/props/surfaces.zarr] [--report results/seismic_qc]
Each zone boundary is matched to the horizon that reproduces its well marker (median
|horizon - marker TWT| at the wells < 2 ms). A boundary without a horizon (LBR Cycle 3: BRF base,
LBRc top) is derived; the candidate with the smallest leave-one-well-out mismatch is kept.
Writes the surfaces (il, xl) to a small Zarr store, horizon_mismatch*.csv (real ids; results/ is
git-ignored) and horizons.md (codenames).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from mars import dataset, paths  # noqa: E402
from mars.horizons import (  # noqa: E402
    grid_horizon,
    idw,
    proportional_surface,
    read_xyt,
    sample_surface,
    well_marker_times,
)
from mars.io.segy import ilxl_to_xy  # noqa: E402
from mars.naming import codename  # noqa: E402

BOUNDARIES = {  # zone boundary -> well marker, top to base
    "top_UBR": "UBR Cycle 5 / Top UBR",
    "top_BRF": "LBR Cycle 4 / BRF Shale",
    "top_LBRc": "LBR Cycle 3",
    "top_TAF": "Talang Akar",
}
OTHER_MARKERS = ("Top LBR Lithostrat", "LBR Cycle 2", "LBR Cycle 1", "Gumai Carb Cycle 2")


def wells_in(data: Path) -> list[str]:
    return [dataset.well_id(f.stem) for f in sorted((data / "wells").glob("*.las"))]


def derive_boundary(mt, marker, candidates, aff, il, xl):
    """Evaluate candidate derivations of a boundary at the wells (leave-one-well-out IDW).

    ``candidates``: label -> (top grid, base grid or None). With a base grid the well value is
    the isochron fraction, otherwise the thickness below ``top``.
    """
    w = mt[mt.marker == marker].reset_index(drop=True)
    xy = w[["x", "y"]].to_numpy()
    out = {}
    for label, (top, base) in candidates.items():
        t_top = sample_surface(top, aff, il, xl, w.x, w.y)
        t_base = None if base is None else sample_surface(base, aff, il, xl, w.x, w.y)
        val = w.twt.to_numpy() - t_top if base is None else (w.twt.to_numpy() - t_top) / (t_base - t_top)
        loo = np.full(len(w), np.nan)
        for i in range(len(w)):
            keep = np.arange(len(w)) != i
            v = idw(xy[keep], val[keep], xy[i:i + 1])[0]
            pred = t_top[i] + v if base is None else t_top[i] + np.clip(v, 0, 1) * (t_base[i] - t_top[i])
            loo[i] = pred - w.twt.iloc[i]
        out[label] = {"wells": w.well.tolist(), "xy": xy, "values": val, "loo": loo,
                      "loo_mae": float(np.nanmean(np.abs(loo))),
                      "loo_max": float(np.nanmax(np.abs(loo))),
                      "value_median": float(np.nanmedian(val))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path,
                    default=paths.RAW)
    ap.add_argument("--cube", type=Path, default=paths.CUBES / "zp.zarr")
    ap.add_argument("--out", type=Path,
                    default=paths.PROPS / "surfaces.zarr")
    ap.add_argument("--report", type=Path, default=paths.RESULTS / "seismic_qc")
    args = ap.parse_args()
    args.report.mkdir(parents=True, exist_ok=True)

    cube = xr.open_zarr(args.cube)
    aff = cube.attrs["affine_ilxl_to_xy"]
    il, xl = cube["il"].values, cube["xl"].values

    grids, info = {}, {}
    for f in sorted((args.data / "horizons").glob("*.xyt")):
        name = f.name.replace("-Time.xyt", "")
        grids[name], info[name] = grid_horizon(read_xyt(f), aff, il, xl)
        print(name, info[name], flush=True)

    mt = well_marker_times(args.data, wells_in(args.data), [*BOUNDARIES.values(), *OTHER_MARKERS])
    for name, g in grids.items():
        mt[name] = sample_surface(g, aff, il, xl, mt["x"], mt["y"]) - mt["twt"]
    mt.to_csv(args.report / "horizon_marker_mismatch_all.csv", index=False, float_format="%.2f")

    match, scores = {}, {}
    for b, m in BOUNDARIES.items():
        sub = mt[mt.marker == m]
        scores[b] = {h: float(np.nanmedian(np.abs(sub[h]))) for h in grids}
        best = min(scores[b], key=scores[b].get)
        if scores[b][best] < 2.0:
            match[b] = best
    print("matched", match)

    surf = {b: grids[h] for b, h in match.items()}
    derivation = {}
    for b in [b for b in BOUNDARIES if b not in match]:
        if b != "top_LBRc":
            raise RuntimeError(f"no horizon for {b} and no derivation rule")
        cands = {
            "isochron fraction between top_BRF and top_TAF": (surf["top_BRF"], surf["top_TAF"]),
            "isochron fraction between top_BRF and Base LBR (= Top LBR Lithostrat)":
                (surf["top_BRF"], grids["Base LBR"]),
            "thickness below top_BRF": (surf["top_BRF"], None),
        }
        res = derive_boundary(mt, BOUNDARIES[b], cands, aff, il, xl)
        for k, v in res.items():
            print(f"{b} via {k}: LOO MAE {v['loo_mae']:.2f} ms, max {v['loo_max']:.2f} ms")
        best = min(res, key=lambda k: res[k]["loo_mae"])
        top, base = cands[best]
        gx, gy = ilxl_to_xy(aff, *np.meshgrid(il, xl, indexing="ij"))
        field = idw(res[best]["xy"], res[best]["values"],
                    np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        s = top + field if base is None else proportional_surface(top, base, field)
        surf[b] = np.clip(s, surf["top_BRF"], surf["top_TAF"])
        derivation[b] = {"method": best, "interpolation": "inverse-distance weighting, power 2",
                         "n_wells": len(res[best]["wells"]),
                         "value_median": round(res[best]["value_median"], 3),
                         "candidates": {k: {"loo_mae_ms": round(v["loo_mae"], 2),
                                            "loo_max_ms": round(v["loo_max"], 2)}
                                        for k, v in res.items()},
                         "loo": dict(zip(res[best]["wells"], res[best]["loo"].tolist(), strict=True))}

    rows = []
    for b, m in BOUNDARIES.items():
        sub = mt[mt.marker == m]
        if b in derivation:
            d = [derivation[b]["loo"].get(w, np.nan) for w in sub.well]
        else:
            d = sample_surface(surf[b], aff, il, xl, sub.x, sub.y) - sub.twt.to_numpy()
        rows += [{"boundary": b, "well": w, "mismatch_ms": float(v)}
                 for w, v in zip(sub.well, d, strict=True)]
    st = pd.DataFrame(rows)
    st.to_csv(args.report / "horizon_mismatch.csv", index=False, float_format="%.2f")

    for b in derivation:
        derivation[b].pop("loo")
    ds = xr.Dataset(
        {b: (("il", "xl"), surf[b].astype("float32")) for b in BOUNDARIES}
        | {"horizon_" + h.replace(" ", "_"): (("il", "xl"), g.astype("float32"))
           for h, g in grids.items()},
        coords={"il": il, "xl": xl},
        attrs={"units": "ms TWT", "boundaries": json.dumps(BOUNDARIES),
               "horizon_for_boundary": json.dumps(match), "derivation": json.dumps(derivation),
               "gridding": json.dumps(info)},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(args.out, mode="w", consolidated=True)

    summ = st.groupby("boundary")["mismatch_ms"].agg(
        n="count", median="median", mae=lambda v: np.nanmean(np.abs(v)),
        max_abs=lambda v: np.nanmax(np.abs(v)))
    print(summ.round(2))
    write_report(args.report / "horizons.md", match, derivation, scores, summ, st)


def write_report(path, match, derivation, scores, summ, st):
    lines = [
        "# Horizons and zone boundaries (Phase 5 step 1)", "",
        "Petrel time horizons (`data/raw/horizons/*.xyt`) were gridded onto the cube IL/XL grid. "
        "Every node lies on a bin centre (offset ≤ 0.034 bins), so the nodes are snapped and "
        "coverage is 100%. Marker TWT = 2 × LAS one-way time at the marker MD, at the marker "
        "X/Y. Built by `scripts/build_surfaces.py` → `data/interim/props/surfaces.zarr`.", "",
        "| boundary | well marker | source | n wells | median ms | MAE ms | max abs ms |",
        "|---|---|---|---|---|---|---|"]
    for b, m in BOUNDARIES.items():
        src = f"horizon `{match[b]}`" if b in match else f"derived: {derivation[b]['method']}"
        r = summ.loc[b]
        lines.append(f"| {b} | {m} | {src} | {int(r.n)} | {r['median']:+.2f} | {r.mae:.2f} | "
                     f"{r.max_abs:.2f} |")
    lines += ["", "Median |horizon − marker| (ms) of the best three horizons per boundary:", ""]
    for b, sc in scores.items():
        top3 = sorted(sc.items(), key=lambda kv: kv[1])[:3]
        lines.append(f"- {b}: " + ", ".join(f"`{h}` {v:.1f}" for h, v in top3))
    for b, d in derivation.items():
        lines += ["", f"**{b}** has no horizon. Candidates: the value at each well (isochron "
                  "fraction or thickness) is interpolated by IDW (power 2) and checked leave one "
                  "well out:", ""]
        lines += [f"- {k}: LOO MAE {v['loo_mae_ms']} ms, max {v['loo_max_ms']} ms"
                  for k, v in d["candidates"].items()]
        lines += ["", f"Chosen: **{d['method']}** (median value {d['value_median']}). The surface "
                  "honours every well exactly, so the table reports its leave-one-well-out "
                  "mismatch. It is clipped to lie between top_BRF and top_TAF."]
    lines += ["", "Other horizons: `Top GC C2` is above Top UBR (Gumai carbonate), `Base LBR` "
              "reproduces **Top LBR Lithostrat** (not LBR Cycle 3), and `Top BSMT` is basement.",
              "", "## Mismatch per well (ms; derived boundary = leave-one-well-out)", "",
              "| well | " + " | ".join(BOUNDARIES) + " |", "|---|" + "---|" * len(BOUNDARIES)]
    piv = st.pivot(index="well", columns="boundary", values="mismatch_ms")
    for w, r in piv.iterrows():
        cells = ["" if pd.isna(r.get(b)) else f"{r[b]:+.1f}" for b in BOUNDARIES]
        lines.append(f"| {codename(w)} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
