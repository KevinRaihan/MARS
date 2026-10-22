"""Phase 4: more wells. Coverage inventory + leave-one-well-out comparison, 12 vs 22 wells.

Vs-free MARS (IP, VP, RHOB) with the Phase 3 configuration: UBR (named markers) upscaled
25 ft, LBR reservoir without BRF (LBRc) upscaled 50 ft; SWT excludes RONALDO (post-production).

Usage:  python scripts/run_phase4.py
Outputs: results/phase4/{inventory.csv, lowo_compare.csv, per_well_blind.csv, summary.md}
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mars import dataset, paths  # noqa: E402
from mars.conditioning import load_well_zone, upscale_well, well_group  # noqa: E402
from mars.validation import MarsModel, SklearnModel, lowo_cv  # noqa: E402

DATA = paths.RAW
OUT = paths.RESULTS / "phase4"
THESIS12 = dataset.wells("first12")
NEW10 = dataset.wells("added10")
ZONES = {"UBRn": 25, "LBRc": 50}
VSFREE = ("IP", "VP", "RHOB")
TARGETS = ("PHIE", "VSH", "SWT")


def well_group22(w: str) -> str:
    """Sidetracks/pilots with their parent; W-12st -> W-12, W-12p -> W-12."""
    return well_group(w)


def load_zone(wells, zone, L):
    frames, inv = [], []
    for w in wells:
        try:
            z = load_well_zone(DATA, w, zone, require_vs=False)
        except Exception as e:  # noqa: BLE001
            inv.append({"well": w, "zone": zone, "status": f"no zone: {str(e)[:60]}"})
            continue
        row = {"well": w, "zone": zone, "status": "ok", "n_raw": len(z),
               "thickness_ft": float(z.DEPT.max() - z.DEPT.min()) if len(z) else 0.0,
               **{f"cov_{t}": float(z[t].notna().mean()) if t in z and len(z) else 0.0 for t in TARGETS},
               "vs_present": float(z["VS"].notna().mean()) if len(z) else 0.0,
               "rho_med": float(z.RHOB.median()) if len(z) else np.nan,
               "vp_med": float(z.VP.median()) if len(z) else np.nan}
        if len(z) < 20:
            row["status"] = "too few samples"
            inv.append(row)
            continue
        u = upscale_well(z, L)
        row["n_upscaled"] = len(u)
        inv.append(row)
        frames.append(u)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), inv


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inv_all, rows, per_well = [], [], []
    for zone, L in ZONES.items():
        df, inv = load_zone(THESIS12 + NEW10, zone, L)
        inv_all += inv
        for target in TARGETS:
            base = df.dropna(subset=[target])
            if target == "SWT":
                base = base[~base.WELL.isin(dataset.wells("swt_exclude"))]
            for label, wells in (("12 thesis-era wells", THESIS12), ("22 wells", THESIS12 + NEW10)):
                d = base[base.WELL.isin(wells)]
                for name, mk in [("Vs-free 2D", lambda: MarsModel(nd=2, base=VSFREE)),
                                 ("Vs-free 3D", lambda: MarsModel(nd=3, base=VSFREE)),
                                 ("GBM Vs-free", lambda: SklearnModel("hgb", base=VSFREE))]:
                    r = lowo_cv(d, target, mk, group_fn=well_group22)
                    rows.append({"zone": zone, "target": target, "set": label, "model": name,
                                 "wells": d.WELL.nunique(), "groups": d.WELL.map(well_group22).nunique(),
                                 "r_in": r["insample"]["r"], "blind_pooled": r["blind_pooled"]["r"],
                                 "blind_median": r["blind_r_median"],
                                 "blind_rmse": r["blind_pooled"]["rmse"],
                                 "selected": " | ".join(r.get("selected_all_data") or [])})
                    if label == "22 wells" and name == "Vs-free 2D":
                        for f in r["folds"]:
                            per_well.append({"zone": zone, "target": target, **{k: f[k] for k in ("held_out", "n", "r", "rmse", "bias")},
                                             "new_well": f["held_out"] not in {well_group22(w) for w in THESIS12}})
                    x = rows[-1]
                    print(f"{zone} {target:4s} {label:20s} {name:12s} wells={x['wells']:2d} "
                          f"in={x['r_in']:.3f} blind={x['blind_pooled']:.3f} median={x['blind_median']:.3f} [{x['selected']}]",
                          flush=True)
    inv = pd.DataFrame(inv_all)
    inv.to_csv(OUT / "inventory.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "lowo_compare.csv", index=False)
    pw = pd.DataFrame(per_well)
    pw.to_csv(OUT / "per_well_blind.csv", index=False)
    piv = res.pivot_table(index=["zone", "target", "model"], columns="set", values="blind_pooled").round(3)
    md = ["# Phase 4: 12 vs 22 wells (Vs-free MARS, LOWO blind r pooled)", "", piv.to_markdown(), "",
          "## Inventory", "", inv.round(3).to_markdown(index=False), "",
          "## Per-well blind r (22 wells, Vs-free 2D)", "",
          pw.pivot_table(index="held_out", columns=["zone", "target"], values="r").round(2).to_markdown()]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", OUT / "summary.md")


if __name__ == "__main__":
    main()
