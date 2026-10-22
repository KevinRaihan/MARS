"""Spatial blind test: hold out a whole region of wells once (frozen sets), fit on the rest.

Same configuration as scripts/fit_final_transforms.py (Vs-free library, UBRn 25 ft / LBRc 50 ft,
SWT without RONALDO, ND chosen by LOWO on the *training* wells only, LBRc PHIE trained and scored on
reservoir rock VSH <= 0.49), plus the Zp-only control. Each holdout set is predicted exactly once.

Holdout sets (frozen in results/spatial_blind/holdout.json):
  EAST   the eastern block (shaly LBR facies): tests extrapolation across a facies change.
  SOUTH  the southern edge of the western platform (same facies, outside the training cluster).

Usage:  python scripts/run_spatial_blind.py
Outputs: results/spatial_blind/{holdout.json, spatial_blind.csv, per_well.csv, summary.md}
         (summary/per-well use codenames; the CSV with real ids stays local)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fit_final_transforms import (  # noqa: E402
    MIN_GAIN,
    RESERVOIR_ONLY,
    TARGETS,
    VSFREE,
    WELLS,
    zp_only,
)

from mars import dataset, paths  # noqa: E402
from mars.conditioning import load_well_zone, upscale_well  # noqa: E402
from mars.naming import codename  # noqa: E402
from mars.validation import MarsModel, lowo_cv, metrics  # noqa: E402

DATA = paths.RAW
OUT = paths.RESULTS / "spatial_blind"
ZONES = {"UBRn": 25, "LBRc": 50}
HOLDOUT = {
    "EAST": list(dataset.wells("holdout_EAST")),
    "SOUTH": list(dataset.wells("holdout_SOUTH")),
}


def load(zone: str, length: float) -> pd.DataFrame:
    frames = []
    for w in WELLS:
        try:
            frames.append(upscale_well(load_well_zone(DATA, w, zone, require_vs=False), length))
        except Exception:  # noqa: BLE001 - well lacks the zone markers
            pass
    return pd.concat(frames, ignore_index=True)


def choose_nd(d: pd.DataFrame, target: str, make, nds) -> tuple[int, dict]:
    """Smallest ND unless a higher one wins LOWO blind r by > MIN_GAIN (training wells only)."""
    res = {nd: lowo_cv(d, target, lambda nd=nd: make(nd)) for nd in nds}
    best = nds[0]
    for nd in nds[1:]:
        if res[nd]["blind_pooled"]["r"] > res[best]["blind_pooled"]["r"] + MIN_GAIN:
            best = nd
    return best, res[best]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = OUT / "holdout.json"
    if frozen.exists() and json.loads(frozen.read_text()) != HOLDOUT:
        raise SystemExit("holdout.json differs from HOLDOUT: the sets are frozen, do not edit them")
    frozen.write_text(json.dumps(HOLDOUT, indent=1))
    rows, per_well = [], []
    for zone, L in ZONES.items():
        df = load(zone, L)
        for target in TARGETS:
            d = df[~df.WELL.isin(dataset.wells("swt_exclude"))] if target == "SWT" else df
            cut = RESERVOIR_ONLY.get((zone, target))
            if cut is not None:
                d = d[d.VSH <= cut]
            d = d.dropna(subset=[target])
            for name, hold in HOLDOUT.items():
                train, test = d[~d.WELL.isin(hold)], d[d.WELL.isin(hold)]
                base = {"zone": zone, "target": target, "holdout": name,
                        "train_wells": train.WELL.nunique(), "test_wells": test.WELL.nunique(),
                        "n_test": len(test)}
                if test.WELL.nunique() == 0:
                    rows.append({**base, "note": "no test samples (e.g. no reservoir rock)"})
                    continue
                for variant, make, nds in (
                        ("full", lambda nd: MarsModel(nd=nd, base=VSFREE), (1, 2, 3)),
                        ("Zp-only", zp_only, (1, 2))):
                    nd, cv = choose_nd(train, target, make, nds)
                    model = make(nd).fit(train, target)
                    pred = model.predict(test)
                    m = metrics(test[target].to_numpy(), pred)
                    rows.append({**base, "variant": variant, "nd": nd,
                                 "attrs": " | ".join(model.transform_.attrs),
                                 "r": m["r"], "rmse": m["rmse"], "bias": m.get("bias", np.nan),
                                 "train_lowo_r": cv["blind_pooled"]["r"]})
                    for w, g in test.assign(pred=pred).groupby("WELL"):
                        if len(g) >= 5:
                            mw = metrics(g[target].to_numpy(), g["pred"].to_numpy())
                            per_well.append({**base, "variant": variant, "well": codename(w),
                                             "n": len(g), "r": mw["r"], "rmse": mw["rmse"]})
                    print(f"{zone:5s} {target:4s} {name:5s} {variant:7s} nd={nd} "
                          f"r={m['r']:.3f} rmse={m['rmse']:.3f} (train LOWO "
                          f"{cv['blind_pooled']['r']:.3f})", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "spatial_blind.csv", index=False, float_format="%.4f")
    pw = pd.DataFrame(per_well)
    pw.to_csv(OUT / "per_well.csv", index=False, float_format="%.4f")
    lowo = pd.read_csv(paths.TRANSFORMS / "summary.csv")
    lowo = lowo.set_index(["zone", "target"])["blind_r"]
    full = res[res.get("variant") == "full"].copy()
    full["lowo_r_22wells"] = [lowo.get((z, t), np.nan) for z, t in zip(full.zone, full.target,
                                                                         strict=True)]
    zp = res[res.get("variant") == "Zp-only"].set_index(["zone", "target", "holdout"])["r"]
    full["zp_only_r"] = [zp.get(k, np.nan) for k in zip(full.zone, full.target, full.holdout,
                                                          strict=True)]
    cols = ["zone", "target", "holdout", "train_wells", "test_wells", "n_test", "nd", "r",
            "rmse", "bias", "zp_only_r", "train_lowo_r", "lowo_r_22wells"]
    hold_txt = "; ".join(f"**{k}**: {', '.join(codename(w) for w in v)}"
                         for k, v in HOLDOUT.items())
    md = ["# Spatial blind test (frozen holdout regions)", "",
          f"Holdout sets, predicted once: {hold_txt}.",
          "Configuration as the final transforms; ND chosen by LOWO on the training wells only. "
          "LBRc PHIE is trained and scored on reservoir rock (log VSH ≤ 0.49); SWT excludes "
          "RONALDO. `r` = pooled over all held-out samples; `lowo_r_22wells` = leave-one-well-out "
          "blind r of the final transform for comparison.", "",
          full[cols].round(3).to_markdown(index=False), ""]
    skipped = res[res.get("variant").isna()] if "variant" in res else res.iloc[:0]
    if len(skipped):
        md += ["Not scored: " + "; ".join(f"{r.zone} {r.target} {r.holdout} ({r.note})"
                                          for r in skipped.itertuples()), ""]
    md += ["## Per well (full transform)", "",
           pw[pw.variant == "full"].drop(columns=["variant", "train_wells", "test_wells",
                                                  "n_test"]).round(3).to_markdown(index=False)]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", OUT / "summary.md")


if __name__ == "__main__":
    main()
