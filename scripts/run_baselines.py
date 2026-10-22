"""MARS against baseline regressions on the 22-well data (leave-one-well-out, nested selection).

Same data and configuration as scripts/fit_final_transforms.py (Vs-free base IP, VP, RHOB;
UBRn 25 ft, LBRc 50 ft; SWT without RONALDO; LBRc PHIE on reservoir rock VSH <= 0.49). Models:
best single attribute (MARS-1D), MARS-2D, MARS-3D, OLS and LASSO on the three base attributes,
and gradient-boosted trees (GBM). Reports in-sample r next to blind r, to show how the gap grows
with model freedom.

Usage:  python scripts/run_baselines.py
Outputs: results/baselines/{baselines.csv, summary.md}
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from fit_final_transforms import RESERVOIR_ONLY, TARGETS, VSFREE, load  # noqa: E402

from mars import dataset, paths  # noqa: E402
from mars.validation import MarsModel, SklearnModel, lowo_cv  # noqa: E402

OUT = paths.RESULTS / "baselines"
ZONES = {"UBRn": 25, "LBRc": 50}
MODELS = {
    "Best single attribute": lambda: MarsModel(nd=1, base=VSFREE),
    "MARS-2D": lambda: MarsModel(nd=2, base=VSFREE),
    "MARS-3D": lambda: MarsModel(nd=3, base=VSFREE),
    "OLS": lambda: SklearnModel("ols", base=VSFREE),
    "LASSO": lambda: SklearnModel("lasso", base=VSFREE),
    "GBM": lambda: SklearnModel("hgb", base=VSFREE),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for zone, L in ZONES.items():
        df = load(zone, L)
        for target in TARGETS:
            d = df[~df.WELL.isin(dataset.wells("swt_exclude"))] if target == "SWT" else df
            cut = RESERVOIR_ONLY.get((zone, target))
            if cut is not None:
                d = d[d.VSH <= cut]
            for name, mk in MODELS.items():
                r = lowo_cv(d, target, mk)
                rows.append({"zone": zone, "target": target, "model": name,
                             "r_in": r["insample"]["r"], "blind_pooled": r["blind_pooled"]["r"],
                             "blind_median": r["blind_r_median"], "wells": d.WELL.nunique()})
                x = rows[-1]
                print(f"{zone} {target:4s} {name:22s} in={x['r_in']:.3f} blind={x['blind_pooled']:.3f} "
                      f"median={x['blind_median']:.3f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "baselines.csv", index=False)
    piv = res.pivot_table(index="model", columns=["zone", "target"], values=["r_in", "blind_pooled"],
                          sort=False).round(3)
    (OUT / "summary.md").write_text("# MARS vs baselines, 22 wells (LOWO)\n\n" + piv.to_markdown() + "\n",
                                    encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
