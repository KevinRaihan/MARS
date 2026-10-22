"""Phase 4 reservoir-rock rule: LOWO blind r on all samples vs reservoir rock only (22 wells).

The eastern wells' LBR is a shaly non-reservoir facies with nearly constant PHIE, so pooled
PHIE r there means "no signal". This compares training/scoring on all samples with samples that
pass the GGRP net cut-off (UBR Vsh <= 0.60, LBR Vsh <= 0.49); SWT excludes RONALDO.
It motivated the two-step LBR rule (predict VSH -> mask -> predict PHIE).

Usage:  python scripts/run_reservoir_rule.py
Output: results/phase4/reservoir_cutoff.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from run_phase4 import NEW10, THESIS12, VSFREE, load_zone  # noqa: E402

from mars import dataset, paths  # noqa: E402
from mars.validation import MarsModel, lowo_cv  # noqa: E402

OUT = paths.RESULTS / "phase4" / "reservoir_cutoff.txt"


def main() -> None:
    lines = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    df, _ = load_zone(THESIS12 + NEW10, "LBRc", 50)
    d = df.dropna(subset=["PHIE", "VSH"])
    log("LBRc PHIE std per well (all) and reservoir fraction (VSH <= 0.49):")
    per = d.groupby("WELL").apply(lambda g: pd.Series({"std_all": g.PHIE.std(),
                                                       "frac_res": (g.VSH <= 0.49).mean()}),
                                  include_groups=False)
    log(per.round(3).T.to_string())
    for label, sub in (("all LBRc samples", d), ("reservoir only (VSH<=0.49)", d[d.VSH <= 0.49])):
        for nd in (2, 3):
            r = lowo_cv(sub, "PHIE", lambda nd=nd: MarsModel(nd=nd, base=VSFREE))
            folds = {x["held_out"]: round(x["r"], 2) for x in r["folds"]}
            log(f"{label:28s} {nd}D wells={sub.WELL.nunique()} blind r={r['blind_pooled']['r']:.3f} "
                f"rmse={r['blind_pooled']['rmse']:.4f} median={r['blind_r_median']:.3f} "
                f"[{' | '.join(r['selected_all_data'])}]")
            log(f"    per-well: {folds}")
    for zone, L, cut in (("UBRn", 25, 0.60), ("LBRc", 50, 0.49)):
        df, _ = load_zone(THESIS12 + NEW10, zone, L)
        for target in ("PHIE", "SWT"):
            d = df.dropna(subset=[target, "VSH"])
            if target == "SWT":
                d = d[~d.WELL.isin(dataset.wells("swt_exclude"))]
            for label, sub in (("all", d), (f"VSH<={cut}", d[d.VSH <= cut])):
                res = {nd: lowo_cv(sub, target, lambda nd=nd: MarsModel(nd=nd, base=VSFREE))
                       for nd in (2, 3)}
                log(f"{zone} {target:4s} {label:9s} wells={sub.WELL.nunique():2d} n={len(sub):5d}"
                    f"  2D={res[2]['blind_pooled']['r']:.3f} "
                    f"(rmse {res[2]['blind_pooled']['rmse']:.3f})"
                    f"  3D={res[3]['blind_pooled']['r']:.3f} "
                    f"(rmse {res[3]['blind_pooled']['rmse']:.3f})")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
