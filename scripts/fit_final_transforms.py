"""Phase 3/4 deliverable: fit and save the chosen MARS transforms for seismic application.

Configuration (from vault/06-Plans/phase3-findings.md):
  * Vs-free library (IP, VP = Zp/rho, RHOB) -- the predicted-Vs logs are not trusted;
  * zones: UBR (upscaled 25 ft), LBRc reservoir without BRF shale (50 ft), BRF (50 ft);
  * SWT fitted without RONALDO (logged post-production, 2021);
  * dimension chosen by leave-one-well-out blind r (1D/2D/3D; a higher ND must win by > 0.02);
  * a Zp-only control (base IP) is fitted next to each transform.

Usage:  python scripts/fit_final_transforms.py
Outputs: results/transforms/<zone>_<target>.json (+ _zponly.json) and results/transforms/summary.md
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from mars import dataset, paths  # noqa: E402
from mars.conditioning import load_well_zone, upscale_well  # noqa: E402
from mars.validation import MarsModel, lowo_cv  # noqa: E402

DATA = paths.RAW
OUT = paths.TRANSFORMS
WELLS = dataset.wells("all")  # first 12 + the 10 Phase 4 additions
VSFREE = ("IP", "VP", "RHOB")
ZONES = {"UBRn": 25, "LBRc": 50, "BRF": 50}
# Train only on reservoir-quality rock (GGRP net cutoffs); apply where predicted VSH passes it.
RESERVOIR_ONLY = {("LBRc", "PHIE"): 0.49}
TARGETS = ("PHIE", "VSH", "SWT")
MIN_GAIN = 0.02


def load(zone: str, length: float) -> pd.DataFrame:
    frames = []
    for w in WELLS:
        try:
            frames.append(upscale_well(load_well_zone(DATA, w, zone, require_vs=False), length))
        except Exception as e:  # noqa: BLE001 - well lacks the zone markers
            print(f"  skip {w} {zone}: {e}")
    return pd.concat(frames, ignore_index=True)


def zp_only(nd: int) -> MarsModel:
    """Control: P-impedance only (same-base pairs such as IP + ln(IP) capture its non-linearity)."""
    return MarsModel(nd=nd, base=("IP",), same_base_allowed=True)


def stability(res: dict) -> float:
    """Fraction of CV folds that selected exactly the attributes chosen on all data."""
    final = set(res["selected_all_data"])
    folds = [set(f["selected"]) for f in res["folds"] if f["selected"]]
    return sum(f == final for f in folds) / max(len(folds), 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for zone, L in ZONES.items():
        df = load(zone, L)
        for target in TARGETS:
            d = df[~df.WELL.isin(dataset.wells("swt_exclude"))] if target == "SWT" else df
            vsh_cut = RESERVOIR_ONLY.get((zone, target))
            if vsh_cut is not None:
                d = d[d.VSH <= vsh_cut]
            cands = {nd: lowo_cv(d, target, lambda nd=nd: MarsModel(nd=nd, base=VSFREE)) for nd in (1, 2, 3)}
            best_nd = 1
            for nd in (2, 3):
                if cands[nd]["blind_pooled"]["r"] > cands[best_nd]["blind_pooled"]["r"] + MIN_GAIN:
                    best_nd = nd
            ctrls = {k: lowo_cv(d, target, lambda k=k: zp_only(k)) for k in (1, 2)}
            ctrl_nd = 2 if ctrls[2]["blind_pooled"]["r"] > ctrls[1]["blind_pooled"]["r"] + MIN_GAIN else 1
            ctrl = ctrls[ctrl_nd]
            chosen = cands[best_nd]
            for tag, res, base, nd in (("", chosen, VSFREE, best_nd), ("_zponly", ctrl, ("IP",), ctrl_nd)):
                model = (zp_only(nd) if tag else MarsModel(nd=nd, base=base)).fit(d, target)
                t = model.transform_
                t.meta = {"zone": zone, "upscale_ft": L, "wells": sorted(d.WELL.unique()),
                          "n_samples": int(d[target].notna().sum()), "nd": nd, "base": list(base),
                          "blind_pooled": res["blind_pooled"], "blind_r_median": res["blind_r_median"],
                          "insample": res["insample"], "selection_stability": stability(res),
                          "folds": res["folds"], "excluded": list(dataset.wells("swt_exclude")) if target == "SWT" else [],
                          "apply_where_vsh_le": vsh_cut,
                          "note": "apply to laterally smoothed cubes; clip rho to [1.8, 3.0] g/cc"}
                t.to_json(OUT / f"{zone}_{target}{tag}.json")
            rows.append({"zone": zone, "L_ft": L, "target": target, "nd": best_nd,
                         "attrs": " | ".join(chosen["selected_all_data"]),
                         "blind_r": chosen["blind_pooled"]["r"], "blind_r_median": chosen["blind_r_median"],
                         "blind_rmse": chosen["blind_pooled"]["rmse"], "r_in": chosen["insample"]["r"],
                         "stability": stability(chosen),
                         "zp_only_blind_r": ctrl["blind_pooled"]["r"], "wells": d.WELL.nunique()})
            print(f"{zone:5s} {target:4s} nd={best_nd} blind r={rows[-1]['blind_r']:.3f} "
                  f"(Zp-only {rows[-1]['zp_only_blind_r']:.3f}) stability={rows[-1]['stability']:.0%}  "
                  f"[{rows[-1]['attrs']}]", flush=True)
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "summary.csv", index=False)
    md = ["# Final MARS transforms (Phase 3)", "",
          "Vs-free library (IP, VP, RHOB); upscaled logs; LOWO blind metrics over the listed wells. "
          "`stability` = share of CV folds that picked the same attributes. JSON files in this folder "
          "(`<zone>_<target>.json`, Zp-only controls `*_zponly.json`).", "",
          s.round(3).to_markdown(index=False)]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", OUT / "summary.md")


if __name__ == "__main__":
    main()
