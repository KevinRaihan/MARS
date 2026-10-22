"""Phase 3 experiments on the 12 thesis wells (LOWO, blind), saved for reproducibility.

Steps (all by default; pick with --steps):
  vsfree   Vs-free library vs predicted-Vs library, MARS 1-3D and GBM, log scale
           -> results/phase3/vsfree_lowo.{csv,txt}
  upscale  Backus upscaling length L = 0 / 25 / 50 / 99 ft (SWT without RONALDO)
           -> results/phase3/upscale_lowo.{csv,txt}
  brf      LBRc predicted blind when trained with vs without the BRF shale (L = 0 / 25 / 50 ft)
           -> results/phase3/brf_zone.{csv,txt}
These reproduce the tables in vault/06-Plans/phase3-findings.md. The offset diagnosis
(step 1) is scripts/diagnose_offsets.py.

Usage:  python scripts/run_phase3.py [--steps vsfree upscale brf]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mars import dataset, paths  # noqa: E402
from mars.conditioning import load_well_zone, upscale_well, well_group  # noqa: E402
from mars.validation import MarsModel, SklearnModel, lowo_cv, metrics  # noqa: E402

OUT = paths.RESULTS / "phase3"
WELLS = dataset.wells("first12")
VSFREE = ("IP", "VP", "RHOB")
TARGETS = ("PHIE", "VSH", "SWT")


class Tee:
    """Print to stdout and to a text log."""

    def __init__(self, path: Path):
        self.f = open(path, "w", encoding="utf-8")  # noqa: SIM115

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


def step_vsfree() -> None:
    log, rows = Tee(OUT / "vsfree_lowo.txt"), []
    for zone in ("UBR", "LBR"):
        df = pd.concat([load_well_zone(paths.RAW, w, zone) for w in WELLS], ignore_index=True)
        for target in TARGETS:
            for name, mk in [("MARS-2D full (Vs)", lambda: MarsModel(nd=2)),
                             ("MARS-3D full (Vs)", lambda: MarsModel(nd=3)),
                             ("MARS-1D Vs-free", lambda: MarsModel(nd=1, base=VSFREE)),
                             ("MARS-2D Vs-free", lambda: MarsModel(nd=2, base=VSFREE)),
                             ("MARS-3D Vs-free", lambda: MarsModel(nd=3, base=VSFREE)),
                             ("GBM Vs-free", lambda: SklearnModel("hgb", base=VSFREE))]:
                r = lowo_cv(df, target, mk)
                rows.append({"zone": zone, "target": target, "model": name,
                             "wells": df.WELL.nunique(), "r_in": r["insample"]["r"],
                             "blind_pooled": r["blind_pooled"]["r"],
                             "blind_median": r["blind_r_median"],
                             "sel": " | ".join(r.get("selected_all_data") or [])})
                x = rows[-1]
                log(f"{zone} {target:4s} {name:20s} in={x['r_in']:.3f} "
                    f"blind_pooled={x['blind_pooled']:.3f} median={x['blind_median']:.3f}  {x['sel']}")
    pd.DataFrame(rows).to_csv(OUT / "vsfree_lowo.csv", index=False)


def step_upscale() -> None:
    log, rows = Tee(OUT / "upscale_lowo.txt"), []
    for zone in ("UBR", "LBR"):
        raw = {w: load_well_zone(paths.RAW, w, zone) for w in WELLS}
        for L in (0, 25, 50, 99):
            df = pd.concat([raw[w] if L == 0 else upscale_well(raw[w], L) for w in WELLS],
                           ignore_index=True)
            for target in TARGETS:
                d = df if target != "SWT" else df[~df.WELL.isin(dataset.wells("swt_exclude"))]  # post-production SWT
                for name, mk in [("Vs-free 2D", lambda: MarsModel(nd=2, base=VSFREE)),
                                 ("Vs-free 3D", lambda: MarsModel(nd=3, base=VSFREE)),
                                 ("full 2D (Vs)", lambda: MarsModel(nd=2)),
                                 ("GBM Vs-free", lambda: SklearnModel("hgb", base=VSFREE))]:
                    r = lowo_cv(d, target, mk)
                    rows.append({"zone": zone, "L_ft": L, "target": target, "model": name,
                                 "n": len(d), "r_in": r["insample"]["r"],
                                 "blind_pooled": r["blind_pooled"]["r"],
                                 "blind_median": r["blind_r_median"],
                                 "sel": " | ".join(r.get("selected_all_data") or [])})
                log(f"{zone} L={L:3d} {target}: "
                    + "  ".join(f"{x['model']}={x['blind_pooled']:.3f}" for x in rows[-4:]))
    pd.DataFrame(rows).to_csv(OUT / "upscale_lowo.csv", index=False)


def _load(zone: str, L: float, log) -> pd.DataFrame:
    out = []
    for w in WELLS:
        try:
            z = load_well_zone(paths.RAW, w, zone)
        except Exception as e:  # noqa: BLE001 - two wells lack the LBR markers
            log("  skip", w, zone, e)
            continue
        out.append(z if L == 0 else upscale_well(z, L))
    return pd.concat(out, ignore_index=True)


def step_brf() -> None:
    log, rows = Tee(OUT / "brf_zone.txt"), []
    for L in (0, 25, 50):
        core, allz = _load("LBRc", L, log), _load("LBR_all", L, log)
        log(f"L={L}: LBRc wells={core.WELL.nunique()} n={len(core)}; LBR_all n={len(allz)} "
            f"(BRF share {1 - len(core) / len(allz):.0%})")
        for target in TARGETS:
            for nd in (2, 3):
                for train_name, train in (("with BRF (LBR_all)", allz),
                                          ("without BRF (LBRc)", core)):
                    c, t = core.dropna(subset=[target]), train.dropna(subset=[target])
                    if target == "SWT":
                        ex = dataset.wells("swt_exclude")
                        c, t = c[~c.WELL.isin(ex)], t[~t.WELL.isin(ex)]
                    g_c, g_t = c.WELL.map(well_group), t.WELL.map(well_group)
                    yhat = np.full(len(c), np.nan)
                    for g in sorted(g_c.unique()):  # hold out a well group, score on its LBRc
                        m = MarsModel(nd=nd, base=VSFREE).fit(t[g_t != g], target)
                        te = (g_c == g).to_numpy()
                        yhat[te] = m.predict(c[te])
                    r = metrics(c[target], yhat)
                    rows.append({"L": L, "target": target, "nd": nd, "train": train_name,
                                 "blind_r_on_LBRc": r["r"], "rmse": r["rmse"]})
                log(f"  {target} {nd}D: " + "  ".join(
                    f"{x['train']}={x['blind_r_on_LBRc']:.3f}" for x in rows[-2:]))
    pd.DataFrame(rows).to_csv(OUT / "brf_zone.csv", index=False)


STEPS = {"vsfree": step_vsfree, "upscale": step_upscale, "brf": step_brf}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--steps", nargs="+", choices=list(STEPS), default=list(STEPS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in a.steps:
        print(f"== {s}", flush=True)
        STEPS[s]()


if __name__ == "__main__":
    main()
