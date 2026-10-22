"""Phase 2 ablation: how much does each legacy fix change in-sample and blind correlation?

Cases: the six thesis cases (PHIE/VSH/SWT x UBR/LBR) on the thesis wells RONALDO and
MESSI. "Blind" = train on one well, predict the other (2 folds), pooled.

Variants (each adds one fix to the previous):
  L0  legacy pipeline exactly as in 2024 (signed arg-max)                       [B1 present]
  L1  + best pair by |r|                                                        [B1 fixed]
  C2  + closed-form search instead of 1-deg grid with SF=15/max (same data/library) [B5]
  C3  + domain-aware fitted attribute library (no silently dropped columns)     [B4]
  C4  + per-curve NaN handling, padded-Vs trimming, no same-base / collinear pairs [B11, B13]
  C5  C4 with 3 attributes;  C6  C4 with 4 attributes
Baselines on the C4 data: MARS-1D (best single attribute), OLS-11, LASSO-11, GBM-11.

Optionally (--multiwell) a preview LOWO over all staged wells (Vs is *predicted* in all but
two wells, so treat that table as indicative only).

Usage:
    python scripts/run_ablation.py [--cases VSH:LBR ...] [--multiwell] [--skip-legacy]
Outputs: results/ablation/{ablation_2well.csv,.md,.json}, results/ablation/legacy_cache/*.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mars import dataset, paths  # noqa: E402
from mars.attributes import AttributeLibrary, valid_columns  # noqa: E402
from mars.conditioning import load_well_zone  # noqa: E402
from mars.legacy import load_legacy_zone  # noqa: E402
from mars.search import closed_form_search, legacy_max_correlation, legacy_search_2d, legacy_select  # noqa: E402
from mars.thesis_reference import THESIS_TABLES  # noqa: E402
from mars.transform import MarsTransform  # noqa: E402
from mars.validation import MarsModel, SklearnModel, lowo_cv, metrics  # noqa: E402

OUT = paths.RESULTS / "ablation"
CACHE = OUT / "legacy_cache"
WELLS = dataset.wells("thesis_pair")  # RONALDO, MESSI
ALL_WELLS = dataset.wells("first12")


# ---------------------------------------------------------------------------------------------
# legacy variants (L0, L1)
# ---------------------------------------------------------------------------------------------
def legacy_matrix(data, key: str, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Full legacy correlation matrix for (X, y), cached on disk (slow: minutes)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{key}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    t0 = time.time()
    res = legacy_search_2d(X, y)
    f.write_bytes(pickle.dumps(res.corr_matrix))
    print(f"    legacy search {key}: {time.time() - t0:.0f}s", flush=True)
    return res.corr_matrix


def legacy_fit(X: pd.DataFrame, y: pd.Series, mat: pd.DataFrame, by_abs: bool) -> dict:
    a1, a2, _ = legacy_select(mat, by_abs=by_abs)
    r, theta = legacy_max_correlation(X[a1], X[a2], y)
    sf1, sf2 = 15 / X[a1].max(), 15 / X[a2].max()
    tau = X[a1] * sf1 * np.sin(np.radians(theta)) + X[a2] * sf2 * np.cos(np.radians(theta))
    m, c = np.polyfit(tau.to_numpy(float), y.to_numpy(float), 1)
    return {"a1": a1, "a2": a2, "sf1": sf1, "sf2": sf2, "theta": theta, "m": m, "c": c}


def legacy_predict(p: dict, X: pd.DataFrame) -> np.ndarray:
    th = np.radians(p["theta"])
    tau = X[p["a1"]] * p["sf1"] * np.sin(th) + X[p["a2"]] * p["sf2"] * np.cos(th)
    return (p["m"] * tau + p["c"]).to_numpy(float)


def run_legacy(prop: str, zone: str, by_abs: bool, data_dir: Path) -> dict:
    d = load_legacy_zone(data_dir, zone)
    X, y, w = d.X, d.targets[prop], d.targets["WELL"]
    mat = legacy_matrix(d, f"{prop}_{zone}_both", X, y)
    p_all = legacy_fit(X, y, mat, by_abs)
    ins = metrics(y, legacy_predict(p_all, X))
    yhat = np.full(len(y), np.nan)
    folds = []
    for test_well in WELLS:
        tr, te = (w != test_well).to_numpy(), (w == test_well).to_numpy()
        Xt, yt = X[tr].reset_index(drop=True), y[tr].reset_index(drop=True)
        m_tr = legacy_matrix(d, f"{prop}_{zone}_train_{'+'.join(sorted(set(WELLS) - {test_well}))}", Xt, yt)
        p = legacy_fit(Xt, yt, m_tr, by_abs)
        yhat[te] = legacy_predict(p, X[te])
        folds.append({"held_out": test_well, **metrics(y[te], yhat[te]), "selected": [p["a1"], p["a2"]]})
    return {"insample": ins, "blind_pooled": metrics(y, yhat), "folds": folds,
            "selected_all_data": [p_all["a1"], p_all["a2"]]}


# ---------------------------------------------------------------------------------------------
# closed-form on legacy data (C2: legacy library; C3: fitted safe library)
# ---------------------------------------------------------------------------------------------
def closed_on_legacy(prop: str, zone: str, data_dir: Path, safe_library: bool) -> dict:
    d = load_legacy_zone(data_dir, zone)
    if safe_library:  # rebuild base attributes from the legacy rows, then use AttributeLibrary
        base_cols = ["IP", "IS", "VP/VS", "LAMDA_RHO", "MU_RHO", "LAMDA_PER_MU", "LAMDA_MIN_MU",
                     "POISSON_RATIO", "ERHO", "KRHO", "PDF"]
        base = d.X[base_cols]
    y, w = d.targets[prop], d.targets["WELL"]

    def fit(rows: np.ndarray):
        if safe_library:
            lib = AttributeLibrary().fit(base[rows])
            Xa = lib.transform(base)
            groups = None
        else:
            cols = valid_columns(d.X[rows])
            Xa = d.X[cols]
            lib = AttributeLibrary(("id",)).fit(Xa[rows])
            groups = None
        Xt = Xa[rows]
        Xt = Xt.loc[:, np.isfinite(Xt.to_numpy()).all(0)]
        res = closed_form_search(Xt, y[rows], nd=2, top_k=5, groups=groups, max_cond=1e12)
        lib_cols = AttributeLibrary(("id",)).fit(Xa[list(res.best)][rows])
        t = MarsTransform.fit(Xa[rows], y[rows], res.best, lib_cols, target=prop)
        return t, Xa

    allrows = np.ones(len(y), bool)
    t_all, Xa = fit(allrows)
    ins = metrics(y, t_all.predict(Xa))
    yhat = np.full(len(y), np.nan)
    folds = []
    for test_well in WELLS:
        te = (w == test_well).to_numpy()
        t, Xa_f = fit(~te)
        yhat[te] = t.predict(Xa_f[te])
        folds.append({"held_out": test_well, **metrics(y[te], yhat[te]), "selected": list(t.attrs)})
    return {"insample": ins, "blind_pooled": metrics(y, yhat), "folds": folds,
            "selected_all_data": list(t_all.attrs)}


# ---------------------------------------------------------------------------------------------
def summarize(name: str, res: dict) -> dict:
    f = {x["held_out"]: x for x in res["folds"]}
    return {"variant": name, "r_in": res["insample"]["r"], "r_blind": res["blind_pooled"]["r"],
            "rmse_blind": res["blind_pooled"]["rmse"],
            "r_blind_RONALDO": f.get(WELLS[0], {}).get("r"), "r_blind_MESSI": f.get(WELLS[1], {}).get("r"),
            "selected": " | ".join(res.get("selected_all_data") or [])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=paths.RAW, type=Path)
    ap.add_argument("--cases", nargs="*", default=[f"{p}:{z}" for p, z in THESIS_TABLES])
    ap.add_argument("--skip-legacy", action="store_true")
    ap.add_argument("--multiwell", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, details = [], {}
    for case in a.cases:
        prop, zone = case.split(":")
        print(f"== {case}", flush=True)
        res = {}
        if not a.skip_legacy:
            res["L0 legacy (2024)"] = run_legacy(prop, zone, False, a.data)
            res["L1 +abs argmax (B1)"] = run_legacy(prop, zone, True, a.data)
        res["C2 +closed form (B5)"] = closed_on_legacy(prop, zone, a.data, safe_library=False)
        res["C3 +safe library (B4)"] = closed_on_legacy(prop, zone, a.data, safe_library=True)
        cond = pd.concat([load_well_zone(a.data, w, zone) for w in WELLS], ignore_index=True)
        for name, mk in [("C4 +conditioning, no collinear (B11,B13)", lambda: MarsModel(nd=2)),
                         ("C5 MARS-3D", lambda: MarsModel(nd=3)),
                         ("C6 MARS-4D", lambda: MarsModel(nd=4)),
                         ("base: best single attr", lambda: MarsModel(nd=1)),
                         ("base: OLS-11", lambda: SklearnModel("ols")),
                         ("base: LASSO-11", lambda: SklearnModel("lasso")),
                         ("base: GBM-11", lambda: SklearnModel("hgb"))]:
            res[name] = lowo_cv(cond, prop, mk)
        for name, r in res.items():
            s = summarize(name, r)
            s.update({"case": case, "thesis_r": THESIS_TABLES[(prop, zone)]["r"]})
            rows.append(s)
            print(f"   {name:42s} r_in={s['r_in']:.3f}  r_blind={s['r_blind']:.3f}  [{s['selected']}]",
                  flush=True)
        details[case] = res
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ablation_2well.csv", index=False)
    (OUT / "ablation_2well.json").write_text(json.dumps(details, indent=1, default=float))
    md = ["# Ablation: thesis wells RONALDO + MESSI (blind = train one well, predict the other)", ""]
    for case, g in df.groupby("case", sort=False):
        md += [f"## {case} (thesis in-sample r = {g['thesis_r'].iloc[0]:.3f})", "",
               "| variant | r in-sample | r blind (pooled) | blind RONALDO | blind MESSI | RMSE blind | selected (all data) |",
               "|---|---|---|---|---|---|---|"]
        for _, s in g.iterrows():
            md.append(f"| {s['variant']} | {s['r_in']:.3f} | {s['r_blind']:.3f} | {s['r_blind_RONALDO']:.3f} | "
                      f"{s['r_blind_MESSI']:.3f} | {s['rmse_blind']:.4f} | {s['selected']} |")
        md.append("")
    (OUT / "ablation_2well.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", OUT / "ablation_2well.md")

    if a.multiwell:
        mrows = []
        for case in a.cases:
            prop, zone = case.split(":")
            frames = []
            for w in ALL_WELLS:
                try:
                    frames.append(load_well_zone(a.data, w, zone))
                except Exception as e:  # noqa: BLE001 - a well may lack the zone
                    print(f"   skip {w} {zone}: {e}")
            cond = pd.concat(frames, ignore_index=True)
            for name, mk in [("MARS-2D", lambda: MarsModel(nd=2)), ("MARS-3D", lambda: MarsModel(nd=3)),
                             ("best single attr", lambda: MarsModel(nd=1)),
                             ("OLS-11", lambda: SklearnModel("ols")), ("GBM-11", lambda: SklearnModel("hgb"))]:
                r = lowo_cv(cond, prop, mk)
                mrows.append({"case": case, "variant": name, "n_wells": cond["WELL"].nunique(),
                              "r_in": r["insample"]["r"], "r_blind_pooled": r["blind_pooled"]["r"],
                              "r_blind_median": r["blind_r_median"],
                              "selected": " | ".join(r.get("selected_all_data") or [])})
                print(f"   [multi] {case} {name:18s} r_in={mrows[-1]['r_in']:.3f} "
                      f"blind_pooled={mrows[-1]['r_blind_pooled']:.3f} median={mrows[-1]['r_blind_median']:.3f}",
                      flush=True)
        pd.DataFrame(mrows).to_csv(OUT / "ablation_multiwell.csv", index=False)


if __name__ == "__main__":
    main()
