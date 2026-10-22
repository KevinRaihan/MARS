"""Closed-form MARS against the published brute-force angle search, on the final 22-well data.

For a fixed attribute subset, MARS (Alvarez et al. 2015) scans the rotation angle(s) and keeps
the one whose rotated attribute correlates best with the target. The closed form gives the same
optimum directly: u* ∝ C^-1 r, R^2 = r' C^-1 r. This script

  1. scans the angle(s) on a 1-degree grid for one 2D case (UBRn PHIE) and one 3D case
     (LBRc VSH) of the final transforms, and stores the correlation surface and both optima;
  2. runs the full subset search (ND 1-3) by brute force (correlation of every rotated attribute,
     computed from the samples) and by the closed form, and compares the winners and run times.

Angles follow the paper's convention on standardised attributes:
  2D: u = (sin th, cos th);  3D: u = (sin i sin j, cos i sin j, cos j).

Usage:  python scripts/run_closed_form_check.py
Outputs: results/closed_form/{curve_2d.csv, surface_3d.npz, search_timing.csv, summary.json}
"""

from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fit_final_transforms import RESERVOIR_ONLY, VSFREE, load  # noqa: E402

from mars import paths  # noqa: E402
from mars.attributes import AttributeLibrary  # noqa: E402
from mars.search import closed_form_search  # noqa: E402

OUT = paths.RESULTS / "closed_form"
CASES = {"2d": ("UBRn", 25, "PHIE"), "3d": ("LBRc", 50, "VSH")}
STEP = 1.0  # degrees, as in the published method


def case_data(zone: str, length: float, target: str):
    d = load(zone, length).dropna(subset=[target])
    cut = RESERVOIR_ONLY.get((zone, target))
    if cut is not None:
        d = d[d.VSH <= cut]
    lib = AttributeLibrary().fit(d[list(VSFREE)])
    X = lib.transform(d[list(VSFREE)])
    X = X.loc[:, np.isfinite(X.to_numpy()).all(0)]
    groups = {c: lib.base_of(c) for c in X.columns}
    return X, d[target].to_numpy(float), groups


def standardise(X: np.ndarray, y: np.ndarray):
    return (X - X.mean(0)) / X.std(0), (y - y.mean()) / y.std()


def directions(nd: int, step: float = STEP):
    """Unit vectors on the angle grid (half-sphere: u and -u give the same |rho|)."""
    if nd == 2:
        th = np.arange(-90, 90, step)
        t = np.radians(th)
        return np.stack([np.sin(t), np.cos(t)], 1), (th,)
    i = np.arange(-180, 180, step)
    j = np.arange(0, 90 + step / 2, step)
    ii, jj = np.meshgrid(np.radians(i), np.radians(j), indexing="ij")
    U = np.stack([np.sin(ii) * np.sin(jj), np.cos(ii) * np.sin(jj), np.cos(jj)], -1).reshape(-1, 3)
    return U, (i, j)


def brute_corr(Xs: np.ndarray, ys: np.ndarray, U: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """|corr(X u, y)| for every direction, computed from the samples (no closed-form shortcut)."""
    out = np.empty(len(U))
    for a in range(0, len(U), chunk):
        T = Xs @ U[a:a + chunk].T
        T = T - T.mean(0)
        out[a:a + chunk] = np.abs(T.T @ ys) / (np.linalg.norm(T, axis=0) * np.linalg.norm(ys))
    return out


def closed(Xs: np.ndarray, ys: np.ndarray):
    n = len(ys)
    C, r = Xs.T @ Xs / n, Xs.T @ ys / n
    u = np.linalg.solve(C, r)
    return float(np.sqrt(r @ u)), u / np.linalg.norm(u)


def angles(u: np.ndarray) -> list[float]:
    u = u if u[-1] >= 0 else -u
    if len(u) == 2:
        return [float(np.degrees(np.arctan2(u[0], u[1])))]
    return [float(np.degrees(np.arctan2(u[0], u[1]))),
            float(np.degrees(np.arctan2(np.hypot(u[0], u[1]), u[2])))]


def brute_search(X: pd.DataFrame, y: np.ndarray, groups: dict, nd: int):
    names = list(X.columns)
    Xs, ys = standardise(X.to_numpy(float), y)
    U, _ = directions(nd) if nd > 1 else (np.ones((1, 1)), None)
    best = (-1.0, None)
    n = 0
    for S in combinations(range(len(names)), nd):
        if len({groups[names[k]] for k in S}) < nd:
            continue
        rho = brute_corr(Xs[:, S], ys, U).max()
        n += 1
        if rho > best[0]:
            best = (float(rho), tuple(names[k] for k in S))
    return best, n


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for key, (zone, L, target) in CASES.items():
        X, y, groups = case_data(zone, L, target)
        attrs = list(json.loads((paths.TRANSFORMS / f"{zone}_{target}.json").read_text())["attrs"])
        Xs, ys = standardise(X[attrs].to_numpy(float), y)
        nd = len(attrs)
        U, axes = directions(nd)
        rho = brute_corr(Xs, ys, U)
        R, u = closed(Xs, ys)
        k = int(rho.argmax())
        summary[key] = {"zone": zone, "target": target, "attrs": attrs, "n": len(y),
                        "grid_r": float(rho[k]), "grid_angles": angles(U[k]),
                        "closed_r": R, "closed_angles": angles(u),
                        "grid_points": len(U)}
        if nd == 2:
            pd.DataFrame({"theta": axes[0], "rho": rho}).to_csv(OUT / "curve_2d.csv", index=False)
            r_ = Xs.T @ ys / len(ys)
            c = float(Xs[:, 0] @ Xs[:, 1] / len(ys))
            summary[key].update({"r1": float(r_[0]), "r2": float(r_[1]), "c": c})
        else:
            np.savez_compressed(OUT / "surface_3d.npz", i=axes[0], j=axes[1],
                                rho=rho.reshape(len(axes[0]), len(axes[1])))
        print(key, summary[key], flush=True)

        # full subset search: brute force vs closed form
        rows = []
        for nd_s in (1, 2, 3):
            t0 = time.perf_counter()
            (rb, sb), n_sub = brute_search(X, y, groups, nd_s)
            tb = time.perf_counter() - t0
            t0 = time.perf_counter()
            res = closed_form_search(X, y, nd=nd_s, groups=groups, top_k=5)
            tc = time.perf_counter() - t0
            rc, sc = float(res.table.iloc[0]["R"]), tuple(res.table.iloc[0]["attrs"])
            rows.append({"case": key, "zone": zone, "target": target, "nd": nd_s, "subsets": n_sub,
                         "grid_points": 1 if nd_s == 1 else len(directions(nd_s)[0]),
                         "brute_r": rb, "brute_attrs": " | ".join(sb), "brute_s": tb,
                         "closed_r": rc, "closed_attrs": " | ".join(sc), "closed_s": tc,
                         "same_subset": set(sb) == set(sc)})
            print(rows[-1], flush=True)
        summary[key]["search"] = rows
    timing = pd.DataFrame([r for v in summary.values() for r in v.pop("search")])
    timing.to_csv(OUT / "search_timing.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
