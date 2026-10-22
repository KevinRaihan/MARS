"""MARS search.

``legacy_search_2d`` reproduces the 2024 notebooks bit-for-bit, including their defects
(signed arg-max, NaN columns silently skipped, 1-degree grid on [-100, 100], SF = scale/max).
It exists for the parity test only; see vault 04-Legacy-Code/known-bugs.md (B1, B4, B5).

``multiple_correlation`` is the closed form that the rotation search approximates
(vault 02-Theory/mars-method.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LEGACY_THETAS = np.arange(-100, 101)


@dataclass(frozen=True)
class Mars2DResult:
    attr1: str
    attr2: str
    sf1: float
    sf2: float
    theta_deg: float
    r: float
    m: float
    c: float
    corr_matrix: pd.DataFrame | None = None

    def tau(self, a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
        th = np.radians(self.theta_deg)
        return a1 * self.sf1 * np.sin(th) + a2 * self.sf2 * np.cos(th)

    def predict(self, a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
        return self.m * self.tau(a1, a2) + self.c


def legacy_max_correlation(a1: pd.Series, a2: pd.Series, y: pd.Series, scale: float = 15.0):
    """``calculate_max_correlation`` from MARS_COMBINED.ipynb, unchanged in behaviour."""
    sf1, sf2 = scale / a1.max(), scale / a2.max()
    max_corr, max_theta = 0, None
    y = np.asarray(y, dtype=float)
    with np.errstate(all="ignore"):
        for theta in LEGACY_THETAS:
            t = a1 * sf1 * np.sin(np.radians(theta)) + a2 * sf2 * np.cos(np.radians(theta))
            corr = np.corrcoef(t, y)[0, 1]
            if abs(corr) > abs(max_corr):  # False for NaN -> column silently skipped
                max_corr, max_theta = corr, theta
    return max_corr, max_theta


def legacy_search_2d(X: pd.DataFrame, y: pd.Series, scale: float = 15.0) -> Mars2DResult:
    """Exhaustive pair search exactly as in the 2024 notebooks (including i == j pairs).

    The best pair is chosen with ``np.argmax`` on the *signed* correlation matrix (legacy bug B1).
    """
    attrs = list(X.columns)
    n = len(attrs)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            r, _ = legacy_max_correlation(X[attrs[i]], X[attrs[j]], y, scale)
            mat[i, j] = mat[j, i] = r
    bi, bj = np.unravel_index(np.argmax(mat), mat.shape)
    a1, a2 = attrs[bi], attrs[bj]
    r, theta = legacy_max_correlation(X[a1], X[a2], y, scale)
    sf1, sf2 = scale / X[a1].max(), scale / X[a2].max()
    t = X[a1] * sf1 * np.sin(np.radians(theta)) + X[a2] * sf2 * np.cos(np.radians(theta))
    m, c = np.polyfit(np.asarray(t, float), np.asarray(y, float), 1)
    return Mars2DResult(a1, a2, float(sf1), float(sf2), float(theta), float(mat[bi, bj]),
                        float(m), float(c), pd.DataFrame(mat, index=attrs, columns=attrs))


def legacy_select(corr_matrix: pd.DataFrame, by_abs: bool = False) -> tuple[str, str, float]:
    """Pick the best pair from a legacy correlation matrix.

    ``by_abs=False`` reproduces the 2024 signed arg-max (bug B1); ``True`` is the fix.
    """
    m = corr_matrix.to_numpy(dtype=float)
    i, j = np.unravel_index(np.argmax(np.abs(m) if by_abs else m), m.shape)
    return corr_matrix.index[i], corr_matrix.columns[j], float(m[i, j])


# ---------------------------------------------------------------------------------------------
# Closed-form best-subset search (Phase 2; fixes B1, B5, B13, B16)
# ---------------------------------------------------------------------------------------------
# For a fixed subset S of attributes, max over rotation angles of |corr(tau, y)| equals the
# multiple correlation R_S of OLS y ~ 1 + X_S (vault 02-Theory/mars-method.md). With the
# attribute correlation matrix C and attribute-target correlations r:
#     R_S^2 = r_S' C_SS^-1 r_S
# which is evaluated for all subsets in batches, without any angle grid.


@dataclass
class SearchResult:
    table: pd.DataFrame          # one row per evaluated subset: attrs, R, cond (top_k only)
    n_evaluated: int
    n_rejected_collinear: int
    nd: int

    @property
    def best(self) -> tuple[str, ...]:
        return tuple(self.table.iloc[0]["attrs"])


def _standardized_corr(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xs = (X - X.mean(0)) / X.std(0)
    ys = (y - y.mean()) / y.std()
    n = len(y)
    return Xs.T @ Xs / n, Xs.T @ ys / n


def closed_form_search(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    nd: int = 2,
    top_k: int = 50,
    groups: dict[str, str] | None = None,
    max_cond: float = 1e6,
    batch: int = 50_000,
) -> SearchResult:
    """Exhaustive search over all ``nd``-attribute subsets using the closed-form R.

    Parameters
    ----------
    X : finite attribute library (rows = samples). Columns with NaN must be removed beforehand.
    groups : optional ``{column: base_attribute}``. Subsets with two columns from the same base
        attribute (e.g. ``IP`` and ``1/IP``) are skipped: they are near-collinear and unstable
        when applied to noisier seismic attributes (legacy defect B13).
    max_cond : subsets whose correlation matrix condition number exceeds this are rejected as
        collinear (their OLS weights are ill-determined).
    """
    from itertools import combinations, islice

    Xv = np.asarray(X, dtype=float)
    yv = np.asarray(y, dtype=float)
    if not np.isfinite(Xv).all() or not np.isfinite(yv).all():
        raise ValueError("X and y must be finite; drop/clean rows and columns first")
    names = list(X.columns)
    C, r = _standardized_corr(Xv, yv)
    gid = None
    if groups is not None:
        labels = {g: k for k, g in enumerate(sorted(set(groups[c] for c in names)))}
        gid = np.array([labels[groups[c]] for c in names])

    keep_R: list[np.ndarray] = []
    keep_idx: list[np.ndarray] = []
    keep_cond: list[np.ndarray] = []
    n_eval = n_rej = 0
    it = combinations(range(len(names)), nd)
    while True:
        idx = np.array(list(islice(it, batch)), dtype=np.int64)
        if idx.size == 0:
            break
        idx = idx.reshape(-1, nd)
        if gid is not None and nd > 1:
            g = gid[idx]
            ok = (np.diff(np.sort(g, axis=1), axis=1) != 0).all(axis=1)
            idx = idx[ok]
            if idx.size == 0:
                continue
        Cs = C[idx[:, :, None], idx[:, None, :]]
        rs = r[idx]
        if nd == 1:
            R2 = rs[:, 0] ** 2
            cond = np.ones(len(idx))
        else:
            ev = np.linalg.eigvalsh(Cs)
            cond = ev[:, -1] / np.maximum(ev[:, 0], 1e-300)
            good = cond < max_cond
            n_rej += int((~good).sum())
            idx, Cs, rs, cond = idx[good], Cs[good], rs[good], cond[good]
            if idx.size == 0:
                continue
            beta = np.linalg.solve(Cs, rs[..., None])[..., 0]
            R2 = np.einsum("bk,bk->b", rs, beta)
        n_eval += len(idx)
        R = np.sqrt(np.clip(R2, 0, 1))
        if len(R) > top_k:  # keep only this batch's top_k to bound memory
            sel = np.argpartition(-R, top_k)[:top_k]
            R, idx, cond = R[sel], idx[sel], cond[sel]
        keep_R.append(R)
        keep_idx.append(idx)
        keep_cond.append(cond)

    if not keep_R:
        raise ValueError(f"no admissible {nd}-attribute subsets (all filtered by groups/max_cond); "
                         "allow same-base subsets or lower nd")
    R = np.concatenate(keep_R)
    idx = np.concatenate(keep_idx)
    cond = np.concatenate(keep_cond)
    order = np.argsort(-R)[:top_k]
    table = pd.DataFrame({
        "attrs": [tuple(names[k] for k in idx[o]) for o in order],
        "R": R[order],
        "cond": cond[order],
    })
    return SearchResult(table=table, n_evaluated=n_eval, n_rejected_collinear=n_rej, nd=nd)


def multiple_correlation(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Multiple correlation R of OLS ``y ~ 1 + X`` and the slope coefficients.

    For a fixed attribute subset this equals ``max over rotation angles of |corr(tau, y)|``.
    """
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.corrcoef(A @ beta, y)[0, 1]), beta[1:]
