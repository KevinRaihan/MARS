"""Validation harness: leave-one-well-out CV with nested selection, metrics and baselines.

Every model exposes ``fit(df, target) -> self`` and ``predict(df) -> ndarray``, where ``df`` holds
the base elastic attributes (see :mod:`mars.conditioning`). For MARS, the *whole* search (library
fitting, subset selection, rotation, rescale) runs inside each training fold, so the held-out well
never influences which attributes are chosen (vault 02-Theory/validation-statistics.md).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mars.attributes import BASE_ATTRIBUTES, DEFAULT_TRANSFORMS, AttributeLibrary
from mars.conditioning import well_group
from mars.search import closed_form_search
from mars.transform import MarsTransform


def metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    if len(y) < 3 or y.std() == 0 or yhat.std() == 0:
        nan = np.nan
        return {"n": int(len(y)), "r": nan, "r2": nan, "rmse": nan, "mae": nan, "bias": nan}
    e = yhat - y
    return {"n": int(len(y)), "r": float(np.corrcoef(y, yhat)[0, 1]),
            "r2": float(1 - (e**2).sum() / ((y - y.mean()) ** 2).sum()),
            "rmse": float(np.sqrt((e**2).mean())), "mae": float(np.abs(e).mean()),
            "bias": float(e.mean())}


# ----------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------
@dataclass
class MarsModel:
    """Closed-form MARS: fitted library -> best ``nd``-subset -> rotation -> linear rescale."""

    nd: int = 2
    transforms: tuple[str, ...] = DEFAULT_TRANSFORMS
    base: tuple[str, ...] = BASE_ATTRIBUTES
    same_base_allowed: bool = False
    max_cond: float = 1e6
    top_k: int = 20
    transform_: MarsTransform | None = field(default=None, init=False)
    search_: object = field(default=None, init=False)

    @property
    def name(self) -> str:
        return f"MARS-{self.nd}D"

    def fit(self, df: pd.DataFrame, target: str) -> MarsModel:
        d = df.dropna(subset=[target])
        lib = AttributeLibrary(self.transforms).fit(d[list(self.base)])
        X = lib.transform(d[list(self.base)])
        X = X.loc[:, np.isfinite(X.to_numpy()).all(0)]
        groups = None if self.same_base_allowed else {c: lib.base_of(c) for c in X.columns}
        self.search_ = closed_form_search(X, d[target], nd=self.nd, top_k=self.top_k,
                                          groups=groups, max_cond=self.max_cond)
        self.transform_ = MarsTransform.fit(X, d[target], self.search_.best, lib, target)
        self._lib = lib
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._lib.transform(df[list(self.base)], columns=self.transform_.attrs)
        return self.transform_.predict(X)


@dataclass
class SklearnModel:
    """Baseline on standardized base attributes (OLS / LASSO / gradient boosting)."""

    kind: str = "ols"
    base: tuple[str, ...] = BASE_ATTRIBUTES

    @property
    def name(self) -> str:
        return {"ols": "OLS-11", "lasso": "LASSO-11", "hgb": "GBM-11"}[self.kind]

    def _make(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import LassoCV, LinearRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        if self.kind == "ols":
            return make_pipeline(StandardScaler(), LinearRegression())
        if self.kind == "lasso":
            return make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=20000))
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                             min_samples_leaf=40, random_state=0)

    def fit(self, df: pd.DataFrame, target: str) -> SklearnModel:
        d = df.dropna(subset=[target])
        X, y = d[list(self.base)].to_numpy(float), d[target].to_numpy(float)
        self.model_ = self._make().fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(df[list(self.base)].to_numpy(float))


# ----------------------------------------------------------------------------------------------
# Cross-validation
# ----------------------------------------------------------------------------------------------
def lowo_cv(df: pd.DataFrame, target: str, make_model: Callable[[], object],
            group_fn: Callable[[str], str] = well_group) -> dict:
    """Leave-one-group-out CV (groups = wells, sidetracks grouped with their parent).

    Returns per-fold metrics, pooled blind metrics (all held-out predictions together), the
    in-sample fit on all data, and what each fold selected (for MARS: the attribute subset).
    """
    d = df.dropna(subset=[target]).reset_index(drop=True)
    groups = d["WELL"].map(group_fn)
    folds, yhat = [], np.full(len(d), np.nan)
    for g in sorted(groups.unique()):
        test = groups == g
        if test.all():
            continue
        model = make_model().fit(d[~test], target)
        p = model.predict(d[test])
        yhat[test.to_numpy()] = p
        sel = list(model.transform_.attrs) if isinstance(model, MarsModel) else None
        folds.append({"held_out": g, **metrics(d.loc[test, target], p), "selected": sel})
    full = make_model().fit(d, target)
    insample = metrics(d[target], full.predict(d))
    out = {"model": full.name, "target": target, "insample": insample,
           "blind_pooled": metrics(d[target], yhat), "folds": folds,
           "blind_r_median": float(np.nanmedian([f["r"] for f in folds])) if folds else np.nan,
           "blind_predictions": pd.DataFrame({"WELL": d["WELL"], "y": d[target], "yhat": yhat})}
    if isinstance(full, MarsModel):
        out["selected_all_data"] = list(full.transform_.attrs)
    return out
