"""Elastic attributes and the transform library used by MARS.

The base formulas reproduce ``preprocess_data`` from the 2024 notebooks. Velocities in ft/s and
density in g/cc give impedances in ft/s*g/cc, the same units as the inversion cubes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

BASE_ATTRIBUTES = ("IP", "IS", "VP/VS", "LAMDA_RHO", "MU_RHO", "LAMDA_PER_MU", "LAMDA_MIN_MU",
                   "POISSON_RATIO", "ERHO", "KRHO", "PDF")


def elastic_attributes(vp: pd.Series, vs: pd.Series, rho: pd.Series) -> pd.DataFrame:
    """Compute the 11 base elastic attributes of the thesis (Tabel 3.1, top row)."""
    ip, is_ = vp * rho, vs * rho
    out = pd.DataFrame(index=vp.index)
    out["IP"] = ip
    out["IS"] = is_
    out["VP/VS"] = vp / vs
    out["LAMDA_RHO"] = ip**2 - 2 * is_**2
    out["MU_RHO"] = is_**2
    out["LAMDA_PER_MU"] = out["LAMDA_RHO"] / out["MU_RHO"]
    out["LAMDA_MIN_MU"] = out["LAMDA_RHO"] - out["MU_RHO"]
    out["POISSON_RATIO"] = ((vp / vs) ** 2 - 2) / (2 * (vp / vs) ** 2 - 2)
    out["ERHO"] = (rho**2 * vs**2 * (3 * vp**2 - 4 * vs**2)) / (vp**2 - vs**2)
    out["KRHO"] = ip**2 - (4 / 3) * is_**2
    # As coded in 2024; to be checked against Mazumdar (2007) (vault: elastic-attributes).
    out["PDF"] = (ip + np.sqrt(2) * is_) / (2 * (ip**2 - is_**2))
    return out


# name template -> function. Order matters: it fixes the column order, which decides
# ties in the legacy arg-max.
Transform = tuple[str, Callable[[pd.Series], pd.Series]]

TRANSFORMS_THESIS: tuple[Transform, ...] = (
    ("{}", lambda x: x),
    ("ln({})", np.log),
    ("e^({})", np.exp),
    ("1/{}", lambda x: 1 / x),
    ("{}^2", lambda x: x**2),
    ("{}^-2", lambda x: x**-2),
)
"""Transform set of the thesis-era notebooks (MARS_3D, MARS_<well>, MARS_COMBINED_3D...)."""

TRANSFORMS_FINAL_NOTEBOOK: tuple[Transform, ...] = (*TRANSFORMS_THESIS[:5], ("sqrt({})", np.sqrt))
"""Transform set of the last saved MARS_COMBINED.ipynb (x^-2 replaced by sqrt)."""

LEGACY_EXCLUDE_THESIS = ("IS^2", "MU_RHO^-2")
"""Duplicates the thesis dropped: IS^2 == MU_RHO, MU_RHO^-2 == IS^-4 has no Table 3.1 slot."""


def transform_library(
    base: pd.DataFrame,
    transforms: Sequence[Transform] = TRANSFORMS_THESIS,
    exclude: Sequence[str] = LEGACY_EXCLUDE_THESIS,
) -> pd.DataFrame:
    """Apply every transform to every base attribute, in legacy column order.

    Invalid values (log of negatives, exp overflow) are kept as NaN/inf, as in 2024. Use
    :func:`valid_columns` to see which columns are usable.
    """
    cols = {}
    with np.errstate(all="ignore"):
        for attr in base.columns:
            for template, f in transforms:
                cols[template.format(attr)] = f(base[attr])
    lib = pd.DataFrame(cols, index=base.index)
    return lib.drop(columns=[c for c in exclude if c in lib.columns])


def valid_columns(lib: pd.DataFrame) -> list[str]:
    """Columns that are finite everywhere (the only ones the legacy search could use)."""
    finite = np.isfinite(lib.to_numpy(dtype=float)).all(axis=0)
    return [c for c, ok in zip(lib.columns, finite, strict=True) if ok]


def elastic_attributes_from_impedance(zp, zs, rho):
    """Base attributes from inversion products (Zp, Zs, density), e.g. seismic cubes.

    Identical to :func:`elastic_attributes` with Vp = Zp/rho and Vs = Zs/rho.
    """
    return elastic_attributes(zp / rho, zs / rho, rho)


# ---------------------------------------------------------------------------------------------
# Corrected attribute library (Phase 2): fixes legacy defect B4
# ---------------------------------------------------------------------------------------------
# Each transform declares the domain it needs. The library is *fitted* on training data:
# transforms whose domain is violated anywhere in the training data are not offered (instead of
# silently producing NaN/inf columns), and exp() is applied to min-max scaled values so it cannot
# overflow. Fitted parameters are stored so the identical library can be applied to seismic data.

TRANSFORM_SPECS: dict[str, dict] = {
    "id": {"template": "{}", "domain": "any"},
    "ln": {"template": "ln({})", "domain": "positive"},
    "exp": {"template": "exp_mm({})", "domain": "any"},  # exp of (x - min)/(max - min)
    "inv": {"template": "1/{}", "domain": "nonzero_same_sign"},
    "sq": {"template": "{}^2", "domain": "any"},
    "sqrt": {"template": "sqrt({})", "domain": "positive"},
    "invsq": {"template": "{}^-2", "domain": "nonzero_same_sign"},
}
DEFAULT_TRANSFORMS = ("id", "ln", "exp", "inv", "sq", "sqrt", "invsq")


def _domain_ok(x: np.ndarray, domain: str) -> bool:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return False
    if domain == "positive":
        return bool((x > 0).all())
    if domain == "nonzero_same_sign":
        return bool((x > 0).all() or (x < 0).all())
    return True


class AttributeLibrary:
    """Fitted, domain-aware transform library (see module notes).

    Usage::

        lib = AttributeLibrary().fit(base_train)      # base: DataFrame of base attributes
        X_train = lib.transform(base_train)
        X_seis = lib.transform(base_from_cube)        # same columns, same scaling
    """

    def __init__(self, transforms: Sequence[str] = DEFAULT_TRANSFORMS):
        unknown = set(transforms) - set(TRANSFORM_SPECS)
        if unknown:
            raise ValueError(f"unknown transforms {unknown}")
        self.transforms = tuple(transforms)
        self.columns: list[dict] = []   # {"name", "base", "transform", "params"}
        self.skipped: list[dict] = []   # {"name", "reason"}

    def fit(self, base: pd.DataFrame) -> AttributeLibrary:
        self.columns, self.skipped = [], []
        seen: dict[bytes, str] = {}
        for attr in base.columns:
            x = base[attr].to_numpy(dtype=float)
            for t in self.transforms:
                spec = TRANSFORM_SPECS[t]
                name = spec["template"].format(attr)
                if not _domain_ok(x, spec["domain"]):
                    reason = f"domain '{spec['domain']}' violated"
                    self.skipped.append({"name": name, "reason": reason})
                    continue
                params = {}
                if t == "exp":
                    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
                    if hi <= lo:
                        self.skipped.append({"name": name, "reason": "constant attribute"})
                        continue
                    params = {"lo": lo, "hi": hi}
                col = {"name": name, "base": attr, "transform": t, "params": params}
                v = self._apply(col, x)
                if not np.isfinite(v[np.isfinite(x)]).all():
                    self.skipped.append({"name": name, "reason": "non-finite values"})
                    continue
                # exact duplicates (e.g. IS^2 == MU_RHO, LAMDA... identities) add nothing
                fv = v[np.isfinite(v)]
                sd = fv.std()
                key = np.round((fv - fv.mean()) / sd if sd > 0 else fv * 0, 8).tobytes()
                if key in seen:
                    self.skipped.append({"name": name, "reason": f"duplicate of {seen[key]}"})
                    continue
                seen[key] = name
                self.columns.append(col)
        return self

    @staticmethod
    def _apply(col: dict, x: np.ndarray) -> np.ndarray:
        t, p = col["transform"], col["params"]
        with np.errstate(all="ignore"):
            if t == "id":
                return x.copy()
            if t == "ln":
                return np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), np.nan)
            if t == "exp":
                return np.exp((x - p["lo"]) / (p["hi"] - p["lo"]))
            if t == "inv":
                return np.where(x != 0, 1.0 / np.where(x != 0, x, 1.0), np.nan)
            if t == "sq":
                return x**2
            if t == "sqrt":
                return np.where(x >= 0, np.sqrt(np.where(x >= 0, x, 0.0)), np.nan)
            if t == "invsq":
                return np.where(x != 0, 1.0 / np.where(x != 0, x, 1.0) ** 2, np.nan)
        raise ValueError(t)

    def transform(self, base: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
        """Compute fitted columns (all, or only ``columns``). Out-of-domain values become NaN."""
        if not self.columns:
            raise RuntimeError("library not fitted")
        cols = self._select(columns)
        out = {c["name"]: self._apply(c, base[c["base"]].to_numpy(dtype=float)) for c in cols}
        return pd.DataFrame(out, index=base.index)

    def _select(self, columns: Sequence[str] | None) -> list[dict]:
        if columns is None:
            return self.columns
        wanted = set(columns)
        return [c for c in self.columns if c["name"] in wanted]

    @property
    def names(self) -> list[str]:
        return [c["name"] for c in self.columns]

    def base_of(self, name: str) -> str:
        return next(c["base"] for c in self.columns if c["name"] == name)

    def to_dict(self, columns: Sequence[str] | None = None) -> dict:
        cols = self._select(columns)
        return {"transforms": list(self.transforms), "columns": cols}

    @classmethod
    def from_dict(cls, d: dict) -> AttributeLibrary:
        lib = cls(d["transforms"])
        lib.columns = list(d["columns"])
        return lib
