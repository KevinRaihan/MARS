"""A fitted MARS transform: attributes -> rotated attribute tau -> target units.

    tau = sum_k w_k * (A_k - mu_k) / sigma_k        (||w|| = 1, the rotation direction)
    y   = m * tau + c

This is the closed-form equivalent of Alvarez et al. (2015) eqs. 1, 2 and 4. With z-scored
attributes the rotation direction is the normalised OLS coefficient vector, so no angle grid is
needed. :meth:`paper_angles` converts the direction back to the paper's angle(s) for any choice of
paper-style scale factors, for reporting and for comparison with the 2024 thesis.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from mars.attributes import AttributeLibrary, elastic_attributes


@dataclass
class MarsTransform:
    attrs: list[str]
    mu: list[float]
    sigma: list[float]
    w: list[float]
    m: float
    c: float
    target: str
    library: dict                  # AttributeLibrary.to_dict(columns=attrs)
    units: str = "ft/s*g/cc"        # units of Ip/Is the library was fitted in
    train_r: float | None = None
    meta: dict = field(default_factory=dict)

    # ----- fitting ---------------------------------------------------------------------------
    @classmethod
    def fit(cls, X: pd.DataFrame, y: pd.Series, attrs, library: AttributeLibrary, target: str,
            **meta) -> MarsTransform:
        A = X[list(attrs)].to_numpy(dtype=float)
        yv = np.asarray(y, dtype=float)
        mu, sd = A.mean(0), A.std(0)
        Z = (A - mu) / sd
        D = np.column_stack([np.ones(len(yv)), Z])
        beta = np.linalg.lstsq(D, yv, rcond=None)[0][1:]
        w = beta / np.linalg.norm(beta)
        tau = Z @ w
        m, c = np.polyfit(tau, yv, 1)
        r = float(np.corrcoef(m * tau + c, yv)[0, 1])
        return cls(attrs=list(attrs), mu=mu.tolist(), sigma=sd.tolist(), w=w.tolist(),
                   m=float(m), c=float(c), target=target, library=library.to_dict(list(attrs)),
                   train_r=r, meta=meta)

    # ----- application -----------------------------------------------------------------------
    def tau(self, X: pd.DataFrame) -> np.ndarray:
        A = X[self.attrs].to_numpy(dtype=float)
        return ((A - np.array(self.mu)) / np.array(self.sigma)) @ np.array(self.w)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict from a frame that already holds the library columns ``self.attrs``."""
        return self.m * self.tau(X) + self.c

    @staticmethod
    def _base_frame(vp, vs, rho) -> pd.DataFrame:
        """Base attributes plus plain VP, VS, RHOB columns (used by Vs-free libraries)."""
        vp, rho = pd.Series(np.ravel(vp), dtype=float), pd.Series(np.ravel(rho), dtype=float)
        if vs is None:
            vs = pd.Series(np.nan, index=vp.index)
        else:
            vs = pd.Series(np.ravel(vs), dtype=float)
        with np.errstate(all="ignore"):
            base = elastic_attributes(vp, vs, rho)
        base["VP"], base["VS"], base["RHOB"] = vp, vs, rho
        return base

    def predict_from_logs(self, vp, vs, rho) -> np.ndarray:
        base = self._base_frame(vp, vs, rho)
        return self.predict(AttributeLibrary.from_dict(self.library).transform(base))

    def predict_from_impedance(self, zp, zs, rho) -> np.ndarray:
        """Apply to inversion products (e.g. flattened seismic cubes). ``zs`` may be None for
        transforms that do not use shear (Vs-free libraries)."""
        zp_f, rho_f = np.ravel(np.asarray(zp, float)), np.ravel(np.asarray(rho, float))
        vs = None if zs is None else np.ravel(np.asarray(zs, float)) / rho_f
        base = self._base_frame(zp_f / rho_f, vs, rho_f)
        out = self.predict(AttributeLibrary.from_dict(self.library).transform(base))
        return out.reshape(np.shape(zp))

    # ----- reporting -------------------------------------------------------------------------
    def paper_angles(self, scale_factors) -> list[float]:
        """Angles (degrees) of Alvarez eqs. 1/2 that give the same tau direction.

        ``scale_factors`` are the paper's S_k applied to raw attributes. Because the paper's
        tau has no centring, only the direction matters: S_k * dir_k is proportional to
        w_k / sigma_k.
        2D: tau = A1 S1 sin(th) + A2 S2 cos(th).
        3D: tau = (A1 S1 sin(i) + A2 S2 cos(i)) sin(j) + A3 S3 cos(j).
        """
        d = np.array(self.w) / np.array(self.sigma) / np.asarray(scale_factors, dtype=float)
        d = d / np.linalg.norm(d)
        if len(d) == 2:
            return [float(np.degrees(np.arctan2(d[0], d[1])))]
        if len(d) == 3:
            j = np.degrees(np.arctan2(np.hypot(d[0], d[1]), d[2]))
            i = np.degrees(np.arctan2(d[0], d[1]))
            return [float(i), float(j)]
        raise ValueError("paper angles defined for 2 or 3 attributes only")

    # ----- persistence -----------------------------------------------------------------------
    def to_json(self, path: str | Path | None = None) -> str:
        s = json.dumps(asdict(self), indent=1)
        if path is not None:
            Path(path).write_text(s)
        return s

    @classmethod
    def from_json(cls, path_or_str: str | Path) -> MarsTransform:
        p = Path(path_or_str) if not str(path_or_str).lstrip().startswith("{") else None
        return cls(**json.loads(p.read_text() if p else str(path_or_str)))
