"""Apply MARS transforms to the inversion cubes: preparation, zoning and streaming.

Preparation (per inline chunk, with a halo so the smoothing has no seams):

1. clip density to ``RHO_CLIP`` and mask (NaN) Zp outside ``ZP_VALID`` (non-physical
   inversion values; see :func:`count_nonphysical`);
2. NaN-aware Gaussian smoothing (normalised convolution: ``G*(v m) / G*m``, ``m`` = finite mask,
   NaN kept where the input was NaN). Lateral sigma ~ ``lateral_sigma_m`` converted to traces
   per axis (25 m inline bins, 6.25 m crossline bins), vertical sigma per zone chosen to match
   the log upscaling of the transforms: a boxcar of L ft spans ``2 L / V`` s of TWT, and a
   Gaussian with the same variance has ``sigma = width / sqrt(12)``;
3. zone labels from the boundary surfaces; each zone's transforms are applied to the version
   smoothed with that zone's vertical sigma.

The array maths (smoothing, attributes, rotation) runs on NumPy/SciPy or CuPy
(:func:`get_backend`); the transform evaluation reproduces
:meth:`mars.transform.MarsTransform.predict_from_impedance` exactly (Vs-free libraries only).
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field

import numpy as np

RHO_CLIP = (1.8, 3.0)
# Zp outside this range (ft/s*g/cc) is masked (NaN), not clipped: the 22 well logs span
# 11,603-59,948 over the Baturaja zones, while the inversion has values down to -11,073.
ZP_VALID = (5000.0, 60000.0)
IL_BIN_M, XL_BIN_M = 25.0, 6.25


# ---------------------------------------------------------------------------- backend


@dataclass
class Backend:
    name: str
    xp: object
    ndi: object

    def asarray(self, a, dtype=np.float32):
        return self.xp.asarray(a, dtype=dtype)

    def to_host(self, a) -> np.ndarray:
        return a.get() if self.name == "cupy" else np.asarray(a)

    def sync(self):
        if self.name == "cupy":
            self.xp.cuda.Stream.null.synchronize()

    def free(self):
        if self.name == "cupy":
            self.xp.get_default_memory_pool().free_all_blocks()


def cpu_backend() -> Backend:
    import scipy.ndimage as ndi

    return Backend("numpy", np, ndi)


def get_backend(name: str = "auto") -> Backend:
    """``"numpy"``, ``"cupy"`` or ``"auto"`` (CuPy if it imports and runs a test filter on the
    GPU, else NumPy). Never raises for ``"auto"``."""
    if name == "numpy":
        return cpu_backend()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import cupy as cp
            import cupyx.scipy.ndimage as cndi

            test = cndi.gaussian_filter1d(cp.arange(64, dtype=cp.float32), 2.0)
            cp.cuda.Stream.null.synchronize()
            assert np.isfinite(float(test.sum()))
        return Backend("cupy", cp, cndi)
    except Exception as e:  # noqa: BLE001 - any CuPy/driver problem means CPU
        if name == "cupy":
            raise
        warnings.warn(f"CuPy unavailable, using NumPy ({type(e).__name__}: {e})", stacklevel=2)
        return cpu_backend()


# ---------------------------------------------------------------------------- smoothing


def lateral_sigma_traces(sigma_m: float) -> tuple[float, float]:
    """Gaussian sigma in (inline, crossline) traces for a lateral sigma in metres."""
    return sigma_m / IL_BIN_M, sigma_m / XL_BIN_M


def vertical_sigma_ms(upscale_ft: float, velocity_ft_s: float) -> float:
    """Gaussian sigma (ms TWT) with the variance of an ``upscale_ft`` boxcar at ``velocity``."""
    width_ms = 2.0 * upscale_ft / velocity_ft_s * 1000.0
    return width_ms / math.sqrt(12.0)


def _gauss(be: Backend, a, sigma):
    """Separable Gaussian with per-axis sigma (0 = no smoothing on that axis)."""
    for ax, s in enumerate(sigma):
        if s > 0:
            a = be.ndi.gaussian_filter1d(a, s, axis=ax, mode="nearest", truncate=4.0)
    return a


def nan_smooth(be: Backend, a, sigma, mask=None):
    """NaN-aware Gaussian smoothing by normalised convolution (NaN stays NaN)."""
    xp = be.xp
    m = xp.isfinite(a) if mask is None else mask
    num = _gauss(be, xp.where(m, a, 0).astype(xp.float32), sigma)
    den = _gauss(be, m.astype(xp.float32), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return xp.where(m & (den > 1e-6), out, xp.nan).astype(xp.float32)


@dataclass
class SmoothingPlan:
    lateral_sigma_m: float = 50.0
    vertical_sigma_ms: dict = field(default_factory=dict)  # zone -> sigma ms
    dt_ms: float = 1.0

    def halo_inlines(self) -> int:
        return int(math.ceil(4 * lateral_sigma_traces(self.lateral_sigma_m)[0]))

    def halo_ms(self) -> float:
        return 4 * max(self.vertical_sigma_ms.values(), default=0.0)


def iter_smoothed(be: Backend, zp, rho, plan: SmoothingPlan, core=slice(None),
                  t_core=slice(None)):
    """Clip density, smooth laterally once, then vertically once per distinct zone sigma.

    Yields ``(zones, zp_s, rho_s)`` per distinct vertical sigma, cropped to ``core`` inlines and
    ``t_core`` samples (backend float32, NaN where either input is NaN). Only one smoothed
    version is alive at a time, to bound GPU/host memory.
    """
    xp = be.xp
    zp = be.asarray(zp)
    zp[(zp < ZP_VALID[0]) | (zp > ZP_VALID[1])] = xp.nan
    rho = xp.clip(be.asarray(rho), *RHO_CLIP)  # NaN stays NaN
    m = xp.isfinite(zp) & xp.isfinite(rho)
    sil, sxl = lateral_sigma_traces(plan.lateral_sigma_m)
    lat_zp = _gauss(be, xp.where(m, zp, 0).astype(xp.float32), (sil, sxl, 0))
    del zp
    lat_rho = _gauss(be, xp.where(m, rho, 0).astype(xp.float32), (sil, sxl, 0))
    del rho
    den_lat = _gauss(be, m.astype(xp.float32), (sil, sxl, 0))
    groups: dict[float, list[str]] = {}
    for zone, s_ms in plan.vertical_sigma_ms.items():
        groups.setdefault(round(s_ms / plan.dt_ms, 6), []).append(zone)
    for s, zones in groups.items():
        den = _gauss(be, den_lat, (0, 0, s))[core, :, t_core]
        bad = ~m[core, :, t_core] | (den <= 1e-6)
        den = xp.where(bad, 1.0, den)
        res = []
        for v in (lat_zp, lat_rho):
            r = _gauss(be, v, (0, 0, s))[core, :, t_core] / den
            r[bad] = xp.nan
            res.append(r.astype(xp.float32))
        del den, bad
        yield zones, res[0], res[1]
        del res


def count_nonphysical(zp, rho, labels=None) -> dict:
    """Counts of live samples with Zp outside ``ZP_VALID`` (masked) and density outside
    ``RHO_CLIP`` (clipped), overall or per zone label (host arrays of equal shape)."""
    zp, rho = np.asarray(zp), np.asarray(rho)
    live = np.isfinite(zp) & np.isfinite(rho)
    with np.errstate(invalid="ignore"):
        bad_zp = live & ((zp < ZP_VALID[0]) | (zp > ZP_VALID[1]))
        bad_rho = live & ((rho < RHO_CLIP[0]) | (rho > RHO_CLIP[1]))
    if labels is None:
        return {"live": int(live.sum()), "zp_masked": int(bad_zp.sum()),
                "rho_clipped": int(bad_rho.sum())}
    out = {}
    for k in np.unique(labels):
        sel = labels == k
        out[int(k)] = {"live": int((live & sel).sum()), "zp_masked": int((bad_zp & sel).sum()),
                       "rho_clipped": int((bad_rho & sel).sum())}
    return out


def smooth_versions(be: Backend, zp, rho, plan: SmoothingPlan) -> dict:
    """{zone: (zp_s, rho_s)} for every zone of ``plan`` (convenience; holds all versions)."""
    out = {}
    for zones, z, r in iter_smoothed(be, zp, rho, plan):
        for zone in zones:
            out[zone] = (z, r)
    return out


# ---------------------------------------------------------------------------- transforms


def _apply_column(xp, col: dict, x):
    t, p = col["transform"], col["params"]
    if t == "id":
        return x
    if t == "ln":
        return xp.where(x > 0, xp.log(xp.where(x > 0, x, 1.0)), xp.nan)
    if t == "exp":
        return xp.exp((x - p["lo"]) / (p["hi"] - p["lo"]))
    if t == "inv":
        return xp.where(x != 0, 1.0 / xp.where(x != 0, x, 1.0), xp.nan)
    if t == "sq":
        return x**2
    if t == "sqrt":
        return xp.where(x >= 0, xp.sqrt(xp.where(x >= 0, x, 0.0)), xp.nan)
    if t == "invsq":
        return xp.where(x != 0, 1.0 / xp.where(x != 0, x, 1.0) ** 2, xp.nan)
    raise ValueError(t)


def predict_backend(be: Backend, transform, zp, rho):
    """MARS prediction from impedance and density on the backend (float64 maths).

    Supports Vs-free libraries (bases IP, VP, RHOB); equals
    ``transform.predict_from_impedance(zp, None, rho)``.
    """
    xp = be.xp
    zp = xp.asarray(zp, dtype=xp.float64)
    rho = xp.asarray(rho, dtype=xp.float64)
    with np.errstate(all="ignore"):
        base = {"IP": zp, "RHOB": rho, "VP": zp / rho}
        cols = {c["name"]: c for c in transform.library["columns"]}
        tau = xp.zeros_like(zp)
        for name, mu, sd, w in zip(transform.attrs, transform.mu, transform.sigma, transform.w,
                                   strict=True):
            col = cols[name]
            if col["base"] not in base:
                raise ValueError(f"{name}: base {col['base']} not available (Vs-free only)")
            tau = tau + w * (_apply_column(xp, col, base[col["base"]]) - mu) / sd
    return transform.m * tau + transform.c


@dataclass
class ZoneRecipe:
    """What to predict in one zone: name -> transform, plus an optional reservoir rule."""

    zone: str
    label: int
    transforms: dict  # output name -> MarsTransform
    mask_rule: dict = field(default_factory=dict)  # output -> (vsh output, cutoff)


def apply_zone(be: Backend, recipe: ZoneRecipe, zp, rho, in_zone, clip01: bool = True) -> dict:
    """Predict every output of ``recipe`` where ``in_zone``; NaN elsewhere.

    Only in-zone samples are evaluated. Outputs named in ``mask_rule`` are NaN where the
    referenced VSH output exceeds the cutoff (two-step application). Values are clipped to
    [0, 1] when ``clip01``.
    """
    xp = be.xp
    sel = xp.nonzero(in_zone.ravel())[0]
    z1, r1 = zp.ravel()[sel], rho.ravel()[sel]
    flat = {}
    order = sorted(recipe.transforms, key=lambda k: k in recipe.mask_rule)  # VSH before PHIE
    for name in order:
        y = predict_backend(be, recipe.transforms[name], z1, r1)
        if clip01:
            y = xp.clip(y, 0.0, 1.0)  # NaN stays NaN
        if name in recipe.mask_rule:
            src, cut = recipe.mask_rule[name]
            y = xp.where(flat[src] <= cut, y, xp.nan)
        flat[name] = y
    out = {}
    for name, y in flat.items():
        full = xp.full(in_zone.size, xp.nan, dtype=xp.float32)
        full[sel] = y.astype(xp.float32)
        out[name] = full.reshape(in_zone.shape)
    return out


def process_chunk(be: Backend, zp, rho, labels, recipes: list[ZoneRecipe], plan: SmoothingPlan,
                  out_names: list[str], core: slice, t_core: slice) -> dict:
    """Smooth a chunk (with halo), apply each zone recipe and return host arrays for the core.

    ``labels`` holds zone labels on the core inlines and core time window (``t_core``).
    """
    res = {n: np.full(labels.shape, np.nan, np.float32) for n in out_names}
    lab = be.asarray(labels, dtype=np.int8)
    by_zone = {rc.zone: rc for rc in recipes}
    for zones, z, r in iter_smoothed(be, zp, rho, plan, core, t_core):
        for zone in zones:
            rc = by_zone.get(zone)
            if rc is None:
                continue
            for name, arr in apply_zone(be, rc, z, r, lab == rc.label).items():
                h = be.to_host(arr)
                res[name] = np.where(np.isnan(res[name]), h, res[name])
                del arr
        del z, r
        be.free()
    return res


def timed(fn, *a, **k):
    t = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t
