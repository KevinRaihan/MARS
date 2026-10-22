import math
import warnings

import numpy as np
import pandas as pd
import pytest

from mars.attributes import AttributeLibrary
from mars.seismic_apply import (
    SmoothingPlan,
    ZoneRecipe,
    apply_zone,
    cpu_backend,
    get_backend,
    iter_smoothed,
    lateral_sigma_traces,
    nan_smooth,
    predict_backend,
    process_chunk,
    vertical_sigma_ms,
)
from mars.transform import MarsTransform

CPU = cpu_backend()


def _gpu():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        be = get_backend("auto")
    if be.name != "cupy":
        pytest.skip("CuPy/GPU not available")
    return be


def _cube(seed=0, shape=(12, 40, 60)):
    rng = np.random.default_rng(seed)
    zp = (25000 + 3000 * rng.standard_normal(shape)).astype(np.float32)
    rho = (2.4 + 0.15 * rng.standard_normal(shape)).astype(np.float32)
    zp[:, :, :8] = np.nan  # padding above the live window
    rho[:, :, :8] = np.nan
    rho[0, 0, 30] = -0.5  # non-physical value, clipped
    return zp, rho


def _transform(target_fn, attrs, seed=1):
    rng = np.random.default_rng(seed)
    zp = 20000 + 8000 * rng.random(400)
    rho = 2.1 + 0.6 * rng.random(400)
    base = pd.DataFrame({"IP": zp, "VP": zp / rho, "RHOB": rho})
    lib = AttributeLibrary().fit(base)
    X = lib.transform(base)
    y = pd.Series(target_fn(zp, rho) + 0.01 * rng.standard_normal(400))
    return MarsTransform.fit(X, y, attrs, lib, "T")


def test_sigma_conversions():
    assert lateral_sigma_traces(50.0) == (2.0, 8.0)  # 25 m inline, 6.25 m crossline bins
    assert math.isclose(vertical_sigma_ms(25, 10000), 5.0 / math.sqrt(12))


def test_nan_smooth_keeps_constants_and_nan():
    a = np.full((6, 10, 30), 2.5, np.float32)
    a[:, :, :5] = np.nan
    out = nan_smooth(CPU, a, (1.0, 3.0, 2.0))
    assert np.isnan(out[:, :, :5]).all()
    np.testing.assert_allclose(out[:, :, 5:], 2.5, rtol=1e-6)  # no leak of the padding


def test_anisotropic_smoothing_cpu_gpu_equivalent():
    gpu = _gpu()
    zp, rho = _cube()
    plan = SmoothingPlan(50.0, {"UBRn": 1.4, "LBRc": 3.0})
    cpu_v = [(z, r) for _, z, r in iter_smoothed(CPU, zp, rho, plan)]
    gpu_v = [(gpu.to_host(z), gpu.to_host(r)) for _, z, r in iter_smoothed(gpu, zp, rho, plan)]
    for (zc, rc), (zg, rg) in zip(cpu_v, gpu_v, strict=True):
        np.testing.assert_allclose(zg, zc, rtol=1e-5, equal_nan=True)
        np.testing.assert_allclose(rg, rc, rtol=1e-5, equal_nan=True)
    assert np.nanmin(cpu_v[0][1]) >= 1.8


def test_predict_backend_matches_transform():
    t = _transform(lambda zp, rho: 0.5 - 1e-5 * zp + 0.1 * rho,
                   ["sqrt(IP)", "exp_mm(RHOB)", "VP^-2"])
    rng = np.random.default_rng(3)
    zp, rho = 21000 + 7000 * rng.random(50), 2.15 + 0.5 * rng.random(50)
    np.testing.assert_allclose(predict_backend(CPU, t, zp, rho),
                               t.predict_from_impedance(zp, None, rho), rtol=1e-12)


def test_two_step_masking_and_zone_labels():
    vsh_t = _transform(lambda zp, rho: 1.6 - 4e-5 * zp, ["IP"])
    phie_t = _transform(lambda zp, rho: 0.4 - 0.15 * rho, ["RHOB"])
    rc = ZoneRecipe("LBRc", 3, {"vsh": vsh_t, "phie": phie_t}, {"phie": ("vsh", 0.49)})
    zp = np.linspace(20000, 34000, 40).reshape(1, 2, 20)
    rho = np.full_like(zp, 2.4)
    in_zone = np.zeros(zp.shape, bool)
    in_zone[..., 5:15] = True
    out = apply_zone(CPU, rc, zp, rho, in_zone)
    assert np.isnan(out["vsh"][~in_zone]).all() and np.isfinite(out["vsh"][in_zone]).all()
    reservoir = in_zone & (out["vsh"] <= 0.49)
    assert 0 < reservoir.sum() < in_zone.sum()
    assert np.isfinite(out["phie"][reservoir]).all()
    assert np.isnan(out["phie"][~reservoir]).all()
    assert (out["vsh"][in_zone] >= 0).all() and (out["vsh"][in_zone] <= 1).all()


def test_process_chunk_core_and_halo():
    zp, rho = _cube(shape=(10, 20, 40))
    t = _transform(lambda zp, rho: 0.3 - 0.1 * rho, ["RHOB"])
    rc = ZoneRecipe("UBRn", 1, {"phie": t})
    plan = SmoothingPlan(25.0, {"UBRn": 1.0})
    labels = np.zeros((4, 20, 30), np.int8)
    labels[:, :, 10:20] = 1
    out = process_chunk(CPU, zp, rho, labels, [rc], plan, ["phie"], slice(3, 7), slice(5, 35))
    assert out["phie"].shape == labels.shape
    assert np.isfinite(out["phie"][labels == 1]).all() and np.isnan(out["phie"][labels != 1]).all()
