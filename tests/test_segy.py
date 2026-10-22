import numpy as np
import pytest

from mars.io.segy import (
    apply_coordinate_scalar,
    decimate_traces,
    fit_affine,
    ilxl_to_xy,
    live_extent,
    xy_to_ilxl,
)


def _grid():
    # rotated, anisotropic bin grid (synthetic geometry, 25 m x 6.25 m bins)
    il, xl = np.meshgrid(np.arange(100, 439, 13), np.arange(1000, 2756, 45), indexing="ij")
    ang = np.deg2rad(15.0)
    di, dx = 25.0, 6.25
    x = 500000 + (il - 100) * di * np.cos(ang) + (xl - 1000) * dx * np.sin(ang)
    y = 1000000 + (il - 100) * di * np.sin(ang) - (xl - 1000) * dx * np.cos(ang)
    return il, xl, x, y


def test_affine_round_trip_corners():
    il, xl, x, y = _grid()
    aff = fit_affine(il, xl, np.round(x, 2), np.round(y, 2))
    assert aff["rms_m"] < 0.01
    cil = np.array([100, 100, 438, 438])
    cxl = np.array([1000, 2755, 1000, 2755])
    cx, cy = ilxl_to_xy(aff, cil, cxl)
    bil, bxl = xy_to_ilxl(aff, cx, cy)
    np.testing.assert_allclose(bil, cil, atol=1e-6)
    np.testing.assert_allclose(bxl, cxl, atol=1e-6)
    np.testing.assert_allclose(cx[0], 500000, atol=0.01)
    np.testing.assert_allclose(cy[0], 1000000, atol=0.01)


def test_coordinate_scalar():
    out = apply_coordinate_scalar([1000, 1000, 1000], [-100, 10, 0])
    np.testing.assert_allclose(out, [10, 10000, 1000])


def test_decimate_band_limited_trace_with_padding():
    dt_in, factor, n = 0.1, 10, 4001  # ms
    t = np.arange(n) * dt_in
    sig = 20000 + 3000 * np.sin(2 * np.pi * 0.030 * t) + 1500 * np.sin(2 * np.pi * 0.071 * t + 1.0)
    noise = 400 * np.sin(2 * np.pi * 0.900 * t)  # 900 Hz: above the 500 Hz output Nyquist
    tr = (sig + noise).astype(np.float32)
    first, last = 1203, 3497
    tr[:first] = 0
    tr[last + 1 :] = 0
    out = decimate_traces(tr[None], factor)[0]
    assert out.dtype == np.float32 and out.shape == (401,)
    k = np.arange(401)
    live = (k * factor >= first) & (k * factor <= last)
    assert np.isnan(out[~live]).all() and np.isfinite(out[live]).all()
    ref = sig[k * factor]
    err = np.abs(out[live] - ref[live])
    # interior: alias suppressed, no phase shift; edges: no ringing from the zero padding
    assert err[5:-5].max() < 20
    assert err.max() < 150


def test_live_extent_and_dead_trace():
    tr = np.zeros((2, 50), dtype=np.float32)
    tr[0, 7:31] = 1.5
    first, last = live_extent(tr)
    assert first.tolist() == [7, -1] and last.tolist() == [30, -1]
    out = decimate_traces(tr, 5)
    assert np.isnan(out[1]).all()
    np.testing.assert_allclose(out[0][2:7], 1.5, atol=1e-5)


@pytest.mark.parametrize("factor", [2, 10])
def test_constant_trace_preserved(factor):
    out = decimate_traces(np.full((1, 101), 2.32, dtype=np.float32), factor)
    np.testing.assert_allclose(out, 2.32, rtol=1e-6)
