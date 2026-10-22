import numpy as np
import pandas as pd
import pytest

from mars.io.segy import fit_affine
from mars.tie import (
    CubeWindow,
    best_shift,
    offset_ilxl,
    resample_to_time,
    sample_cube,
    select_well_tops,
    trajectory,
    xy_along_md,
)


@pytest.fixture(autouse=True)
def _field_prefix(monkeypatch):
    """Tops identifiers are '<prefix><well>'; use a fixed synthetic prefix, not the local config."""
    from mars import dataset

    monkeypatch.setattr(dataset, "config",
                        lambda: {"las_prefix": "Field_", "unprefixed": [], "well_sets": {}})


TOPS = pd.DataFrame({
    "well": ["Field_X-1"] * 3 + ["Field_X-1 (st)"] * 2,
    "surface": ["a", "b", "c", "a", "b"],
    "x": [1000.0, 1010.0, 1030.0, 0.0, 0.0],
    "y": [5000.0, 4990.0, 4970.0, 0.0, 0.0],
    "z": [-100.0, -200.0, -300.0, 0, 0],
    "md": [100.0, 200.0, 300.0, 150.0, 250.0],
})


def test_xy_interpolation_and_extrapolation():
    t = select_well_tops(TOPS, "X-1", marker_md=pd.Series([100.0, 200.0, 300.0]))
    assert (t.well == "Field_X-1").all()
    x, y = xy_along_md(t, [100, 150, 250, 350, 50])
    np.testing.assert_allclose(x, [1000, 1005, 1020, 1040, 995])
    np.testing.assert_allclose(y, [5000, 4995, 4980, 4960, 5005])
    # decorated identifier chosen by matching marker MDs
    t2 = select_well_tops(TOPS, "X-1", marker_md=pd.Series([150.0, 250.0]))
    assert (t2.well == "Field_X-1 (st)").all()


def test_trajectory_twt_is_twice_owt():
    tr = trajectory(TOPS[TOPS.well == "Field_X-1"], [150, 250, 400], [100, 300], [600, 700])
    np.testing.assert_allclose(tr.twt[:2], [1250, 1350])
    assert np.isnan(tr.twt[2])  # outside the logged time curve


def _cube():
    il, xl, t = np.arange(10, 20), np.arange(100, 130), 1000 + np.arange(50.0)
    v = (2.0 * il[:, None, None] + 0.5 * xl[None, :, None] + 0.1 * t[None, None, :]).astype(float)
    return il, xl, t, v


def test_sample_cube_is_exact_for_trilinear_field():
    il, xl, t, v = _cube()
    fil, fxl, ft = np.array([12.3, 15.0]), np.array([110.6, 125.25]), np.array([1010.4, 1033.0])
    out = sample_cube(v, il[0], xl[0], t[0], 1.0, fil, fxl, ft)
    np.testing.assert_allclose(out, 2 * fil + 0.5 * fxl + 0.1 * ft)
    near = sample_cube(v, il[0], xl[0], t[0], 1.0, fil, fxl, ft, method="nearest")
    np.testing.assert_allclose(near, 2 * np.round(fil) + 0.5 * np.round(fxl) + 0.1 * ft)
    # outside the grid or touching NaN -> NaN
    v2 = v.copy()
    v2[2, 10, 10] = np.nan
    bad = sample_cube(v2, il[0], xl[0], t[0], 1.0, [12.5, 25.0], [110.5, 110.0], [1010.5, 1010])
    assert np.isnan(bad).all()


def test_cube_window_extraction_and_shift():
    import xarray as xr

    il, xl, t, _ = _cube()
    rng = np.random.default_rng(1)
    v = rng.normal(size=(len(il), len(xl), len(t)))
    da = xr.DataArray(v, dims=("il", "xl", "twt_ms"), coords={"il": il, "xl": xl, "twt_ms": t})
    g_il, g_xl = np.meshgrid(il, xl, indexing="ij")
    aff = fit_affine(g_il, g_xl, 500000 + 6.25 * g_xl, 1000000 - 25.0 * g_il)
    dil, dxl = offset_ilxl(aff, 12.5, -50.0)
    np.testing.assert_allclose([dil, dxl], [2.0, 2.0], atol=1e-9)
    md = np.arange(0, 20.0)
    tops = pd.DataFrame({"well": "W", "md": [0.0, 19.0],
                         "x": 500000 + 6.25 * np.array([110.0, 112.0]),
                         "y": 1000000 - 25.0 * np.array([14.0, 14.0])})
    traj = trajectory(tops, md, md, 505 + md / 2)  # TWT 1010 ... 1029 ms
    win = CubeWindow(da, aff, traj, margin_traces=2, margin_ms=10)
    got = win.sample(0.0)
    want = sample_cube(v, il[0], xl[0], t[0], 1.0, np.full(20, 14.0),
                       110 + 2 * md / 19, traj.twt)
    np.testing.assert_allclose(got, want)
    # a log equal to the cube 3 ms later is found with shift +3
    log = win.sample(3.0)
    shift, r, _, _ = best_shift(log, win.sample, max_ms=5)
    assert shift == 3.0 and r > 0.999


def test_resample_to_time_shares_bins():
    twt = np.array([0.1, 0.4, 1.2, 1.6, 2.5])
    c, m = resample_to_time(twt, [1, 3, 5, np.nan, 7])
    np.testing.assert_allclose(c, [0.5, 1.5, 2.5])
    np.testing.assert_allclose(m, [2, 5, 7])
    c2, m2 = resample_to_time(twt, [np.nan, np.nan, 1, 1, 1])
    assert len(c2) == 3 and np.isnan(m2[0])
