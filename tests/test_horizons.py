import numpy as np
import pandas as pd

from mars.horizons import (
    grid_horizon,
    idw,
    proportional_surface,
    sample_surface,
    zone_labels,
)
from mars.io.segy import fit_affine, ilxl_to_xy

IL = np.arange(100, 120)
XL = np.arange(500, 540)


def _affine():
    gi, gx = np.meshgrid(IL, XL, indexing="ij")
    return fit_affine(gi, gx, 500000 + 0.06 * gi + 6.25 * gx, 1000000 - 25.0 * gi + 0.015 * gx)


def _plane(il, xl):
    return 1300.0 + 2.0 * (il - IL[0]) + 0.5 * (xl - XL[0])


def test_grid_horizon_snaps_and_fills_holes():
    aff = _affine()
    gi, gx = np.meshgrid(IL[2:15], XL[5:30], indexing="ij")  # partial coverage
    x, y = ilxl_to_xy(aff, gi + 0.02, gx - 0.03)  # nodes slightly off the bin centres
    df = pd.DataFrame({"x": x.ravel(), "y": y.ravel(), "twt": _plane(gi, gx).ravel()})
    df = df.drop(index=[40, 41, 100])  # holes inside the covered area
    grid, info = grid_horizon(df, aff, IL, XL)
    assert info["snapped"] == len(df) and info["filled"] == 3
    np.testing.assert_allclose(grid[2:15, 5:30], _plane(gi, gx), atol=1e-9)
    assert np.isnan(grid[0]).all() and np.isnan(grid[:, -1]).all()  # no extrapolation


def test_sample_surface_bilinear():
    aff = _affine()
    gi, gx = np.meshgrid(IL, XL, indexing="ij")
    grid = _plane(gi, gx)
    x, y = ilxl_to_xy(aff, np.array([104.3, 110.0]), np.array([512.6, 530.25]))
    np.testing.assert_allclose(sample_surface(grid, aff, IL, XL, x, y),
                               _plane(np.array([104.3, 110.0]), np.array([512.6, 530.25])))
    grid[5, 13] = np.nan
    x, y = ilxl_to_xy(aff, np.array([105.5]), np.array([513.5]))
    assert np.isnan(sample_surface(grid, aff, IL, XL, x, y)).all()


def test_zone_labels_and_crossing_surfaces():
    top = np.array([[10.0, 10.0]])
    mid = np.array([[13.0, 9.0]])  # crosses above the top in the second trace
    base = np.array([[16.0, 16.0]])
    lab = zone_labels(np.arange(20.0), [top, mid, base])
    assert lab.dtype == np.int8
    assert lab[0, 0].tolist() == [0] * 10 + [1] * 3 + [2] * 3 + [0] * 4
    assert lab[0, 1].tolist() == [0] * 9 + [2] * 7 + [0] * 4  # zone 2 = [mid, base)
    assert (zone_labels(np.arange(5.0), [np.full((1, 1), np.nan), base[:, :1]]) == 0).all()


def test_proportional_surface_and_idw():
    top, base = np.full((2, 2), 100.0), np.full((2, 2), 140.0)
    np.testing.assert_allclose(proportional_surface(top, base, np.array([[0.25, 1.5]] * 2)),
                               [[110, 140]] * 2)
    known = np.array([[0.0, 0.0], [100.0, 0.0]])
    v = idw(known, np.array([0.2, 0.6]), np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]]))
    np.testing.assert_allclose(v, [0.2, 0.4, 0.6], atol=1e-9)
