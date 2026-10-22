"""The rotation search must agree with the closed-form multiple correlation."""

import numpy as np
import pandas as pd
import pytest

from mars.search import legacy_max_correlation, multiple_correlation


@pytest.mark.parametrize("b1,b2", [(0.3, -0.8), (-1.0, -0.2), (0.7, 0.7)])
def test_grid_matches_closed_form_2d(b1, b2):
    rng = np.random.default_rng(0)
    n = 2000
    z1, z2 = rng.normal(size=n), rng.normal(size=n)
    a1, a2 = 10 + z1, 50 + 5 * z2  # different scales/offsets, like real attributes
    y = b1 * z1 + b2 * z2 + 0.3 * rng.normal(size=n)
    r_grid, theta = legacy_max_correlation(pd.Series(a1), pd.Series(a2), pd.Series(y))
    R, _ = multiple_correlation(np.column_stack([a1, a2]), y)
    assert theta is not None
    assert abs(abs(r_grid) - R) < 2e-3  # 1-degree grid loses a little
    assert abs(r_grid) <= R + 1e-12  # the closed form is the optimum


def test_nan_column_is_silently_skipped_like_legacy():
    a1 = pd.Series([1.0, 2.0, np.nan, 4.0])
    a2 = pd.Series([1.0, 3.0, 2.0, 5.0])
    r, theta = legacy_max_correlation(a1, a2, pd.Series([1.0, 2.0, 3.0, 4.0]))
    assert r == 0 and theta is None  # documented legacy defect B4
