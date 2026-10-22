"""Phase 2: corrected attribute library, closed-form search, MarsTransform, CV harness."""

import numpy as np
import pandas as pd
import pytest

from mars import dataset
from mars.attributes import AttributeLibrary, elastic_attributes
from mars.conditioning import constant_run_mask, load_well_zone, well_group
from mars.search import closed_form_search, multiple_correlation
from mars.transform import MarsTransform
from mars.validation import MarsModel, SklearnModel, lowo_cv, metrics


def _synthetic(n=1500, p=12, seed=0, true=(2, 7, 9), coef=(0.8, -0.5, 0.3), noise=0.3):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, p))
    X = pd.DataFrame(5 + Z * np.linspace(1, 50, p), columns=[f"a{k}" for k in range(p)])
    y = sum(c * Z[:, k] for k, c in zip(true, coef, strict=True)) + noise * rng.normal(size=n)
    return X, pd.Series(y)


@pytest.mark.parametrize("nd,true", [(1, (4,)), (2, (2, 7)), (3, (2, 7, 9))])
def test_closed_form_finds_true_subset(nd, true):
    coef = (0.8, -0.5, 0.3)[:nd]
    X, y = _synthetic(true=true, coef=coef)
    res = closed_form_search(X, y, nd=nd, top_k=5)
    assert set(res.best) == {f"a{k}" for k in true}
    R, _ = multiple_correlation(X[list(res.best)].to_numpy(), y.to_numpy())
    assert res.table.iloc[0]["R"] == pytest.approx(R, abs=1e-9)
    assert res.n_evaluated == len(list(__import__("itertools").combinations(range(12), nd)))


def test_groups_and_collinearity_filters():
    X, y = _synthetic()
    X["a2_copy"] = X["a2"] * 3 + 1          # perfectly collinear with a2
    res = closed_form_search(X, y, nd=2, top_k=200, max_cond=1e6)
    assert not any({"a2", "a2_copy"} == set(a) for a in res.table["attrs"])
    assert res.n_rejected_collinear >= 1
    groups = {c: c.split("_")[0] for c in X.columns}
    res_g = closed_form_search(X, y, nd=2, top_k=500, groups=groups, max_cond=1e12)
    assert not any({"a2", "a2_copy"} == set(a) for a in res_g.table["attrs"])


def test_transform_equals_ols_and_roundtrips():
    X, y = _synthetic()
    lib = AttributeLibrary(("id",)).fit(X)
    t = MarsTransform.fit(X, y, ["a2", "a7"], lib, target="y")
    A = np.column_stack([np.ones(len(y)), X[["a2", "a7"]].to_numpy()])
    ols = A @ np.linalg.lstsq(A, y.to_numpy(), rcond=None)[0]
    np.testing.assert_allclose(t.predict(X), ols, rtol=1e-9, atol=1e-9)
    t2 = MarsTransform.from_json(t.to_json())
    np.testing.assert_allclose(t2.predict(X), t.predict(X))


def test_library_domains_duplicates_and_no_overflow():
    base = pd.DataFrame({"pos": np.linspace(2e4, 5e4, 50), "neg": np.linspace(-3, 4, 50),
                         "tiny": np.linspace(1e-9, 3e-9, 50)})
    lib = AttributeLibrary().fit(base)
    names = set(lib.names)
    assert "ln(pos)" in names and "ln(neg)" not in names and "1/neg" not in names
    assert "exp_mm(pos)" in names                       # min-max scaled: no overflow
    X = lib.transform(base)
    assert np.isfinite(X.to_numpy()).all()
    assert "1/tiny" in names and "tiny^-2" in names     # tiny values are not falsely duplicates
    assert any(s["reason"].startswith("duplicate") for s in lib.skipped)  # affine duplicates
    # out-of-domain at application time -> NaN, not an exception
    out = lib.transform(pd.DataFrame({"pos": [-1.0], "neg": [0.0], "tiny": [0.0]}))
    assert np.isnan(out["ln(pos)"].iloc[0])


def test_constant_runs_and_groups():
    s = pd.Series([1.0, 5, 5, 5, 5, 5, 2, 3, 3])
    expected = [False, True, True, True, True, True, False, False, False]
    assert constant_run_mask(s, 5).tolist() == expected
    assert well_group("W-05st") == "W-05" and well_group("W-12p") == "W-12"
    assert well_group("W-14") == "W-14"


def test_lowo_cv_runs_and_is_nested():
    rng = np.random.default_rng(1)
    frames = []
    for w in ["W1", "W2", "W3"]:
        n = 300
        vp = rng.uniform(9000, 14000, n)
        vs = vp / rng.uniform(1.7, 2.0, n)
        rho = rng.uniform(2.2, 2.7, n)
        base = elastic_attributes(pd.Series(vp), pd.Series(vs), pd.Series(rho))
        base["PHIE"] = 0.4 - 1e-5 * base["IP"] + 0.01 * rng.normal(size=n)
        base["WELL"] = w
        frames.append(base)
    df = pd.concat(frames, ignore_index=True)
    res = lowo_cv(df, "PHIE", lambda: MarsModel(nd=2))
    assert len(res["folds"]) == 3 and res["blind_pooled"]["r"] > 0.9
    assert all(f["selected"] for f in res["folds"])
    ols = lowo_cv(df, "PHIE", lambda: SklearnModel("ols"))
    assert ols["blind_pooled"]["r"] > 0.9
    assert metrics([1, 2, 3, 4], [1, 2, 3, 4])["rmse"] == 0


@pytest.mark.data
def test_paper_angle_recovers_thesis_vsh_lbr(data_dir):
    """The closed-form direction maps back to the thesis rotation angle (-34 deg, 1 deg grid)."""
    from mars.legacy import load_legacy_zone
    d = load_legacy_zone(data_dir, "LBR")
    X, y = d.X[["1/IP", "IS^-2"]], d.targets["VSH"]
    lib = AttributeLibrary(("id",)).fit(X)
    t = MarsTransform.fit(X, y, ["1/IP", "IS^-2"], lib, target="VSH")
    (theta,) = t.paper_angles([15 / X["1/IP"].max(), 15 / X["IS^-2"].max()])
    theta = (theta + 90) % 180 - 90
    assert abs(theta - (-34)) < 1.0
    assert t.train_r == pytest.approx(0.901455, abs=1e-6)


@pytest.mark.data
def test_conditioning_trims_padded_shear(data_dir):
    z = load_well_zone(data_dir, dataset.wells("thesis_pair")[0], "UBR")
    padded = dataset.expected()["ronaldo_padded_vs"]  # Petrel-padded constant at the top of RONALDO
    assert not (z["VS"].round(4) == padded).any()
    assert z[["VP", "VS", "RHOB"]].notna().all().all()


def test_upscale_backus_properties():
    from mars.conditioning import upscale_well
    n = 400
    dept = np.arange(n) * 0.5
    homog = pd.DataFrame({"DEPT": dept, "VP": 12000.0, "VS": 6000.0, "RHOB": 2.5,
                          "PHIE": 0.1, "VSH": 0.2, "SWT": 0.9, "WELL": "W"})
    u = upscale_well(homog, 20)
    assert np.allclose(u["VP"], 12000) and np.allclose(u["PHIE"], 0.1)
    # alternating 1-sample layers: modulus -> harmonic mean, density -> arithmetic mean
    vp = np.where(np.arange(n) % 2 == 0, 10000.0, 14000.0)
    rho = np.where(np.arange(n) % 2 == 0, 2.3, 2.6)
    lay = homog.assign(VP=vp, RHOB=rho)
    u = upscale_well(lay, 20).iloc[50:-50]
    m = rho * vp**2
    expect = np.sqrt(1 / np.mean(1 / m) / np.mean(rho))
    assert np.allclose(u["VP"], expect, rtol=2e-2)
    assert np.allclose(u["RHOB"], rho.mean(), rtol=2e-2)


def test_vsfree_transform_applies_to_impedance_cubes():
    rng = np.random.default_rng(3)
    n = 500
    vp, rho = rng.uniform(9000, 15000, n), rng.uniform(2.2, 2.7, n)
    base = pd.DataFrame({"IP": vp * rho, "VP": vp, "RHOB": rho})
    y = pd.Series(2.7 - rho + 1e-6 * vp + 0.01 * rng.normal(size=n))
    lib = AttributeLibrary().fit(base)
    X = lib.transform(base)
    t = MarsTransform.fit(X, y, ["ln(IP)", "RHOB^2"], lib, target="PHIE")
    p_logs = t.predict(X)
    cube = t.predict_from_impedance((vp * rho).reshape(20, 25), None, rho.reshape(20, 25))
    assert cube.shape == (20, 25)
    np.testing.assert_allclose(cube.ravel(), p_logs, rtol=1e-9)


def test_upscale_handles_zone_thinner_than_window():
    from mars.conditioning import upscale_well
    z = pd.DataFrame({"DEPT": np.arange(13) * 0.5, "VP": 11000.0, "VS": np.nan, "RHOB": 2.4,
                      "PHIE": 0.1, "VSH": 0.3, "SWT": 0.9, "WELL": "W"})
    u = upscale_well(z, 50)  # 100-sample window on a 13-sample zone
    assert len(u) == 13 and np.allclose(u["RHOB"], 2.4)
