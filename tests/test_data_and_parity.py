"""Tests on the real (confidential) well data; skipped when MARS_DATA_DIR is absent."""

import numpy as np
import pytest

from mars import dataset
from mars.attributes import TRANSFORMS_FINAL_NOTEBOOK, TRANSFORMS_THESIS, elastic_attributes
from mars.io.las import read_las, scan_las_integrity
from mars.io.tops import legacy_zone, read_marker_txt
from mars.legacy import load_legacy_zone
from mars.search import legacy_max_correlation, multiple_correlation
from mars.thesis_reference import THESIS_PAIRS, THESIS_TABLES

TSETS = {
    "thesis": dict(transforms=TRANSFORMS_THESIS),
    "final": dict(transforms=TRANSFORMS_FINAL_NOTEBOOK, exclude=("IP^2", "sqrt(MU_RHO)")),
}

WELLS = list(dataset.wells("first12"))  # real ids come from data/dataset.json


def ronaldo() -> str:
    """Real id of RONALDO, the thesis well with measured shear velocity."""
    return dataset.wells("thesis_pair")[0]


@pytest.mark.data
@pytest.mark.parametrize("well", WELLS)
def test_staged_las_is_clean(data_dir, well):
    rep = scan_las_integrity(dataset.las_path(data_dir, well))
    assert rep["nonprint_bytes"] == 0 and rep["n_issues"] == 0


@pytest.mark.data
def test_attribute_formulas_match_petrel_curves(data_dir):
    df = read_las(dataset.las_path(data_dir, ronaldo())).dropna(
        subset=["P-VELOCITY", "S-VELOCITY_FINAL", "RHOB", "ACOUSTICIMPEDANCE", "SHEARIMPEDANCE"])
    att = elastic_attributes(df["P-VELOCITY"], df["S-VELOCITY_FINAL"], df["RHOB"])
    np.testing.assert_allclose(att["IP"], df["ACOUSTICIMPEDANCE"], rtol=1e-4)
    np.testing.assert_allclose(att["IS"], df["SHEARIMPEDANCE"], rtol=1e-4)
    np.testing.assert_allclose(att["VP/VS"], df["VP/VS"], rtol=1e-4)


@pytest.mark.data
def test_legacy_zones_ronaldo(data_dir):
    mk = read_marker_txt(data_dir / "tops" / f"{ronaldo()}_Marker.txt")
    ex = dataset.expected()
    ubr = legacy_zone(mk, "UBR", max_depth=ex["ronaldo_max_depth"])
    lbr = legacy_zone(mk, "LBR", max_depth=ex["ronaldo_max_depth"])
    assert [ubr.top, ubr.base] == ex["ronaldo_legacy_UBR"]
    assert [lbr.top, lbr.base] == ex["ronaldo_legacy_LBR"]  # includes BRF shale (legacy B14)


@pytest.fixture(scope="module")
def zone_cache(data_dir):
    cache = {}

    def get(zone, tset):
        if (zone, tset) not in cache:
            cache[(zone, tset)] = load_legacy_zone(data_dir, zone, **TSETS[tset])
        return cache[(zone, tset)]

    return get


@pytest.mark.data
@pytest.mark.parametrize("case", list(THESIS_PAIRS), ids=lambda c: f"{c[0]}-{c[1]}")
def test_thesis_table_row_reproduces(zone_cache, case):
    """Each published row (Tabel 4.1-4.3) is reproduced exactly from its winning pair.

    theta may come back shifted by 180 deg with the sign of r flipped (identical |r| up to 1e-16;
    which one wins is a floating-point tie, see vault known-bugs B1).
    """
    prop, zone = case
    a1, a2, tset = THESIS_PAIRS[case]
    ref = THESIS_TABLES[case]
    d = zone_cache(zone, tset)
    X, y = d.X, d.targets[prop]
    r, theta = legacy_max_correlation(X[a1], X[a2], y)
    assert abs(r) == pytest.approx(ref["r"], rel=1e-8)
    assert (theta - ref["theta"]) % 180 == 0
    sf1, sf2 = 15 / X[a1].max(), 15 / X[a2].max()
    assert sf1 == pytest.approx(ref["sf1"], rel=1e-8)
    assert sf2 == pytest.approx(ref["sf2"], rel=1e-8)
    th = np.radians(ref["theta"])
    tau = X[a1] * sf1 * np.sin(th) + X[a2] * sf2 * np.cos(th)
    m, c = np.polyfit(tau.to_numpy(float), y.to_numpy(float), 1)
    assert m == pytest.approx(ref["m"], rel=1e-6)
    assert c == pytest.approx(ref["c"], rel=1e-6)


@pytest.mark.data
def test_legacy_parity_vsh_lbr_fast(data_dir):
    """Re-evaluate the thesis' winning pair only (the full search lives in scripts/)."""
    d = load_legacy_zone(data_dir, "LBR")
    assert list(d.n_per_well) == list(dataset.wells("thesis_pair"))
    assert list(d.n_per_well.values()) == dataset.expected()["thesis_pair_LBR_samples"]
    y = d.targets["VSH"]
    r, theta = legacy_max_correlation(d.X["1/IP"], d.X["IS^-2"], y)
    ref = THESIS_TABLES[("VSH", "LBR")]
    assert theta == ref["theta"]
    assert r == pytest.approx(ref["r"], rel=1e-8)
    R, _ = multiple_correlation(d.X[["1/IP", "IS^-2"]].to_numpy(float), y.to_numpy(float))
    assert R == pytest.approx(0.901455, abs=1e-6)


@pytest.mark.data
def test_named_zones_ronaldo(data_dir):
    from mars.io.tops import zone_interval
    mk = read_marker_txt(data_dir / "tops" / f"{ronaldo()}_Marker.txt")
    brf = zone_interval(mk, "BRF")
    core = zone_interval(mk, "LBRc")
    ex = dataset.expected()
    assert [brf.top, brf.base] == ex["ronaldo_BRF"]
    assert [core.top, core.base] == ex["ronaldo_LBRc"]
    six_col = dataset.well("six_column_marker")  # a well with the 6-column marker layout
    six = read_marker_txt(data_dir / "tops" / f"{six_col}_Marker.txt")
    assert {"x", "y", "tvdss"} <= set(six.columns)
    top_ubr = six.set_index("formation").loc["UBR Cycle 5 / Top UBR"]
    assert top_ubr["md"] == pytest.approx(ex["six_column_top_ubr_md"])
    assert top_ubr["tvdss"] == ex["six_column_top_ubr_tvdss"]


def test_codenames_cover_all_staged_wells():
    import json
    from pathlib import Path

    from mars.naming import codename
    mapping_file = Path(__file__).resolve().parents[1] / "data" / "well_codenames.json"
    if not mapping_file.exists() or not dataset.wells("all"):
        pytest.skip("codename mapping / dataset.json not present")
    pair = dataset.wells("thesis_pair")
    assert codename(pair[0]) == "RONALDO" and codename(pair[1]) == "MESSI"
    wells = dataset.wells("all")
    names = [codename(w) for w in wells]
    assert len(set(names)) == len(names)
    with pytest.raises(KeyError):
        codename("unknown-well")
    assert json.loads(mapping_file.read_text())[pair[0]] == "RONALDO"
