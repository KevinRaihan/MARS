"""Pipeline wiring, mars.paths overrides and the Phase 5 map/section helpers (synthetic data)."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_stages_point_to_existing_scripts():
    sys.path.insert(0, str(ROOT / "scripts"))
    pipe = importlib.import_module("run_pipeline")
    names = [s.name for s in pipe.STAGES]
    assert len(names) == len(set(names))
    for s in pipe.STAGES:
        assert (ROOT / "scripts" / s.script).exists(), s.script


def test_paths_respect_environment(tmp_path):
    code = "from mars import paths; print(paths.RAW); print(paths.CUBES)"
    env = {**os.environ, "MARS_DATA_DIR": str(tmp_path / "raw"),
           "MARS_INTERIM_DIR": str(tmp_path / "int"), "PYTHONPATH": str(ROOT / "src")}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         check=True).stdout.split()
    assert Path(out[0]) == tmp_path / "raw"
    assert Path(out[1]) == tmp_path / "int" / "cubes"


def _cube(values, il=4, xl=3):
    t = np.arange(values.shape[-1], dtype=float)
    return xr.DataArray(values, dims=("il", "xl", "twt_ms"),
                        coords={"il": np.arange(il), "xl": np.arange(xl), "twt_ms": t})


def test_interval_sum_counts_masked_samples_between_surfaces():
    from mars.viz.maps import interval_sum

    prop = _cube(np.full((4, 3, 10), 0.5))
    top = xr.DataArray(np.full((4, 3), 2.0), dims=("il", "xl"))
    base = xr.DataArray(np.full((4, 3), 6.0), dims=("il", "xl"))
    mask = prop["twt_ms"] >= 4  # samples 2..5 inside, 4..5 pass the mask
    mask = mask.broadcast_like(prop)
    s = interval_sum(prop, top, base, [mask], chunk_il=3)
    np.testing.assert_allclose(s.to_numpy(), 1.0)  # 2 samples x 0.5 x 1 ms


def test_project_to_polyline_distance_and_offset():
    from mars.viz.sections import Section, project_to_polyline

    d = np.arange(0.0, 101.0, 10.0)
    sec = Section(d, np.arange(3.0), np.zeros((d.size, 3)), d, d, x=d, y=np.zeros_like(d),
                  vertices_dist_m=np.array([0.0, 100.0]))
    dist, off = project_to_polyline(sec, [21.0, 79.0], [5.0, -3.0])
    np.testing.assert_allclose(dist, [20.0, 80.0])
    np.testing.assert_allclose(off, [np.hypot(1, 5), np.hypot(1, 3)])
