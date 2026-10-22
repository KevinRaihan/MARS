"""Project locations, overridable by environment variables (local machine, Colab, Kubeflow).

``MARS_DATA_DIR``     the *raw* data folder (wells/, tops/, horizons/); default ``<repo>/data/raw``
``MARS_INTERIM_DIR``  derived large data (cubes/, props/); default ``<repo>/data/interim``
``MARS_RESULTS_DIR``  tables and figures; default ``<repo>/results``
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: Path) -> Path:
    return Path(os.environ[name]) if os.environ.get(name) else default


RAW = _env("MARS_DATA_DIR", ROOT / "data" / "raw")
INTERIM = _env("MARS_INTERIM_DIR", ROOT / "data" / "interim")
RESULTS = _env("MARS_RESULTS_DIR", ROOT / "results")
CUBES = INTERIM / "cubes"
PROPS = INTERIM / "props"
TRANSFORMS = RESULTS / "transforms"
