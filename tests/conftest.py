import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MARS_DATA_DIR", ROOT / "data" / "raw"))


@pytest.fixture(scope="session")
def data_dir() -> Path:
    from mars import dataset

    pair = dataset.wells("thesis_pair")
    if not pair or not dataset.las_path(DATA, pair[0]).exists():
        pytest.skip("real well data not available (set MARS_DATA_DIR)")
    return DATA
