# tests/conftest.py
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import os
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import datetime
from enum import Enum, auto


@pytest.fixture
def tmp_workdir(tmp_path, monkeypatch):
    """A clean temp directory for each test."""
    cwd = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(cwd)


@pytest.fixture
def fake_ldc_table(monkeypatch):
    """Provide a small LDC table."""
    import importlib
    config = importlib.import_module("utils.config")
    ldc = pd.DataFrame({
        "Teff": [3200, 3400, 3600, 3800],
        "logg": [4.8, 4.9, 5.0, 5.1],
        "aLSM": [0.35, 0.33, 0.31, 0.30],
        "bLSM": [0.25, 0.27, 0.29, 0.30],
    })
    monkeypatch.setattr(config, "LDC_PARAMS_MDWARF", ldc, raising=False)
    return ldc


@pytest.fixture
def fake_mdwarf_catalog(monkeypatch, tmp_workdir):
    """Provide config.MDWARF_CATALOG path."""
    import importlib
    config = importlib.import_module("utils.config")
    catalog_path = tmp_workdir / "mdwarf_catalog.csv"
    df = pd.DataFrame({
        "TICID": [123456789],
        "Teff": [3500],
        "logg": [5.0],
        "Mass": [0.45],
        "eMass": [0.05],
        "Rad": [0.48],
        "eRad": [0.04],
    })
    df.to_csv(catalog_path, index=False)
    monkeypatch.setattr(config, "MDWARF_CATALOG", str(catalog_path), raising=False)
    return catalog_path


@pytest.fixture
def stub_T14(monkeypatch):
    """Stub transit duration function."""
    from stages import dataprep
    def _T14(P, R_star, M_star, R_planet):
        return 0.125
    monkeypatch.setattr(dataprep, "T14", _T14, raising=True)
    return _T14


@pytest.fixture
def stub_flatten(monkeypatch):
    """Stub flatten with simple median trend."""
    from stages import dataprep

    def _flatten(time, flux, method=None, window_length=None, return_trend=True, **kwargs):
        trend = np.full_like(flux, np.nanmedian(flux))
        flat = flux / (trend + 1e-12)
        return (flat, trend) if return_trend else flat

    monkeypatch.setattr(dataprep, "flatten", _flatten, raising=True)
    return _flatten


@pytest.fixture
def stub_fits(monkeypatch):
    """Stub astropy.io.fits.open."""
    from astropy.io import fits as apf

    class MockHDUList:
        def __iter__(self):
            t = apf.BinTableHDU()
            t.header["EXTNAME"] = ("TABLE", "extension name")
            t.data = np.array([(1.0, 10.0, 1.0e-4, 1.0e-5, 0.1e-4, 0.9e-4)], dtype=[("TIME", "f8"), ("MJD", "f8"), ("FLUX", "f8"), ("BKG_FLUX", "f8"), ("FLUX_ERR", "f8"), ("FLUX_TREND", "f8")])
            self.table = t
            yield self.table
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    def _mock_open(path):
        return MockHDUList()

    monkeypatch.setattr("astropy.io.fits.open", _mock_open)


@pytest.fixture
def TinyTarget(monkeypatch, tmp_path):
    """Simplified Target for testing fixtures."""
    class FakeStage(Enum):
        RAW = auto()
        SEARCHED = auto()
        DT_PRELIM = auto()
        FIT = auto()
        REFINE = auto()
        FINISHED = auto()

    class TinyTarget:
        def __init__(self, ticid, root_dir):
            self.ticid = ticid
            self.root_dir = root_dir
            self.stage = FakeStage.RAW
            self._catalog = {}
            self.catalog = {}
            self.source_fits = []
            self.dt_prelim_found = False

        def save_state(self):
            pass

        def load_state(self):
            pass

        def set_stage(self, stage):
            self.stage = stage

        def stage_at_least(self, stage):
            stages = [e.name for e in FakeStage]
            return stages.index(self.stage.name) >= stages.index(stage.name)

        def _compute_rho_star_if_possible(self):
            pass

        def get_catalog_info(self, ticid, return_df=False):
            if return_df:
                return pd.DataFrame({"Mass": [0.45], "Rad": [0.48]})
            return {}

        @property
        def candidates_run_id(self):
            return f"run_{self.ticid}"

        def discover_ids_from_dirname(cls, dir_path):
            return (int(dir_path.stem.split("-")[1]), "TGLC")

        def new_run_id(self):
            self._run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            return self._run_id

        def candidates_dir(self):
            return self.root_dir

        def candidates_run_path(self, run_id):
            return self.root_dir / f"candidates_{run_id}.json"

        def save_candidates(self, run_id, candidates):
            pass

        def load_candidates(self, path=None):
            return []

        def stage_rank(self):
            stages = ["RAW", "SEARCHED", "DT_PRELIM", "FIT", "REFINE", "FINISHED"]
            return stages.index(self.stage.name)

        def __post_init__(self):
            pass

    return TinyTarget
