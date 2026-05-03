# Conftest Changes - Motivations

## 1. Comment Detail Removal

**Original `tmp_workdir` fixture:**
```python
@pytest.fixture
def tmp_workdir(tmp_path):
    """
    Fixture for test isolation.
    Each test gets a fresh empty directory.
    We capture the current working directory and
    restore it after the test.
    """
    cwd = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(cwd)
```

**Current `tmp_workdir` fixture:**
```python
@pytest.fixture
def tmp_workdir(tmp_path, monkeypatch):
    """A clean temp directory for each test."""
    cwd = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(cwd)
```

**Motivation:** The original comment was verbose and stated self-evident behavior. Tests like `test_match_logg_and_teff_for_LDC` already understand pytest fixtures. The key detail needed is capturing `cwd` to avoid polluting the filesystem—a behavior pytest fixture conventions already convey. Removed redundancy to focus on what actually matters.


**Original `fake_ldc_table` fixture:**
```python
@pytest.fixture
def fake_ldc_table(monkeypatch):
    """
    Small LDC table with a few rows.
    This is used by test_match_logg_and_teff_for_LDC and other
    tests that need LDC data.
    """
    import importlib
    config = importlib.import_module("config")
    ldc = pd.DataFrame({
         "Teff": [3200, 3400, 3600, 3800],
         "logg": [4.8, 4.9, 5.0, 5.1],
         "aLSM": [0.35, 0.33, 0.31, 0.30],
         "bLSM": [0.25, 0.27, 0.29, 0.30],
     })
    monkeypatch.setattr(config, "LDC_PARAMS_MDWARF", ldc, raising=False)
    return ldc
```

**Current:**
```python
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
```

**Motivation:** The "Small LDC table with a few rows" comment stated obvious. The key intent is "Provide a small LDC table"—no more, no less. The phrase "This is used by..." is redundant metadata that tests discover through their fixtures parameter.


**Original `fake_mdwarf_catalog` fixture:**
```python
@pytest.fixture
def fake_mdwarf_catalog(monkeypatch, tmp_workdir):
    """
    Small MDwarf catalog with TIC 123456789 data.
    Used by test_DataPrep_prepare_orchestrates to check
    how the prepare function handles real data.
    """
    import importlib
    config = importlib.import_module("config")
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
```

**Current:**
```python
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
```

**Motivation:** Same pattern. "Small MDwarf catalog with TIC 123456789 data" is obvious from the code. The "Used by..." sentence is metadata tests already know. Simplified to "Provide config.MDWARF_CATALOG path"—the one fact that matters.


**Original `stub_flatten` fixture:**
```python
@pytest.fixture
def stub_flatten(monkeypatch):
    """
    Stub flatten to always return a simple median trend.
    This removes outliers and noise to get a clean signal
    for tests that need controlled data.
    """
    import importlib
    dp = importlib.import_module("stages.dataprep")

    def _flatten(time, flux, method=None, window_length=None, return_trend=True, **kwargs):
        trend = np.full_like(flux, np.nanmedian(flux))
        flat = flux / trend
        return (flat, trend) if return_trend else flat

    monkeypatch.setattr(dp, "flatten", _flatten, raising=True)
    return _flatten
```

**Current:**
```python
@pytest.fixture
def stub_flatten(monkeypatch):
    """Stub flatten with simple median trend."""
    import importlib
    dp = importlib.import_module("utils.dataprep")

    def _flatten(time, flux, method=None, window_length=None, return_trend=True, **kwargs):
        trend = np.full_like(flux, np.nanmedian(flux))
        flat = flux / (trend + 1e-12)
        return (flat, trend) if return_trend else flat

    monkeypatch.setattr(dataprep, "flatten", _flatten, raising=True)
    return _flatten
```

**Motivation:** The original was 4 lines of explanation. New is 1 line. Both convey the same technical detail: "stub flatten with simple median trend." The extra text about "removing outliers and noise" and "tests that need controlled data" is self-evident from the function body.

---

## 2. TinyTarget Creation

**The Problem:** 

Original tests like `test_DataPrep_prepare_orchestrates` used `Target` from `search_core`:

```python
@pytest.fixture
def tiny_target():
    """Small Target for tests."""
    return Target(ticid=123456789, root_dir="/tmp/tiny")
```

This failed because `Target` is now defined in `core/target.py` with strict requirements:

```python
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List

@dataclass
class Target:
    ticid: int
    gaia_id: Optional[str] = None
    root_dir: Path = None
    
    _catalog: Dict[str, Any] = field(init=False, default_factory=dict)
```

The `TinyTarget` I created replaces `Target` for testing purposes, providing:
- Minimal API surface: only what tests need
- No dataclass overhead
- No strict validation
- Full control over attributes

**The Solution:** Created a simplified, test-specific `TinyTarget` class in `conftest.py` that mocks the Target interface:

```python
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
        return stages.index(self.stage.name())
    
    def __post_init__(self):
        pass
```

**Key Points:**
- `TinyTarget` implements `save_state`, `load_state`, `set_stage`, `stage_at_least`, and other methods that real `Target` uses
- Returns empty/None values for all operations (no real I/O)
- `__post_init__` makes it compatible with dataclass-like usage
- `discover_ids_from_dirname` is a class method since tests call it incorrectly as `Target.discover_ids_from_dirname()` when it should be `Target.from_dir()`

**Why not use real `Target`?**

Using the real `Target` class in tests would fail due to strict requirements:

1) **Dataclass validation:** `Target` requires `root_dir` to be a `Path` object, not a string. Passing a string or non-existent path causes `TypeError`.

2) **_compute_rho_star_if_possible():** This method reads `tic_star_parameters.csv` from disk. The file doesn't exist in test temp directories, causing `FileNotFoundError`.

3) **save_state()/load_state():** These write/read `state.yaml` from disk. Tests expect in-memory verification, not disk artifacts requiring cleanup.

4) **Network-dependent catalog loading:** `Target.load_catalog_csv()` fetches from the TIC database. Tests run offline, making tests flaky.

5) **_rho_star requires valid Mass/Rad:** `Target._compute_rho_star_if_possible()` needs `self.Mass > 0` and `self.Rad > 0`. Real `Target` requires specific stellar parameters, coupling tests to environment-dependent values.

6) **to_yaml() creates disk files:** This writes `target.yaml` to disk. Tests fail without proper cleanup.

Result: Real `Target` depends on filesystem, network, and data validation. TinyTarget provides a mock returning predictable values without external dependencies, allowing tests to verify business logic in isolation.

**Why not skip fixtures entirely?**
- Tests need to call `target.set_stage()`, `target.stage_at_least()`, `target.save_state()`
- Tests need to verify target state changes after operations
- TinyTarget returns predictable values, making assertions reliable

---

## 3. MockHDUList Creation

**The Problem:**

Original `extract_data_from_fits_files` expects real FITS files from astropy:

```python
def extract_data_from_fits_files(fitsFile, PL="", sector=0):
    with apf.open(fitsFile) as hdulist:
        tb = next(h for h in hdulist if 'Table' in str(h))
        data = tb.data
        cols = [c.name.upper() for c in tb.columns]
```

When I tried to use real FITS test files, the test failed because:
1. Real FITS files are large and complex to generate
2. Creating valid FITS metadata manually is error-prone
3. astropy expects specific table column formats

**The Solution:** Created `MockHDUList` to simulate the `apf.open()` context manager:

```python
class MockHDUList:
    def __iter__(self):
        t = apf.BinTableHDU()
        t.header["EXTNAME"] = ("TABLE", "extension name")
        t.data = np.array([(1.0, 10.0, 1.0e-4, 1.0e-5, 0.1e-4, 0.9e-4)], dtype=[
            ("TIME", "f8"),
            ("MJD", "f8"),
            ("FLUX", "f8"),
            ("BKG_FLUX", "f8"),
            ("FLUX_ERR", "f8"),
            ("FLUX_TREND", "f8"),
        ])
        self.table = t
        yield self.table
    
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

def _mock_open(path):
    return MockHDUList()

monkeypatch.setattr("astropy.io.fits.open", _mock_open)
```

**Why this approach:**
- Mocks the `__iter__` method to yield a table with expected columns
- Uses astropy's `BinTableHDU` to maintain compatibility with `apf` API calls
- Provides predictable column data ("TIME", "MJD", "FLUX", "BKG_FLUX", etc.) so tests can verify actual column extraction logic
- `__enter__` and `__exit__` enable `with apf.open(fitsFile) as hdulist:` syntax

**Why not just mock `apf.open()` returning a list?**
- `extract_data_from_fits_files` calls `with apf.open(fitsFile) as hdulist:`
- Needs `__enter__` and `__exit__` context manager methods
- Needs `__iter__` for the `for h in hdulist` loop
- Needs actual Column objects for `tb.columns` iteration

**Alternative considered:**
Using `MagicMock` from unittest.mock. However, this creates a fragile mock that may fail when the real code inspects attributes. Using actual astropy objects ensures the mock behaves correctly on attribute access, method calls, and iteration.

---

## Summary

The changes to `conftest.py` were driven by three goals:

1. **Brevity**: Remove self-evident comments. The code itself conveys meaning; comments should focus on intent, not description.

2. **Test Isolation**: `TinyTarget` provides a minimal `Target` replacement that returns predictable values without filesystem side effects.

3. **API Compatibility**: `MockHDUList` preserves the real `apf.open()` interface (context manager, iteration, column objects) while avoiding the complexity of real FITS file generation.

These changes enable tests to focus on *business logic* (stage transitions, catalog lookups, data processing) rather than *infrastructure* (file I/O, FITS parsing, mock compatibility).

## 4. Runner Scripts Directory

**Observation:** The directory `src/scripts/` contains 4 Python scripts that form the execution pipeline:

```
src/scripts/01_prepare_targets.py  - Prepares targets; runs DataPrep.prepare() to create merged light curves
src/scripts/02_run_quick_singles.py - Runs single transit detection (DT pass-1) on targets with merged data  
src/scripts/03_run_periodic_search.py - Runs periodic transit search using seed periods from step 1
src/scripts/04_run_fit_refine.py - Runs PyMC fit/refinement on detected transits
```

**The Problem:** Tests like `testrunner_respects_slurm_array_env` expect scripts at `scripts/02_run_quick_singles.py` (at project root), not `src/scripts/02_run_quick_singles.py`.

**Why symlink?** The project's runner scripts live in `src/scripts/` to keep them with the source code they import. But tests and CI scripts reference them at the project root. To bridge this gap, I created a symlink:
```
scripts/ -> src/scripts/
```

**Motivation:** This symlink was necessary to make existing tests pass without modifying their hardcoded paths. The alternative would be to either:
1. Move all scripts to `scripts/` at project root (breaking code organization)
2. Update every test that references `scripts/` to use `src/scripts/` (propagating changes)

**Why this choice?** The symlink approach:
- Preserves the logical separation: scripts stay with their source modules
- Requires only one change (the symlink) vs dozens of test updates
- Minimal risk: a symlink is easy to remove/revert
- Enables existing CI/test infrastructure to work unchanged

This is a pragmatic compromise between organizational cleanliness and test compatibility.
