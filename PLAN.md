# Test Coverage Improvement Plan

## Overview

This plan outlines the approach to increase unit test coverage of the LPP Search Python project, which is a transit detection pipeline for exoplanet discovery using TESS/TGLC light curve data.

## Executive Summary

### Current State
- **13 existing tests** in `src/tests/` covering ~15% of critical modules
- **Critical blocker**: `conftest.py` has syntax errors (indentation issues on lines 89, 92) that prevent any tests from running
- **Coverage gaps**: Core modules have minimal to no test coverage including:
  - `utils/run_json.py` - 1 test for `upsert_run_json`, 0 for `append_run_json_list`
  - `core/target.py` - 0 tests
  - `core/planet_candidate.py` - 0 tests  
  - `core/transit_event.py` - 0 tests
  - `core/periodic_event.py` - 0 tests
  - `utils/run_context.py` - 0 tests
  - `utils/ticid_input_coordination.py` - 0 tests
  - `utils/run_context.py` - 0 tests
  - `scripts/*.py` - indirect test coverage only via runner module loading

### Target State
Aim for 80%+ coverage on critical modules with proper test fixtures, edge case coverage, and integration tests.

---

## Critical Path: Fix Blockers First

### Phase 1: Fix Test Infrastructure Blockers

**Files to fix:**
1. `src/tests/conftest.py` - Lines 89, 92 need `pass` statements
2. `src/stages/__init__.py` - Empty file, consider removing or adding exports

**Why first**: Without fixing conftest.py syntax errors, NO tests can run. This is blocking all other test work.

**Specific fixes:**
```python
# Line 89:
class FakeColumn:
    def __init__(self, name): 
        self.name = name  # Add proper indentation

# Line 92:  
class FakeTable:
    def __init__(self, data, cols):
        pass  # Add pass statement
```

---

## Phase 2: Core Module Coverage (Critical Path)

### Priority Order by module size, coupling, and test difficulty:

#### 2.1 `core/target.py` (~90 lines, critical dependency)
**Coverage target: 85%**

**Key functions to test:**
- `Target.__post_init__()` - initialization validation
- `Target._compute_rho_star_if_possible()` - catalog parsing edge cases  
- `Target.save_state()` / `load_state()` - serialization roundtrip
- `Target.discover_ids_from_dirname()` / `from_dir()` - path parsing
- `Target.load_catalog_csv()` - CSV loading, empty file handling
- `Target.set_stage()` - state machine transitions
- `Target.stage_at_least()` - stage ordering logic
- `Target.new_run_id()` / `candidates_run_path()` - path generation
- `Target.save_candidates()` / `load_candidates()` - JSON I/O

**Test scenarios:**
- Empty catalog initialization
- Invalid catalog data handling
- Pipeline stage transitions (all 6 stages)
- State persistence and restoration
- Candidate save/load roundtrip

#### 2.2 `core/planet_candidate.py` (~120 lines, critical dependency)
**Coverage target: 90%**

**Key functions to test:**
- `PlanetCandidate.candidate_id()` - ID generation edge cases
- `PlanetCandidate.compute_fit_fingerprint()` / `refresh_fit_status()` - fingerprint matching
- `PlanetCandidate.mark_fitted()` / `mark_needs_refit()` - fit state management
- `PlanetCandidate.single_candidates_from_dt_events()` - event conversion
- `to_dict()` / `from_dict()` - serialization
- `observed_transit_times_days` - single vs periodic handling
- `set_transit_times()` - deduplication and validation

**Test scenarios:**
- Single candidate fingerprint generation
- Periodic candidate fingerprint generation (with periods)
- Fingerprint change detection after fit
- Transit time deduplication from NaN filtering
- Empty list handling

#### 2.3 `core/transit_event.py` (~40 lines, minimal complexity)
**Coverage target: 100%**

**Key functions to test:**
- `TransitEvent.to_dict()` / `from_dict()` - serialization roundtrip
- Optional field handling (snr, confidence)

**Test scenarios:**
- Basic TransitEvent creation
- None values in optional fields

#### 2.4 `core/periodic_event.py` (~60 lines, minimal complexity)
**Coverage target: 100%**

**Key functions to test:**
- `PeriodicEvent.to_dict()` / `from_dict()` - serialization roundtrip
- Optional field handling (duration, depth, snr, sde, n_transits_obs)

**Test scenarios:**
- Basic PeriodicEvent creation
- None values in optional fields
- Full roundtrip serialization

---

## Phase 3: Utility Module Coverage

### 3.1 `utils/run_json.py` (~25 lines, critical for run artifacts)
**Coverage target: 100%**

**Key functions to test:**
- `upsert_run_json()` - create new file, append to existing
- `append_run_json_list()` - list operations, key creation

**Test scenarios:**
- First write to non-existent file
- Append to existing JSON
- List key initialization
- Append to non-existent list key

### 3.2 `utils/run_context.py` (~10 lines, utility)
**Coverage target: 100%**

**Key functions to test:**
- `make_run_id()` - timestamp format
- `make_run_path()` - path construction

**Test scenarios:**
- Run ID generation (unique timestamps)
- Run path creation (parents=True behavior)

### 3.3 `utils/find_total_csv.py` (~15 lines, utility)
**Coverage target: 100%**

**Key functions to test:**
- `find_total_csv()` - pattern matching, file selection

**Test scenarios:**
- Exact flavor match
- Fallback to any total.csv
- FileNotFoundError with pattern

### 3.4 `utils/segments.py` (~30 lines, shared by singles/periodic)
**Coverage target: 95%**

**Key functions to test:**
- `find_breaks()` - single break, no breaks, consecutive breaks
- `breaking_up_data()` - empty input, min_size filtering

**Test scenarios:**
- No gaps (single segment)
- Multiple gaps creating multiple segments
- min_size filter removing small segments
- Empty input handling

### 3.5 `utils/running_median.py` (~30 lines)
**Coverage target: 90%**

**Key functions to test:**
- `running_median()` - normal case, edge cases, inf handling

**Test scenarios:**
- Normal kernel size operation
- Kernel larger than data
- Inf values in result
- Empty/short data handling

### 3.6 `utils/config.py` (~25 lines, config)
**Coverage target: 100%**

**Key functions to test:**
- `LDC_PARAMS_MDWARF` filtering
- MODULE-LEVEL imports and filtering

**Note**: This is mostly config; test import works correctly.

---

## Phase 4: Stages Module Coverage

### 4.1 `stages/dataprep.py` (~200 lines, moderately complex)
**Coverage target: 80%**

**Key functions to test:**
- `match_logg_and_teff_for_LDC()` - LDC lookup, NaN handling
- `get_catalog_info()` - TICID lookup, chunking
- `remove_outliers()` - mask generation
- `T14()` - transit duration calculation edge cases
- `flatten_lc()` - catalog edge cases (empty, invalid values)
- `extract_data_from_fits_files()` - FITS parsing, column filtering
- `get_data()` - multi-sector merging, empty sectors
- `DataPrep` class methods (`ensure_catalog`, `find_fits`, `extract_all`, `merge_total`, `prepare`)

**Test scenarios:**
- LDC lookup with perfect Teff match
- LDC lookup with NaN logg fallback
- Logg interpolation with k-nearest
- Empty catalog handling
- Invalid M_star/R_star fallback
- Multi-sector light curve merging
- FITS column extraction

### 4.2 `stages/search_singles.py` (~250 lines, core detection)
**Coverage target: 75%**

**Key functions to test:**
- `make_LightKurveObject()` - light curve object creation
- `calc_rudimentary_snr()` - SNR calculation
- `plot_lc_with_bboxes()` - plotting wrapper
- `DT_analysis()` - stdout capture, bbox output
- `detect_transit_events()` - event construction, SNR/confidence handling
- `singles_search()` - full pipeline with/without exclude_mask

**Test scenarios:**
- Single DT detection
- Multiple detections with duplicate T0 dedup
- No signals found (empty events)
- Exclude mask filtering
- Segment quality cut (breaking_up_data)
- plot_events=True behavior

### 4.3 `stages/search_periodic.py` (~525 lines, complex)
**Coverage target: 60%**

**Key functions to test:**
- `transit_mask()` - full-period masking
- `_nearest_epoch_time()` / `_offset_to_ephemeris()` - epoch calculation
- `epoch_tolerance_days()` - tolerance calculation
- `is_repeat_ephemeris()` - repeat detection
- `compute_sde()` - SDE calculation edge cases
- `seed_period_grid()` - grid generation
- `normalize_depth_to_fractional()` - depth conversion
- `checking_last_BLS_power_for_artificial_inflation()` - power pattern
- `make_power_final()` / `best_peak_index_from_power()` - BLS power processing
- `evaluate_best_peak()` - threshold gate, single-like gate, repeat gate, accept
- `process_bls_results()` / `run_bls()` - BLS interface
- `using_BLS_search()` - retry logic, mask-only, repeat
- `run_seed_prepass_full_lc()` - seed period pre-pass
- `run_periodic_full_and_chunked()` - chunking logic
- `periodic_search()` - full pipeline

**Test scenarios:**
- BLS period detection within thresholds
- SNR threshold failure (stop)
- Single-like candidate (mask only)
- Repeat ephemeris detection
- Seed period grid usage
- Chunked search on multiple segments

---

## Phase 5: Utils/Helpers Coverage

### 5.1 `utils/alias_dedup.py` (~95 lines, dedup logic)
**Coverage target: 75%**

**Key functions to test:**
- `_get_summary()` - nested dict access
- `_median()` / `_hdi16()` - fallback handling
- `_max_rhat()` - rhat extraction from nested dicts
- `_depth_med()` / `_signal_not_zero()` - signal filtering
- `_snr_med()` / `_period()` / `_t0()` - helper access
- `_phase_offset_days()` - phase calculation edge cases
- `_get_observed_transit_times()` - single vs periodic
- `_shared_t0_overlap()` - overlap counting
- `_depth_consistent()` - depth ratio validation
- `_winner_key()` - winner scoring
- `alias_dedup_periodic_candidates()` - full dedup pipeline

**Test scenarios:**
- No candidates (single-item or empty)
- Three-way clustering via union-find
- Depth inconsistency blocking dedup
- Winner selection by SNR, fit status, rhat

### 5.2 `utils/singles_periodicity.py` (~280 lines, complex)
**Coverage target: 70%**

**Key functions to test:**
- `prepping_singles_for_periodic_check()` - candidate period generation
- `score_once_modes()` - scoring, depth consistency, phase RMS
- `extract_all_modes_iterative()` - iterative mode extraction
- `periodic_modes_from_dt_events()` - DT event conversion
- `seed_periods_from_dt_events()` - top-k extraction
- `candidate_from_mode()` - mode to candidate conversion
- `periodic_candidates_from_modes()` - batch conversion
- `mark_single_members_consumed()` - single marking

**Test scenarios:**
- Single transit (no periodic detectable)
- Three-transit periodic detection
- Multiple periodic modes (iterative removal)
- Depth consistency filtering
- Mode-to-candidate conversion with member indices

### 5.3 `utils/ticid_input_coordination.py` (~35 lines)
**Coverage target: 90%**

**Key functions to test:**
- `find_target_dir_by_ticid()` - TICID matching, not found
- `load_ticids_txt()` - file load, empty lines, comments
- `resolve_ticid()` - --ticid flag, --ticid-file flag, positional, error

**Test scenarios:**
- Direct --ticid flag
- --ticid-file with SLURM_ARRAY_TASK_ID
- --ticid-file without SLURM (SystemExit)
- Positional argument fallback

### 5.4 `utils/queue.py` (~15 lines)
**Coverage target: 90%**

**Key functions to test:**
- `enqueue()` - marker file creation, atomic create

**Test scenarios:**
- Single target enqueue
- Duplicate queue (FileExistsError suppressed)

---

## Phase 6: Script-Level Integration Tests

### 6.1 `scripts/02_run_quick_singles.py` (~85 lines)
**Coverage target: 80%**

**Key functions to test:**
- `_has_merged_data()` - stage gate, file fallback
- `main()` - out-of-range index, stage gate, detection, queue

**Test scenarios:**
- Index out of range (exit 2)
- Not ready target (no merged data)
- DT-finding target (stage transition, enqueue)
- Already-searched target (skip, save)

### 6.2 `scripts/03_run_periodic_search.py` (~145 lines)
**Coverage target: 70%**

**Key functions to test:**
- `main()` - index validation, stage gate, dt_prelim check, periodic search, run_json updates

**Test scenarios:**
- Index out of range
- Stage < SEARCHED (exit 3)
- dt_prelim_found=False (skip)
- Full periodic discovery pipeline

### 6.3 `scripts/04_run_fit_refine.py` (~375 lines, complex)
**Coverage target: 60%**

**Key functions to test:**
- `load_run_json()` - basic JSON load
- `write_final_candidates_csv()` - CSV generation
- `append_global_candidates_csv()` - global CSV append
- `fit_and_attach()` - fit success/failure path
- `_summary_median()` - median extraction from nested dict structure
- `periodic_mask_from_fitted_candidate()` - mask generation
- `run_fit_refine_for_target()` - full pipeline, pass2, promotion

**Test scenarios:**
- No periodic events (finalize_pass1_singles_only)
- Single-pass discovery
- Pass-2 search with mask
- Pass2-to-periodic promotion

---

## Phase 7: Engine Coverage (Optional, Deep Dependency)

### 7.1 `engines/pyMC_core.py` (~300 lines)
**Coverage target: 70%**

**Key functions to test:**
- `extract_summary_dataframe()` - summary extraction, columns
- `transit_mask_tensors()` - tensor mask generation
- `sample_until_converged()` - convergence success, failure retries
- `BatmanOp` - perform, grad
- `prepare_fit_data()` - masking, cadence, single vs periodic
- `median_pytensor()` - even/odd case
- `make_windows_from_time_stamps()` - window generation, gaps
- `pymc_fit_candidate()` - Single/Periodic model creation, convergence, exceptions

**Notes**: This requires significant mocking. Consider using `monkeypatch` to mock `batman.TransitModel`, `pm.sample`, etc. Testing will be complex and expensive.

---

## Phase 8: Existing Tests Enhancement

### 8.1 `conftest.py` (Fix first!)
**Actions:**
- Fix syntax errors on lines 89-92
- Add `conftest.py` imports cleanup (unused imports)
- Add `target_state.json` fixture for single-target tests

### 8.2 `tests/test_dataprep.py` (4 tests)
**Enhancement opportunities:**
- Test `match_logg_and_teff_for_LDC` with edge cases (Teff outside range, NaN logg)
- Test `get_catalog_info` chunking path
- Test `remove_outliers` edge cases
- Add `DataPrep.prepare_orchestrates` edge cases (no FITS files, all empty sectors)

### 8.3 `tests/test_singles_search.py` (4 tests)
**Enhancement opportunities:**
- Test `singles_search` with periodic candidates (exclude_mask)
- Test `run_1=False` return value
- Test plotting path

### 8.4 `tests/test_run_quick_singles.py` (3 tests)
**Enhancement opportunities:**
- Test `_load_runner_module` edge cases
- Test index out-of-range behavior
- Test stage gate (already SEARCHED)

---

## Verification Strategy

### Coverage Measurement
Run `pytest --cov=src --cov-report=term-missing --cov-fail-under=80` to measure coverage.

### Incremental Approach
1. Fix conftest.py syntax (Phase 1)
2. Run existing tests: `pytest src/tests/`
3. Reveal which modules have 0% coverage
4. Tackle Phase 2 modules (core/*.py) first
5. Run coverage after each phase

### Test Prioritization
- **Critical path**: Core modules (target.py, planet_candidate.py, transit_event.py, periodic_event.py)
- **High value**: Utils modules (run_json.py, run_context.py, find_total_csv.py, segments.py)
- **Medium value**: Stages modules (dataprep.py, search_singles.py, search_periodic.py)
- **Optional**: Script integration tests, engine coverage

---

## Implementation Notes

### Test Structure Recommendation
Follow existing pattern in conftest.py:
```python
@pytest.fixture
def tmp_workdir(tmp_path, monkeypatch):
    """Clean temp directory scoped to test."""
    cwd = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(cwd)

@pytest.fixture
def fake_ldc_table(monkeypatch):
    """LDC table fixture."""
    config = importlib.import_module("config")
    # ... set up config.LDC_PARAMS_MDWARF
```

### Mocking Strategy
- **conftest.py**: Heavy use of `monkeypatch` for imports and config values
- **Core modules**: Direct instantiation with test fixtures
- **Utils functions**: Direct calls with edge case inputs
- **Stages modules**: Mock heavy dependencies (DeepTransit, BoxLeastSquares)
- **Engine module**: Mock batman and pymc via monkeypatch

### Key Test Patterns
```python
# Basic unit test with direct instantiation
def test_target_save_load(tmp_workdir):
    root = Path("target_tic-123")
    root.mkdir()
    t = Target(ticid=123, gaia_id="GAIA123", root_dir=root)
    t.save_state()
    
    # Reload
    t2 = Target.from_dir(root)
    assert t2.ticid == 123
```

---

## Summary of Files to Modify

| File | Current Coverage | Target Coverage | Priority |
|------|-----------------|-----------------|----------|
| `src/tests/conftest.py` | BLOCKER | PASS | CRITICAL |
| `src/core/target.py` | 0% | 85% | CRITICAL |
| `src/core/planet_candidate.py` | 0% | 90% | CRITICAL |
| `src/core/transit_event.py` | 0% | 100% | HIGH |
| `src/core/periodic_event.py` | 0% | 100% | HIGH |
| `src/utils/run_json.py` | 30% | 100% | HIGH |
| `src/utils/run_context.py` | 0% | 100% | HIGH |
| `src/utils/find_total_csv.py` | 0% | 100% | HIGH |
| `src/utils/segments.py` | 0% | 95% | HIGH |
| `src/utils/running_median.py` | 0% | 90% | MEDIUM |
| `src/utils/config.py` | N/A | PASS | LOW |
| `src/stages/dataprep.py` | ~30% | 80% | MEDIUM |
| `src/stages/search_singles.py` | ~30% | 75% | MEDIUM |
| `src/stages/search_periodic.py` | ~5% | 60% | MEDIUM |
| `src/utils/alias_dedup.py` | 0% | 75% | MEDIUM |
| `src/utils/singles_periodicity.py` | 0% | 70% | MEDIUM |
| `src/utils/ticid_input_coordination.py` | 0% | 90% | MEDIUM |
| `src/utils/queue.py` | 0% | 90% | LOW |
| `src/scripts/*.py` | indirect | 60-80% | OPTIONAL |

---

## Next Steps

1. **Fix conftest.py syntax** - Lines 89, 92 need proper indentation
2. **Run existing tests** to confirm they pass after conftest.py fix
3. **Start Phase 2** - Core modules coverage (target.py, planet_candidate.py, transit_event.py, periodic_event.py)
4. **Measure coverage** after each phase
5. **Prioritize** based on project importance and test complexity
