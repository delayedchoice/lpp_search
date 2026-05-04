# Exoplanet Detection Pipeline - Prefect Orchestration Plan

## Overview

Distributed pipeline architecture for LPP Search exoplanet detection using **Prefect v3** with GPU acceleration on **NVIDIA CUDA (Ubuntu)** or **Apple Silicon MPS**.

### Data Model
Each target directory (`target_<ticid>-gaiaID_<id>/`) contains:
- `target.state.json` - current pipeline stage
- `*_total.csv` - merged lightcurve
- `candidates/run_<timestamp>.json` - candidate artifacts

**Distribution**: Workers receive target path + process_id; data pre-copied locally. Only **path reference** and **process_id** shared.

---

## Pipeline Stages

### Stage 0: Data Preparation (01_prepare_targets.py)
- **Input**: `{target_dir, process_id}` pairs
- **Output**: `*_total.csv`, `tic_star_parameters.csv`
- **GPU**: CPU-only

### Stage 1: Quick Singles Detection (02_run_quick_singles.py)
- **Input**: `{target_dir, process_id}` pairs
- **Output**: `candidates/run_<timestamp>.json` with DT pass-1 events
- **GPU**: CPU-only (DeepTransit)

### Stage 2: Periodic Search (03_run_periodic_search.py)
- **Input**: `{target_dir, process_id}` pairs
- **Output**: `candidates/run_<timestamp>.json` with periodic events
- **GPU**: CPU-only (BLS power spectrum)

### Stage 3: MCMC Fit & Refinement (04_run_fit_refine.py)
- **Input**: `{target_dir, process_id}` pairs
- **Output**: `final_candidates.csv`
- **GPU**: **GPU-accelerated** (CUDA on Ubuntu/NVIDIA, MPS on Apple Silicon)

---

## Deployment Modes

### Single Host (Local Workers)
```bash
pip install prefect torch
export PREFECT_HOME=./prefect
prefect server start
prefect agent start -q "local" --pool-size 8
prefect deployment run exoplanet-pipeline
```

### Multi-Host (Ubuntu Cluster + NVIDIA GPUs)
```bash
export PREFECT_SERVER_URL=https://prefect.example.com
export PREFECT_API_KEY=<api_key>
prefect deployment build -f flows/exoplanet.py -a cluster-pipeline
prefect deployment apply cluster-pipeline.yaml
prefect agent start -q "nvidia-gpu" --pool-size 4
prefect agent start -q "cpu-candidates" --pool-size 16
```

---

## Prefect Flow Structure

### Flow Execution Model

The `exoplanet_pipeline()` flow runs **one target through all four stages sequentially**, then **round-robots to the next target**. Each target is assigned a unique `process_id` (UUID) at discovery, which threads through all subsequent stages for end-to-end tracking.

```python
from prefect import flow, task, get_run_logger
from pathlib import Path
from typing import List, Dict, Any
from uuid import uuid4

@task
def discover_target_dirs() -> List[Dict[str, Any]]:
    """
    Discover all target_* directories.
    Returns list of {path: Path, process_id: str} objects.
    """
    import glob
    target_dirs = sorted(glob.glob("./toi_data/target_*"))
    return [{"path": Path(td), "process_id": str(uuid4())} for td in target_dirs]

@task(log_prints=True)
def prepare_one_target(payload: Dict[str, Any]) -> Dict[str, Any]:
     """Stage 0: Prepare target data."""
    target_dir = payload["path"]
    process_id = payload["process_id"]
    
    from scripts.prepare_targets import prepare_one_target as _prepare
    _prepare(target_dir, process_id)
    
    return {
        "path": str(target_dir),
        "process_id": process_id,
        "stage": "prepared"
    }

@task(log_prints=True)
def run_quick_singles(payload: Dict[str, Any]) -> Dict[str, Any]:
      """Stage 1: Run DT pass-1."""
    target_dir = payload["path"]
    process_id = payload["process_id"]
    
    from scripts.quick_singles import run as _run_singles
    _run_singles(target_dir, process_id)
    
    return {
        "path": str(target_dir),
        "process_id": process_id,
        "stage": "singles",
        "found": True/False
    }

@task(log_prints=True)
def run_periodic_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 2: Run BLS periodic search."""
    target_dir = payload["path"]
    process_id = payload["process_id"]
    
    from scripts.periodic_search import run as _run_periodic
    _run_periodic(target_dir, process_id)
    
    return {
        "path": str(target_dir),
        "process_id": process_id,
        "stage": "periodic",
        "events": 0/1/2/...
    }

@task(log_prints=True)
def run_mcmc_fit_refine(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 3: Run PyMC MCMC."""
    target_dir = payload["path"]
    process_id = payload["process_id"]
    
    from scripts.fit_refine import run_fit_refine_for_target as _run_mcmc
    _run_mcmc(target_dir, process_id)
    
    return {
        "path": str(target_dir),
        "process_id": process_id,
        "stage": "fitted"
    }

@flow(name="exoplanet-pipeline", log_prints=True)
def exoplanet_pipeline():
    
    """
    Orchestration flow for exoplanet detection pipeline.
     Each target gets a unique process_id (UUID) at discovery,
    threaded through all stages for end-to-end tracking.
    
    Returns:
        Dict with stage completion counts, telemetry log with process_ids
         """
    
    logger = get_run_logger()
    
    # Stage discovery -> assigns process_id to each target
    target_payloads = discover_target_dirs()
    logger.info(f"Found {len(target_payloads)} targets to process")
    
     # Initialize telemetry tracking
    telemetry = {
        "total_targets": len(target_payloads),
        "prepared": 0,
        "singles_found": 0,
        "periodic_events_total": 0,
        "mcmc_fitted": 0,
        "log_entries": []
    }
    
    def log(message: str):
        logger.info(message)
        telemetry["log_entries"].append(message)
     
     # Process each target with its assigned process_id
    for payload in target_payloads:
        process_id = payload["process_id"]
        target_name = payload["path"].name
        log(f"Starting pipeline for {target_name} [process_id: {process_id}]")
        
         # Stage 0
        stage0_result = prepare_one_target.run(payload)
        telemetry["prepared"] += 1
        log(f"   {target_name} [{process_id}]: data prepared")
        
         # Stage 1 -> early exit if no detection
        stage1_result = run_quick_singles.run(payload)
        if not stage1_result.get("found", False):
            log(f"   {target_name} [{process_id}]: no transit detected, skipping stages 2-3")
            continue
        telemetry["singles_found"] += 1
        log(f"   {target_name} [{process_id}]: transit detected, proceeding to periodic search")
        
        # Stage 2
        stage2_result = run_periodic_search.run(payload)
        n_events = stage2_result.get("events", 0)
        telemetry["periodic_events_total"] += n_events
        if n_events == 0:
            log(f"   {target_name} [{process_id}]: no periodic events, skipping MCMC fit")
            continue
        log(f"   {target_name} [{process_id}]: {n_events} periodic events, proceeding to MCMC fit")
        
         # Stage 3
        stage3_result = run_mcmc_fit_refine.run(payload)
        telemetry["mcmc_fitted"] += 1
        log(f"   {target_name} [{process_id}]: MCMC complete, pipeline finished")
    
    log(f"\n=== Pipeline Summary ===")
    log(f"Total targets: {telemetry['total_targets']}")
    log(f"Prepared: {telemetry['prepared']}")
    log(f"Transits found (Stage 1): {telemetry['singles_found']}")
    log(f"Periodic events (Stage 2): {telemetry['periodic_events_total']}")
    log(f"MCMC fitted (Stage 3): {telemetry['mcmc_fitted']}")
    
    return telemetry
```

### Telemetry & Logging Example

Each log entry includes `process_id` for end-to-end tracking:

```
Found 10 targets to process
Starting pipeline for target_12345 [process_id: a1b2c3d4-e5f6-7890-abcd-ef1234567890]
  target_12345 [a1b2c3d4-e5f6-7890-abcd-ef1234567890]: data prepared
  target_12345 [a1b2c3d4-e5f6-7890-abcd-ef1234567890]: no transit detected, skipping stages 2-3
Starting pipeline for target_67890 [process_id: b2c3d4e5-f6g7-8901-bcde-f23456789012]
  target_67890 [b2c3d4e5-f6g7-8901-bcde-f23456789012]: data prepared
  target_67890 [b2c3d4e5-f6g7-8901-bcde-f23456789012]: transit detected, proceeding to periodic search
  target_67890 [b2c3d4e5-f6g7-8901-bcde-f23456789012]: 3 periodic events, proceeding to MCMC fit
  target_67890 [b2c3d4e5-f6g7-8901-bcde-f23456789012]: MCMC complete, pipeline finished
```

---

## Distribution Strategy

Workers receive **{path, process_id}** pair; data pre-copied locally:

```python
@task
def copy_to_worker(source_payload: Dict[str, Any], worker_root: Path) -> Dict[str, Any]:
    import shutil
    source = source_payload["path"]
    process_id = source_payload["process_id"]
    
    local = worker_root / source.name
    if not local.exists():
        shutil.copytree(source, local)
    
    return {"path": str(local), "process_id": process_id}
```

---

## Task Graph

```
discover_target_dirs()
       │
       │ Returns: [{path: ..., process_id: "uid-1"}, {path: ..., process_id: "uid-2"}, ...]
       ▼
[for each payload in target_payloads:]
       │
       │ payload = {path: ..., process_id: "uid-XYZ"}
       ▼
prepare_one_target(payload) ──► process_id: "uid-XYZ"
       │
       ▼
run_quick_singles(payload) ──► {found: false} ──► TERMINATE (skip rest)
       │
       │ {found: true}, process_id: "uid-XYZ"
       ▼
run_periodic_search(payload) ──► {events: 0} ──► TERMINATE (skip rest)
       │
       │ {events: N>0}, process_id: "uid-XYZ"
       ▼
run_mcmc_fit_refine(payload) ──► {stage: "fitted"}, process_id: "uid-XYZ"
```

---

## Function Signatures

All pipeline functions accept `{target_dir, process_id}`:

| Function | Signature | Returns |
|----------|-----------|---------|
| `prepare_one_target` | `(target_dir: Path, process_id: str) -> None` | `None` |
| `run_quick_singles` | `(target_dir: Path, process_id: str) -> None` | `None` |
| `run_periodic_search` | `(target_dir: Path, process_id: str) -> None` | `None` |
| `run_mcmc_fit_refine` | `(target_dir: Path, process_id: str) -> None` | `None` |

Low-level logging in each stage must include `process_id`:

```python
# Inside prepare_one_target
def prepare_one_target(target_dir: Path, process_id: str):
    logger.info(f"[{process_id}] Preparing target {target_dir.name}")
     # ... rest of logic
```

---

## Configuration

```bash
export PREFECT_HOME=/prefect/runs
export WORKER_LOCAL_ROOT=/data/workers
```

---

## Migration from Queue System

Replace `utils/queue.py` file markers with Prefect process_id tracking:

```python
# Old: file-based queue markers
# runs/current/queue/02/<ticid>

# New: process_id-based tracking
discover_target_dirs() -> [{path, process_id}, ...]
# All subsequent functions receive/process_id throughout
```

**Migration rationale**:
- Old queue uses file markers (`queue/02/<ticid>`) to signal next stage
- New approach uses `process_id` (UUID) passed through function parameters
- Eliminates polling loop and file race conditions
- Built-in retry, visibility, dashboard in Prefect
- End-to-end tracking: can trace any target through entire pipeline

---

## Summary

| Aspect | Implementation |
|--------|---------------|
| Engine | Prefect v3 |
| Processing Model | One target through all 4 stages; early exit for non-detections |
| Distribution | Pre-copy to workers; {path, process_id} sharing |
| GPU | PyTorch CUDA (Ubuntu/NVIDIA) or MPS (Apple Silicon) |
| CPU | DeepTransit + BLS (16 concurrent workers) |
| Telemetry | process_id-threaded logs; summary metrics |
| Queue | Replaced with process_id parameter passing |
| Mode | Single host (local) or multi-host (cluster) |

---

## Implementation Checklist

- [ ] `discover_target_dirs()` returns `{path, process_id}` list
- [ ] `prepare_one_target(path, process_id)` signature update
- [ ] `run_quick_singles(path, process_id)` signature update  
- [ ] `run_periodic_search(path, process_id)` signature update
- [ ] `run_mcmc_fit_refine(path, process_id)` signature update
- [ ] All low-level logging includes `process_id` prefix
- [ ] Telemetry tracking with process_id in log entries
- [ ] Flow definition uses `payload` dict instead of raw paths
