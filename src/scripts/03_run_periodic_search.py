#!/usr/bin/env python
import glob
import os
import sys
from pathlib import Path
from datetime import datetime
import json


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.target import Target, PipelineStage
from stages.search_periodic import periodic_search, PeriodicSearchConfig
from utils.run_json import upsert_run_json, append_run_json_list
from core.transit_event import TransitEvent

from utils.singles_periodicity import seed_periods_from_dt_events
from utils.queue import enqueue

TARGET_GLOB = "../../toi_data/target_*"   # adjust as needed

def main(idx):
    dirs = sorted(glob.glob(TARGET_GLOB))
    if not (0 <= idx < len(dirs)):
        print(f"[FATAL] idx={idx} out of range for {len(dirs)} targets.")
        sys.exit(2)

    root = Path(dirs[idx])
    target = Target.from_dir(root)

    # Gate: DT pass‑1 must have been performed at least once
    if not target.stage_at_least(PipelineStage.SEARCHED):
        print(f"[FATAL] {root.name}: need DT pass‑1 first (stage < SEARCHED). Run script 02.")
        sys.exit(3)


    # Gate: DT pass-1 must have found something
    if not bool(getattr(target, "dt_prelim_found", False)):
        print(f"[SKIP] {root.name}: dt_prelim_found=False (skip periodic).")
        return

    # Use the most recent DT run file as the provenance anchor
    last_rel = getattr(target, "last_candidates_run", None)
    if not last_rel:
        print(f"[FATAL] {root.name}: no last_candidates_run in state; run script 02 first.")
        sys.exit(3)

    run_path = (target.root_dir / last_rel).resolve()
    if not run_path.exists():
        print(f"[FATAL] {root.name}: last_candidates_run points to missing file: {run_path}")
        sys.exit(4)
    
    run_json = json.loads(run_path.read_text())
    run_id = run_path.stem.replace("run_", "")
    attempt_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # Status breadcrumb (start)
    upsert_run_json(run_path, {
        "status": {
            "stage": "periodic_search",
            "state": "running",
            "attempt_id": attempt_id,
            "updated_at": datetime.now().isoformat(),
        }
    })

    

    raw = run_json.get("dt_events_raw_pass1", [])
    dt_events = [TransitEvent.from_dict(d) for d in raw] if isinstance(raw, list) else []    
    seed_periods = seed_periods_from_dt_events(dt_events, top_k=10, min_support=3)

    upsert_run_json(run_path, {
        "seed_periods_days": seed_periods,
        "seed_source": "dt_pass1_singles_periodicity",
        "seed_generated_at": datetime.now().isoformat(),
    })

    cfg = PeriodicSearchConfig(flavour=target.data_source.value)
    if len(seed_periods) > 0:
        cfg.use_seed_periods = True

        periodic_events = periodic_search(
            target,
            cfg=cfg,
            seed_periods=seed_periods,
            run_id=run_id,
            run_path=run_path
        )

    else:
        periodic_events = periodic_search(
            target,
            cfg=cfg,
            seed_periods=None,
            run_id=run_id,
            run_path=run_path
        )
    # Append attempt history
    attempt_record = {
        "attempt_id": attempt_id,
        "flavour": cfg.flavour,
        "data_source": target.data_source.value,
        "min_snr": cfg.min_snr,
        "min_sde": cfg.min_sde,
        "max_planets": cfg.max_planets,
        "max_iters": cfg.max_iters,
        "use_seed_periods": cfg.use_seed_periods,
        "seed_window_frac": cfg.seed_window_frac,
        "seed_grid_size": cfg.seed_grid_size,
        "power_baseline_kernel": cfg.power_baseline_kernel,
        "df_unc": cfg.df_unc,
        "n_periodic_events": int(len(periodic_events)),
        "periodic_events_raw": [pe.to_dict() for pe in periodic_events],
        "finished_at": datetime.now().isoformat(),
    }
    append_run_json_list(run_path, "periodic_attempts", attempt_record)

    # Also keep a "latest" snapshot for convenience (optional)
    upsert_run_json(run_path, {
        "periodic_events_raw_latest": attempt_record["periodic_events_raw"],
        "periodic_attempt_latest_attempt_id": attempt_id,
        "status": {
            "stage": "periodic_search",
            "state": "done",
            "attempt_id": attempt_id,
            "n_periodic_events": int(len(periodic_events)),
            "updated_at": datetime.now().isoformat(),
        }
    })
    enqueue("04", target.ticid)
    print(f"[DONE] {root.name}: periodic_events={len(periodic_events)} attempt_id={attempt_id}")


if __name__ == "__main__":
    idx_str = os.environ.get("SLURM_ARRAY_TASK_ID") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if idx_str is None:
        print("Usage: python scripts/02_run_quick_singles.py <index>  # or SLURM_ARRAY_TASK_ID")
        sys.exit(1)
    main(int(idx_str))
