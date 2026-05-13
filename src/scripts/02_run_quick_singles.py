#!/usr/bin/env python
import glob
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.target import Target, PipelineStage
from stages.search_singles import singles_search, SinglesSearchConfig
from utils.queue import enqueue

TARGET_GLOB = str(PROJECT_ROOT.parent / "toi_data" / "target_*")


def _has_merged_data(target: Target, flavour: str) -> bool:
    try:
        target.load_state()
    except Exception:
        pass

    # Primary gate: anything MERGED-or-beyond is acceptable
    if target.stage_at_least(PipelineStage.MERGED):
        return True

    # Fallback: look for merged total on disk
    rd = target.root_dir
    return any(rd.glob(f"*{flavour}*_*total.csv")) or any(rd.glob("*total.csv"))

def main(idx):
    dirs = sorted(glob.glob(TARGET_GLOB))
    if not (0 <= idx < len(dirs)):
        print(f"[FATAL] idx={idx} out of range for {len(dirs)} targets.")
        sys.exit(2)

    root = Path(dirs[idx])
    ticid, gaia_id = Target.discover_ids_from_dirname(root)
    t = Target(ticid=int(ticid), gaia_id=gaia_id, root_dir=root)
    t.load_state()

    if not _has_merged_data(t, flavour=t.data_source.value):
        print(f"[{root.name}] Not ready (no merged total CSV). Skipping.")
        return


    cfg = SinglesSearchConfig(flavour=t.data_source.value, confidence=0.55, plot_events=False, verbose=False)
    # Optional: if you want consistent per-run artifacts, uncomment:
    run_id = t.new_run_id()

    run_path = t.candidates_run_path(run_id)


    # Otherwise (simpler): let singles_search manage its own run context
    planet_df, _ = singles_search(t, cfg=cfg, run_1=True, pass_label="pass1")
    print('planet_df', planet_df)

    found = bool(getattr(t, "dt_prelim_found", False))
    print('found', found)

    # Mark quick singles done
    
    if not t.stage_at_least(PipelineStage.SEARCHED):
        t.set_stage(PipelineStage.SEARCHED)

    else:
        # already beyond SEARCHED (FITTED/REPORTED), leave it alone
        t.save_state()


    if found:
        enqueue("03", t.ticid)
    else:
        enqueue("DONE_EMPTY", t.ticid)

    print(f"[TIC {t.ticid}] {'FOUND' if found else 'no'} transit-like signal")


if __name__ == "__main__":
    idx_str = os.environ.get("SLURM_ARRAY_TASK_ID") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if idx_str is None:
        print("Usage: python scripts/02_run_quick_singles.py <index>  # or SLURM_ARRAY_TASK_ID")
        sys.exit(1)
    main(int(idx_str))
