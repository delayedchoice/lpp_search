#!/usr/bin/env python
import os

import glob
from runpy import run_path
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from core.target import Target, PipelineStage
from core.planet_candidate import PlanetCandidate
from core.transit_event import TransitEvent
from core.periodic_event import PeriodicEvent

from utils.find_total_csv import find_total_csv
from utils.run_json import upsert_run_json, append_run_json_list
from utils.singles_periodicity import periodic_modes_from_dt_events, periodic_candidates_from_modes, mark_single_members_consumed


from utils.alias_dedup import alias_dedup_periodic_candidates
from utils.queue import enqueue

from stages.search_singles import singles_search, SinglesSearchConfig
from engines.pyMC_core import pymc_fit_candidate

TARGET_GLOB = "../../toi_data/target_*"   # adjust


# ---------------------------
# I/O helpers
# ---------------------------
def load_run_json(run_path: Path) -> dict:
    return json.loads(run_path.read_text())

def write_final_candidates_csv(target: Target, candidates: list[PlanetCandidate]) -> Path:
    out_path = target.root_dir / "final_candidates.csv"

    rows = []
    for c in candidates:
        rows.append({
            "ticid": target.ticid,
            "gaia_id": target.gaia_id,
            "candidate_id": c.candidate_id(),
            "ptype": c.ptype,
            "t0_days": c.t0_days,
            "period_days": c.period_days,
            "duration_days": c.duration_days,
            "depth": c.depth,
            "n_transits_obs": c.n_transits_obs,
            "fit_is_current": c.fit_is_current,
            "source": c.source,
            "notes": c.notes,
            # store the full summary stats as JSON text so you keep *everything*
            "pymc_summary_json": json.dumps(c.pymc_summary) if c.pymc_summary else "",
        })

    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path

def append_global_candidates_csv(candidates: list[PlanetCandidate], target: Target, global_path: Path) -> None:
    # NOTE: if you run this in parallel across many targets, concurrent appends can collide.
    # In that case, write per-target CSVs and merge later in one job.
    rows = []
    for c in candidates:
        rows.append({
            "ticid": target.ticid,
            "gaia_id": target.gaia_id,
            "candidate_id": c.candidate_id(),
            "ptype": c.ptype,
            "t0_days": c.t0_days,
            "period_days": c.period_days,
            "duration_days": c.duration_days,
            "depth": c.depth,
            "n_transits_obs": c.n_transits_obs,
            "fit_is_current": c.fit_is_current,
            "source": c.source,
            "notes": c.notes,
            "pymc_summary_json": json.dumps(c.pymc_summary) if c.pymc_summary else "",
        })

    df = pd.DataFrame(rows)
    write_header = not global_path.exists()
    df.to_csv(global_path, mode="a", header=write_header, index=False)


# ---------------------------
# Fit helpers
# ---------------------------
def fit_and_attach(target: Target, cand: PlanetCandidate, time, flux, unc, run_path: Path, verbose: bool=False) -> bool:
    """
    Runs PyMC fit and attaches *full summary stats* to the candidate.
    """
    attempt_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    upsert_run_json(run_path, {"status": {"stage": "pymc_fit", "state": "running", "attempt_id": attempt_id}})

    summary_df, ok, _ = pymc_fit_candidate(target, cand, time, flux, unc, verbose=verbose)

    if ok and summary_df is not None:
        cand.pymc_summary = summary_df.to_dict()
        cand.mark_fitted()

        # Update working hypothesis from PyMC medians
        cand.t0_days = _summary_median(cand, "t0", fallback=cand.t0_days)
        if cand.ptype == "Periodic":
            cand.period_days = _summary_median(cand, "Per", fallback=cand.period_days)
        cand.duration_days = _summary_median(cand, "dur", fallback=cand.duration_days)
        cand.depth = _summary_median(cand, "depth", fallback=cand.depth)    
    else:
        cand.fit_is_current = False

    append_run_json_list(run_path, "fit_attempts", {
        "attempt_id": attempt_id,
        "candidate_id": cand.candidate_id(),
        "ptype": cand.ptype,
        "ok": bool(ok),
        "finished_at": datetime.now().isoformat()
    })

    upsert_run_json(run_path, {"status": {"stage": "pymc_fit", "state": "done", "attempt_id": attempt_id}})
    return bool(ok)


def finalize_pass1_singles_only(target, run_path, run_json, global_csv_path):
    raw_pass1 = run_json.get("dt_events_raw_pass1", [])
    pass1_events = [TransitEvent.from_dict(d) for d in raw_pass1] if isinstance(raw_pass1, list) else []

    flavour = target.data_source.value
    total_csv = find_total_csv(target.root_dir, flavour)
    df = pd.read_csv(total_csv).dropna(subset=["FLUX"])
    time = df["TIME"].to_numpy(float)
    flux = df["FLUX"].to_numpy(float)
    unc = df["FLUX_ERR"].to_numpy(float) if "FLUX_ERR" in df.columns else np.full_like(flux, np.nanstd(flux))

    single_candidates = []
    for ev in pass1_events:
        sc = PlanetCandidate(
            ptype="Single",
            t0_days=float(ev.t0_days),
            period_days=None,
            duration_days=float(ev.duration_days),
            depth=float(ev.depth),
            source="DT_pass1",
        )
        fit_and_attach(target, sc, time, flux, unc, run_path, verbose=False)
        single_candidates.append(sc)

    per_target_csv = write_final_candidates_csv(target, single_candidates)
    append_global_candidates_csv(single_candidates, target, global_csv_path)

    target.set_stage(PipelineStage.FITTED)
    enqueue("DONE_FOUND", target.ticid) 
    upsert_run_json(run_path, {"status": {"stage": "fit_refine", "state": "done_no_periodic", "finished_at": datetime.now().isoformat()}})
    print(f"[DONE] {target.root_dir.name}: periodic=0; wrote {per_target_csv}")

def _summary_median(cand, varname, fallback=None):
    """
    cand.pymc_summary is a dict-of-dicts from summary_df.to_dict().
    Expected structure: cand.pymc_summary["Per"]["median"], etc.
    """
    try:
        d = cand.pymc_summary.get(varname, None)
        if isinstance(d, dict) and "median" in d:
            return float(d["median"])
    except Exception:
        pass
    return fallback


def periodic_mask_from_fitted_candidate(time: np.ndarray, cand: PlanetCandidate,
                                        buffer_days: float = 0.2) -> np.ndarray:
    """
    Build a full-length in-transit mask using PyMC medians:
      - Per median
      - t0 median
      - dur median (PyMC deterministic, full duration in days)
    """

    # Pull fitted medians
    P  = _summary_median(cand, "Per", fallback=cand.period_days)
    t0 = _summary_median(cand, "t0", fallback=cand.t0_days)

    # Your PyMC model defines pm.Deterministic("dur", ...)
    dur = _summary_median(cand, "dur", fallback=cand.duration_days)

    if P is None or dur is None or t0 is None:
        return np.zeros_like(time, dtype=bool)

    P = float(P); t0 = float(t0); dur = float(dur)

    phase = np.abs(((time - t0 + 0.5 * P) % P) - 0.5 * P)
    return phase < (0.5 * dur + buffer_days)


# ---------------------------
# Main per-target pipeline
# ---------------------------
def run_fit_refine_for_target(target: Target, global_csv_path: Path) -> None:
    target.load_state()

    # Gate: must have DT pass-1 run file
    last_rel = getattr(target, "last_candidates_run", None)
    if not last_rel:
        print(f"[SKIP] {target.root_dir.name}: no last_candidates_run (run 02 first).")
        return

    run_path = (target.root_dir / last_rel).resolve()
    if not run_path.exists():
        print(f"[SKIP] {target.root_dir.name}: last_candidates_run missing on disk.")
        return

    # Load run json + periodic latest
    run_json = load_run_json(run_path)

    periodic_raw = run_json.get("periodic_events_raw_latest", None)



    if not periodic_raw:
        attempts = run_json.get("periodic_attempts", [])
        if attempts:
            periodic_raw = attempts[-1].get("periodic_events_raw", [])

    if not periodic_raw:
        finalize_pass1_singles_only(target, run_path, run_json, global_csv_path)
        return
    
    

    periodic_events = [PeriodicEvent.from_dict(d) for d in periodic_raw]

    # Load merged total for fitting arrays
    flavour = target.data_source.value
    total_csv = find_total_csv(target.root_dir, flavour)
    df = pd.read_csv(total_csv).dropna(subset=["FLUX"])
    time = df["TIME"].to_numpy(float)
    flux = df["FLUX"].to_numpy(float)
    unc = df["FLUX_ERR"].to_numpy(float) if "FLUX_ERR" in df.columns else np.full_like(flux, np.nanstd(flux))

    upsert_run_json(run_path, {"status": {"stage": "fit_refine", "state": "running", "updated_at": datetime.now().isoformat()}})

    # 1) Convert periodic events -> periodic candidates and fit them
    periodic_candidates = []
    for ev in periodic_events:
        if ev.duration_days is None or ev.depth is None:
            upsert_run_json(run_path, {
                "warnings": [f"Skipped periodic event with missing duration/depth: {ev.to_dict()}"]
            })
            continue
        pc = PlanetCandidate(
            ptype="Periodic",
            t0_days=float(ev.t0_days),
            period_days=float(ev.period_days),
            duration_days=float(ev.duration_days) if ev.duration_days is not None else None,
            depth=float(ev.depth) if ev.depth is not None else None,
            n_transits_obs=ev.n_transits_obs,
            transit_times_days = ev.transit_times_days,
            source="BLS",
        )
        fit_and_attach(target, pc, time, flux, unc, run_path, verbose=False)
        periodic_candidates.append(pc)

    # 2) Mask using fitted periodic candidates ONLY
    intransit = np.zeros_like(time, dtype=bool)
    for pc in periodic_candidates:
        if pc.fit_is_current:
            intransit |= periodic_mask_from_fitted_candidate(time, pc, buffer_days=0.2)

    have_mask = bool(intransit.any())

    # 3) DT pass-2 (residual) ONLY if we actually masked something

    pass2_events = []

    if have_mask:
        singles_cfg = SinglesSearchConfig(flavour=flavour, confidence=0.55, plot_events=False, verbose=False)
        singles_search(
            target,
            cfg=singles_cfg,
            exclude_mask=intransit,
            pass_label="pass2",
            run_id=run_path.stem.replace("run_", ""),
            run_path=run_path
        )
        # Reload run json to get pass2 events

        run_json = load_run_json(run_path)
        raw_pass2 = run_json.get("dt_events_raw_pass2", [])
        pass2_events = [TransitEvent.from_dict(d) for d in raw_pass2] if isinstance(raw_pass2, list) else []

    print("lngth pass2_events: ", len(pass2_events))

    # 4) Fit the pass2 events as singles (then later: promote periodic if periodicity emerges)
    single_candidates = []
    for ev in pass2_events:
        sc = PlanetCandidate(
            ptype="Single",
            t0_days=float(ev.t0_days),
            period_days=None,
            duration_days=float(ev.duration_days),
            depth=float(ev.depth),
            snr=None if ev.snr is None else float(ev.snr),
            source="DT_pass2",
            transit_times_days=[float(ev.t0_days)]
        )
        fit_and_attach(target, sc, time, flux, unc, run_path, verbose=False)
        single_candidates.append(sc)

    # 5) Optional promotion: check if pass2 DT events contain periodic modes
    # (You said: if periodic fit fails, leave them as singles.)
    modes = periodic_modes_from_dt_events(pass2_events, min_support=3, use_depth=True)

    promotions = periodic_candidates_from_modes(
        modes,
        pass2_events,
        source="DT_PASS2_PROMOTED",
        min_support=3,
        notes_prefix="promoted_from_pass2; "
    )

    promoted_periodic_candidates = []
    for pc, member_idx in promotions:
        ok = fit_and_attach(target, pc, time, flux, unc, run_path, verbose=False)
        if ok:
            promoted_periodic_candidates.append(pc)
            # ONLY consume singles if periodic fit succeeds
            mark_single_members_consumed(single_candidates, member_idx, pc.candidate_id())


    alias_dedup_periodic_candidates(periodic_candidates + promoted_periodic_candidates)

    # 6) Write outputs (no PDFs)
    final_candidates = periodic_candidates + promoted_periodic_candidates + single_candidates 

    per_target_csv = write_final_candidates_csv(target, final_candidates)
    append_global_candidates_csv(final_candidates, target, global_csv_path)

    # Update stage
    target.set_stage(PipelineStage.FITTED)
    enqueue("DONE_FOUND", target.ticid)
    upsert_run_json(run_path, {"status": {"stage": "fit_refine", "state": "done", "finished_at": datetime.now().isoformat()}})
    print(f"[DONE] {target.root_dir.name}: wrote {per_target_csv} and appended to {global_csv_path}")


def main(idx: int) -> None:
    dirs = sorted(glob.glob(TARGET_GLOB))
    if not (0 <= idx < len(dirs)):
        print(f"[FATAL] idx={idx} out of range for {len(dirs)} targets.")
        sys.exit(2)

    root = Path(dirs[idx])
    target = Target.from_dir(root)
    print('Target:', target)

    global_csv = Path.cwd() / "all_final_candidates.csv"
    run_fit_refine_for_target(target, global_csv)


if __name__ == "__main__":
    idx_str = os.environ.get("SLURM_ARRAY_TASK_ID") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if idx_str is None:
        print("Usage: python scripts/04_run_fit_refine.py <index>  # or SLURM_ARRAY_TASK_ID")
        sys.exit(1)
    main(int(idx_str))
