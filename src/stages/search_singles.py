# stages/search_singles.py
# Quick single-transit detection (DT) that ONLY sets Target.dt_prelim_found and quick_singles_t0.
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk
import deep_transit as dt

from core.transit_event import TransitEvent
from utils.segments import breaking_up_data
from utils.find_total_csv import find_total_csv
from utils.run_json import upsert_run_json
import utils.config as con

def make_LightKurveObject(time, flux, flux_err):
    lc = lk.TessLightCurve()
    lc.time = time; lc.flux = flux; lc.flux_err = flux_err
    return lc

def calc_rudimentary_snr(depth, Tdur, Ntran=1):
    sigma_1hr_15_Tmag = 6283.6147036936645 * 1e-6


    return (Ntran**0.5 / sigma_1hr_15_Tmag) * depth * np.sqrt(Tdur * 24)

def plot_lc_with_bboxes(lc_object, bboxes, ax=None, epoch=0, **kwargs):
    with plt.style.context('grayscale'):
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(12, 6), constrained_layout=False)
            ax.plot(lc_object.time.value, lc_object.flux.value, color='k', zorder=1e5, **kwargs)
            ax.set_xlabel('Time - T0 (hours)'); ax.set_ylabel('Normalized Flux')
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection
        recs = []
        for real_mask in bboxes:
            new_start = real_mask[1] - epoch
            rec = Rectangle(
                (new_start - real_mask[3]/2, real_mask[2] - real_mask[4]/2),
                real_mask[3], real_mask[4],
                facecolor='indianred', edgecolor='indianred', linewidth=1.0, zorder=5
            )
            recs.append(rec)
            dur = float(real_mask[3])
            depth = float(1.0 - real_mask[4])
            SNR = calc_rudimentary_snr(depth, dur)
            ax.text(new_start + abs(real_mask[3]), real_mask[2] + 0.5*abs(real_mask[4]), s=f"SNR: {SNR:.2f}", color='r')
        ax.add_collection(PatchCollection(recs, lw=0.2, match_original=True, zorder=5))
        return ax

def DT_analysis(time, flux, flux_err, confidence, DT_Quite=True, is_flat=True):
    # print('not even here?')
    if DT_Quite:
        save_stdout, save_stderr = sys.stdout, sys.stderr
        sys.stdout = open('.trash.txt', 'w'); sys.stderr = open('.trash.txt', 'w')
    model = dt.DeepTransit(make_LightKurveObject(time, flux, flux_err), is_flat=is_flat)
    bboxes = model.transit_detection(str(con.MODEL_PATH), confidence_threshold=confidence)
    if DT_Quite:
        sys.stdout.close(); sys.stderr.close()
        sys.stdout, sys.stderr = save_stdout, save_stderr
    return bboxes

@dataclass
class SinglesSearchConfig:
    flavour: str = "TGLC"
    confidence: float = 0.75
    plot_events: bool = False
    verbose: bool = False


def detect_transit_events(time, flux, flux_err, cfg):
    """
    Run DeepTransit and return:
      - events: list[TransitEvent]
      - bboxes: raw DT bboxes (useful for plotting)
    This is event-level output (NOT planet candidates yet).
    """
    bboxes = DT_analysis(time, flux, flux_err, cfg.confidence, DT_Quite=True, is_flat=True)
    events = []
    
    print('bboxes', type(bboxes), bboxes)
    if len(bboxes) == 0:
        return events, bboxes

    for boxes in bboxes:
        # Your established bbox convention (as in legacy):
        # t0=boxes[1], dur=boxes[3], depth=1-boxes[4]
        T0 = float(boxes[1])
        Tdur = float(boxes[3])
        depth = float(1.0 - boxes[4])

        # Optional confidence-like value (only if 0..1)
        conf = None
        try:
            conf_val = float(boxes[0])
            if 0.0 <= conf_val <=1.0:
                conf = conf_val
        except Exception:
            pass

        # Rudimentary SNR (depth, duration) using your helper
        snr = None
        try:
            snr = float(calc_rudimentary_snr(depth, Tdur))
        except Exception:
            pass

        events.append(TransitEvent(
            t0_days=T0,
            duration_days=Tdur,
            depth=depth,
            snr=snr,
            confidence=conf,
        ))

    return events, bboxes



def singles_search(target, *, cfg=SinglesSearchConfig(), run_1=True,
                   exclude_mask=None, pass_label="pass1",
                   run_id=None, run_path=None):
    """
    Quick DT single-transit detection.

    - Pass 1 (default): run on full merged total LC.
    - Pass 2 (later, in finalize): run on residual LC using exclude_mask
      (typically periodic in-transit points).

    This stage:
      - updates Target.dt_prelim_found and Target.quick_singles_t0
      - writes a run artifact candidates/run_<run_id>.json with dt_events_raw_<pass_label>
      - returns (event_df, params_df) for compatibility with existing code/tests
    """
    ticid = int(target.ticid)
    total_csv = find_total_csv(target.root_dir, cfg.flavour)  # existing helper

    df = pd.read_csv(total_csv).dropna(subset=["FLUX"])
    total_time = df["TIME"].to_numpy(dtype=float)
    total_flux = df["FLUX"].to_numpy(dtype=float)


    if "FLUX_ERR" in df.columns:
        total_flux_err = df["FLUX_ERR"].to_numpy(dtype=float)
    else:
        # keep your pragmatic fallback (constant scatter) 
        total_flux_err = np.full_like(total_flux, np.nanstd(total_flux))

    # Optional exclusion mask (used for DT pass 2 later)
    if exclude_mask is not None:
        exclude_mask = np.asarray(exclude_mask, dtype=bool)
        if exclude_mask.shape == total_time.shape:
            keep = ~exclude_mask
            total_time = total_time[keep]
            total_flux = total_flux[keep]
            total_flux_err = total_flux_err[keep]

    # Optional segment quality cut (mirrors your legacy intent
    # Keep only segments with >= ~1 day span; otherwise keep all.
    if total_time.size > 0:
        idx_blocks = breaking_up_data(total_time, break_val=0.5, min_size=1.0)
        if len(idx_blocks) > 1:
            spans = np.array([np.ptp(total_time[idx]) for idx in idx_blocks])
            good_blocks = [idx_blocks[i] for i in range(len(idx_blocks)) if spans[i] > 1.0]
            if len(good_blocks) > 0:
                good_idx = np.concatenate(good_blocks)
                total_time = total_time[good_idx]
                total_flux = total_flux[good_idx]
                total_flux_err = total_flux_err[good_idx]

    # --- DT detection (this is the core) ---
    events, bboxes = detect_transit_events(total_time, total_flux, total_flux_err, cfg)

    # --- Update Target quick-singles state (existing contract) ---
    target.dt_prelim_found = (len(events) > 0)
    target.quick_singles_t0 = sorted({float(e.t0_days) for e in events})
    target.save_state()  # persists dt_prelim_found + quick_singles_t0

    # choose run_id/run_path (finalize should pass these)
    if run_id is None:
        run_id = target.new_run_id()

    run_path = target.candidates_run_path(run_id)    
    
    payload_update = {
        "run_id": run_id,
        "ticid": int(target.ticid),
        "gaia_id": target.gaia_id,
        "total_csv": str(total_csv),
        "dt_config": {"flavour": cfg.flavour, "confidence": cfg.confidence},
        f"dt_events_raw_{pass_label}": [e.to_dict() for e in events],
    }

    upsert_run_json(run_path, payload_update)

    # Optionally store pointers if you added them earlier (safe if missing)
    if hasattr(target, "last_run_id"):
        target.last_run_id = run_id
    if hasattr(target, "last_candidates_run"):
        target.last_candidates_run = str(run_path.relative_to(target.root_dir))
    try:
        target.save_state()
    except Exception:
        pass

    # --- Optional plotting (kept here, not in helper) ---
    if cfg.plot_events and bboxes:
        # minimal shim compatible with plot_lc_with_bboxes
        shim = type("Obj", (), {})()
        shim.time = type("Arr", (), {"value": total_time})()
        shim.flux = type("Arr", (), {"value": total_flux})()

        for boxes in bboxes:
            T0 = float(boxes[1])
            Tdur = float(boxes[3])

            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            ax.set_xlim(T0 - 2*Tdur, T0 + 2*Tdur)
            ax.scatter(total_time, total_flux, color="k", s=6, zorder=10)
            plot_lc_with_bboxes(shim, bboxes, ax=ax)
            plt.show()

    # --- Compatibility return frames (keep for now) ---
    # This is your old "planet_df" style output; we leave it so callers/tests won't break.
    column_names = ["TICID", "planet_name", "period", "T0", "Tdur", "depth"]
    if not run_1:
        column_names.append("SNR")

    event_df = pd.DataFrame(columns=column_names)
    for k, e in enumerate(events, start=1):
        row = [ticid, k, np.inf, float(e.t0_days), float(e.duration_days), float(e.depth)]
        if not run_1:
            row.append(np.nan if e.snr is None else float(e.snr))
        event_df.loc[len(event_df)] = row

    params_df = pd.DataFrame()  # keep empty for quick pass

    return event_df, params_df
    
