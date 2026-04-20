# stages/search_periodic.py
# Periodic transit search stage (BLS/TLS later) operating on a Target.
"""
Where things stand 4/18:
seed pre-pass uses max_iters=1
plots currently off (if you turned them off)
full + chunked search preserved
write periodic_events_raw to run JSON
repeat ephemeris check uses duration-scaled tolerance
strict stop on thresholds
"""


from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from core.target import Target
import config as con

from astropy.timeseries import BoxLeastSquares
import scipy.stats as sst
from datetime import datetime
import matplotlib.pyplot as plt
from core.periodic_event import PeriodicEvent

from utils.find_total_csv import find_total_csv
from utils.running_median import running_median
from utils.run_json import upsert_run_json
from utils.segments import breaking_up_data


@dataclass
class PeriodicSearchConfig:
    flavour: str = "TGLC"

    # strict stop thresholds (active behavior)
    min_snr: float = 7.0
    min_sde: float = 10.0

    # caps
    max_planets: int = 10
    max_iters: int = 50

    # verbosity
    verbose: bool = True
    plots  : bool = False

    # masking controls
    transit_mask_base_buffer_days: float = 0.2
    max_widen_factor: int = 5

    # repeat-ephemeris tolerances (clear names; no eps)
    period_rel_tol: float = 1e-3
    epoch_tol_floor_days: float = 0.02
    epoch_tol_frac_of_duration: float = 0.25  # duration-scaled (your preference)

    # optional seeded pre-pass later (off by default)
    use_seed_periods: bool = False
    seed_window_frac: float = 0.02  # +/- 2% around seed periods
    seed_grid_size: int = 400           # how fine to scan within the window

    power_baseline_kernel: int = 25
    use_iteration_limit_for_threshold_failures = False  # optional continue if threshold failure if you want it

    df_unc = 1E-4

def transit_mask(time: np.ndarray, period: float, duration: float, t0: float, buffer: float) -> np.ndarray:
    """
    Full-length transit mask (days). Mirrors your legacy modulo-based mask.
    """
    t = np.asarray(time, dtype=float)
    P = float(period)
    return np.abs((t - t0 + 0.5 * P) % P - 0.5 * P) < (duration + buffer)



def _nearest_epoch_time(t: float, t0: float, P: float) -> float:
    """Nearest transit center predicted by (t0, P) to time t."""
    n = int(np.rint((t - t0) / P))
    return t0 + n * P


def _offset_to_ephemeris(t: float, t0: float, P: float) -> float:
    """Absolute time offset of t from nearest epoch of ephemeris (t0, P)."""
    return abs(t - _nearest_epoch_time(t, t0, P))



def epoch_tolerance_days(duration_days: float, cfg: PeriodicSearchConfig) -> float:
    """Duration-scaled epoch tolerance (your preference)."""
    return max(cfg.epoch_tol_floor_days, cfg.epoch_tol_frac_of_duration * float(duration_days))



def is_repeat_ephemeris(period, t0, duration, accepted, cfg: PeriodicSearchConfig) -> int | None:
    """
    Return index of accepted candidate that matches new_cand by:
    - same period within period_rel_tol
    - same phase modulo P within duration-scaled tolerance
    """
    Pn = float(period)
    t0n = float(t0)
    dn = float(duration)
    for i, old in enumerate(accepted):

        Po = float(old.period_days)
        t0o = float(old.t0_days)
        if abs(Pn - Po) / max(Po, 1e-12) > cfg.period_rel_tol:
            continue
        # epoch/phase match (mod P)
        tol = epoch_tolerance_days(dn, cfg)
        if _offset_to_ephemeris(t0n, t0o, Po) <= tol:
            return i
    return None


def compute_sde(power_final: np.ndarray, idx_best: int) -> float:
    """
    Simple SDE: (peak - median) / std.
    If you already have a preferred SDE definition, swap it here.
    """
    pf = np.asarray(power_final, dtype=float)
    med = np.nanmedian(pf)
    sig = np.nanstd(pf)
    if not np.isfinite(sig) or sig == 0:
        return 0.0
    return float((pf[idx_best] - med) / sig)


def seed_period_grid(P0, window_frac, ngrid):
    lo = P0 * (1.0 - window_frac)
    hi = P0 * (1.0 + window_frac)
    return np.linspace(lo, hi, int(ngrid))


def normalize_depth_to_fractional(depth_val):
    """
    Force depth into (0, 0.5] as a positive fractional transit depth.
    If something arrives as ~0.997 (i.e., flux level), convert to ~0.003.
    """
    if depth_val is None:
        return None
    d = float(depth_val)
    if not np.isfinite(d):
        return None
    d = abs(d)  # guard against negative sign conventions
    if d > 0.5:
        d = 1.0 - d
    return d


def checking_last_BLS_power_for_artificial_inflation(power_results):
    max_indx = 0
    if max(power_results) == power_results[-1]:
    
        max_indx = 1
        rev_power_results = power_results[::-1]
        for pwr in rev_power_results:
            if pwr == power_results[-1]:
                max_indx+=1
            else:
                break
    if max_indx == 0 or max_indx >= len(power_results):
        return np.arange(len(power_results))
    else:        
        return np.arange(len(power_results)-max_indx)


def make_power_final(power_raw, cfg):
    baseline = running_median(power_raw, kernel=cfg.power_baseline_kernel)
    # avoid divide-by-zero / weird baselines
    baseline = np.where(baseline == 0, np.nanmedian(baseline), baseline)
    return np.array(power_raw, dtype=float) / baseline


def best_peak_index_from_power(power_raw, cfg):
    power_final = make_power_final(power_raw, cfg)
    keep = checking_last_BLS_power_for_artificial_inflation(power_final)
    idx = keep[int(np.argmax(power_final[keep]))]
    return idx, power_final




def evaluate_best_peak(model, results, idx, power_final, cfg, accepted_events):
    period = float(results.period[idx])
    t0 = float(results.transit_time[idx])
    duration = float(results.duration[idx])
    depth = normalize_depth_to_fractional(results.depth[idx])  # your new policy

    # Compute SDE
    mad = sst.median_abs_deviation(power_final)
    print('mad', mad)
    if mad == 0.:
        print('mad == 0: standard deviation is', np.std(power_final), ', mad without running median is', sst.median_abs_deviation(results.power))

        mad = np.nanmax([
            1e-5,
            np.std(power_final),
            sst.median_abs_deviation(results.power)
        ])

    snr_arr = power_final / (mad / 0.67)
    snr = float(snr_arr[idx])
    
    sde  = (power_final[idx] - np.mean(power_final)) / np.std(power_final)
    # threshold gate

    if cfg.verbose:
        print(f"Candidate: P={period:.4f} d, SDE={sde:.2f} (min {cfg.min_sde}), SNR={snr:.2f} (min {cfg.min_snr})")

    if (snr is None) or (snr < cfg.min_snr) or (sde < cfg.min_sde):
        return ("fail_threshold", None, period, t0, duration, None)

    # supported transit times (your B choice)
    supported_times = []
    n_transits_obs = 0
    try:
        stats = model.compute_stats(period, duration, t0)
        counts = stats["per_transit_count"]
        times_all = stats["transit_times"]
        supported_times = list(times_all[counts > 0])
        n_transits_obs = int(np.sum(counts > 0))
    except ValueError:
        n_transits_obs = 0

    # single-like gate: mask and continue, but don't save
    if n_transits_obs <= 1:
        if cfg.verbose:
            print("Candidate looks like a single transit (n_transits_obs =", n_transits_obs, ") - masking and continuing.")
        return ("mask_only", None, period, t0, duration, None)

    # repeat ephemeris gate: widen mask and continue, but don't save
    rep_idx = is_repeat_ephemeris(period, t0, duration, accepted_events, cfg)
    if rep_idx is not None:
        if cfg.verbose:
            print(f"Candidate matches previously accepted candidate #{rep_idx} - masking and continuing.")
        return ("repeat", None, period, t0, duration, rep_idx)


    # accept => create event now (only accepted events returned)
    ev = PeriodicEvent(
        period_days=period,
        t0_days=t0,
        duration_days=duration,
        depth=depth,
        snr=snr,
        sde=sde,
        n_transits_obs=n_transits_obs,
    )
    ev.transit_times_days = [float(x) for x in supported_times]

    if cfg.plots:

        plt.figure(figsize = (10, 6))
        val_triangles = min(snr_arr)-np.std(snr_arr)
        ax = plt.gca()
        ax.scatter(period, val_triangles, color = 'r', marker = '^', s=20, zorder = 10)

        plt.xlim(np.min(results.period), np.max(results.period))
        for n in range(2, 10):
            ax.scatter( n*period,val_triangles, color = 'maroon', marker = '^', s=20, zorder = 10, alpha= 0.8)
            ax.scatter(period / n,val_triangles, color = 'maroon', marker = '^', s=20, zorder = 10, alpha= 0.8)
        plt.ylabel(r'SNR')#, fontsize = 40)
        plt.xlabel('Period (days)')#, fontsize = 40)

        ax.plot(results.period, snr_arr, color = 'k', lw=0.65)

        plt.show()
        plt.close()
        

#         if duration<period:
#             plt.figure(figsize = (5, 5))
#             ax2 = plt.gca()

#             x = ((time_new - t0 + 0.5*period) % period) -( 0.5*period)
#             m = np.abs(x) < 0.5
#             ax2.scatter(
#                 x[m],
#                 flux_new[m],
#                 color='k',
#                 s=5,
#                 alpha=0.8,
#                 zorder=10)

#             x_new = np.linspace(-0.5, 0.5, 1000)

#             f = model.model(x_new + t0, period, duration, t0)

#             f2 = build_box_model(x_new+t0, t0, duration, depth, period)
#             ax2.plot(x_new, f, color='grey', lw = 1, alpha = 0.6, zorder = 5)
# #             ax2.plot(x_new, f2, color='violet', lw = 1, alpha = 0.6, zorder = 5)

# #             ax2.set_xlim(-0.5, 0.5)
#             ax2.set_xlabel('Phase')#, color = 'k', fontsize = 40)
#             ax2.set_ylabel('Relative Flux')#, color = 'k', fontsize = 40);
#             plt.show()

    return ("accept", ev, period, t0, duration, None)


def process_bls_results(model, results, cfg, accepted_events):
    idx, power_final = best_peak_index_from_power(results.power, cfg)
    return evaluate_best_peak(model, results, idx, power_final, cfg, accepted_events)


def run_bls(model, durations, cfg, time_span_days, period_grid=None, max_per=None):
    if period_grid is None:
        # global
        # keep your deliberate frequency_factor logic
        freq_fact_prelim = cfg.df_unc/min(durations) * (time_span_days**2)  # or compute from time_new each loop
        freq_fact_exp = np.ceil(np.log10(freq_fact_prelim))
        freq_factor = max(10, (10**(freq_fact_exp - 1))/2)
        return model.autopower(durations, frequency_factor=freq_factor, maximum_period=max_per)
    else:
        # seed window
        return model.power(period_grid, durations)

def using_BLS_search(time, flux, flux_err=None, intransit=None, period_grid = None,
                     cfg=PeriodicSearchConfig(),
                     accepted_events=None):
    """
    Iterative (while-loop) version of your recursive BLS search.
    Returns:
      - accepted_events: list[PeriodicEvent] (accepted only)
      - intransit: full-length mask (same length as `time`)
    """

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if intransit is None:
        intransit = np.zeros_like(time, dtype=bool)
    else:
        intransit = np.asarray(intransit, dtype=bool)

    if flux_err is None:
        flux_err = np.full_like(flux, np.nanstd(flux))
    else:
        flux_err = np.asarray(flux_err, dtype=float)

    if accepted_events is None:
        accepted_events = []

    # repeat widening bookkeeping (keyed by index in accepted_events)
    repeat_counts = {}

    durations = np.linspace(0.01, 0.5, 50)  # keep your preferred grid

    it = 0
    while it < cfg.max_iters and len(accepted_events) < cfg.max_planets:
        it += 1

        # residual data (mask stays full-length)
        time_new = time[~intransit]
        flux_new = flux[~intransit]
        flux_err_new = flux_err[~intransit]

        if len(time_new) < 50:
            break

        model = BoxLeastSquares(time_new, flux_new)

        # your standard global scan
        max_per = min(50.0, np.ptp(time_new) * 4/5)

        # Prepare data for BLS
        results =  run_bls(model, durations, cfg, np.ptp(time_new), period_grid=period_grid, max_per=max_per)

        # evaluate best peak (single/repeat/accept/stop)
        msg, ev, period, t0, duration, rep_idx = process_bls_results(model, results, cfg, accepted_events)

        # ---- STRICT STOP ----
        if msg == "fail_threshold":
            # ---- OPTIONAL EXTRA PASSES (commented for later) ----
            # If you ever want to try more peaks instead of strict stop:
            # - in the 'fail_threshold' branch, mask this peak and continue
            # - keep a counter to stop after N no-progress attempts.

            if it<cfg.max_iters and cfg.use_iteration_limit_for_threshold_failures:
                if cfg.verbose:
                    print(f"Candidate failed threshold but max_iters not reached (it={it}, max_iters={cfg.max_iters}) - masking and continuing.")
                intransit |= transit_mask(time, period, duration, t0, buffer = cfg.transit_mask_base_buffer_days)
                continue
            else:   
                break

        # ---- SINGLE-LIKE (mask and continue, do not save) ----
        if msg == "mask_only":
            intransit |= transit_mask(time, period, duration, t0, cfg.transit_mask_base_buffer_days)
            continue

        # ---- REPEAT EPHEMERIS (widen mask and continue, do not save) ----
        if msg == "repeat":
            # widen based on how many times we've hit this repeat
            repeat_counts[rep_idx] = repeat_counts.get(rep_idx, 0) + 1
            widen = min(cfg.max_widen_factor, 1 + repeat_counts[rep_idx])

            old = accepted_events[rep_idx]
            intransit |= transit_mask(
                time,
                old.period_days,
                old.duration_days,
                old.t0_days,
                cfg.transit_mask_base_buffer_days * widen
            )
            continue

        # ---- ACCEPT (save event, mask, continue) ----
        if msg == "accept":
            # accepted-only rule
            accepted_events.append(ev)

            intransit |= transit_mask(time, period, duration, t0, cfg.transit_mask_base_buffer_days)
            continue


    return accepted_events, intransit



def run_seed_prepass_full_lc(time, flux, flux_err, cfg, seed_periods, accepted_events, intransit):
    if not (cfg.use_seed_periods and seed_periods):
        return accepted_events, intransit

    # one iteration per seed (keeps it short + non-biasing)

    seed_cfg = PeriodicSearchConfig(**cfg.__dict__)
    seed_cfg.max_iters = 1
    seed_cfg.use_iteration_limit_for_threshold_failures = False  # keep strict stop inside the one step

    for P0 in seed_periods:
        grid = seed_period_grid(float(P0), cfg.seed_window_frac, cfg.seed_grid_size)
        accepted_events, intransit = using_BLS_search(
            time, flux, flux_err=flux_err,
            intransit=intransit,
            period_grid=grid,
            cfg=seed_cfg,
            accepted_events=accepted_events
        )
        if len(accepted_events) >= cfg.max_planets:
            break

    return accepted_events, intransit

    
def run_periodic_full_and_chunked(time, flux, flux_err, cfg, accepted_events=None, intransit=None):

    if accepted_events is None:
        accepted_events = []
    if intransit is None:
        intransit = np.zeros_like(time, dtype=bool)
    # 1) Full search
    accepted_events, intransit = using_BLS_search(time, flux, flux_err=flux_err, cfg=cfg, accepted_events=accepted_events, intransit=intransit)

    # 2) Chunked search
    blocks = breaking_up_data(time)  # shared splitter 
    # Sort blocks by size (your legacy approach does this) 
    blocks = sorted(blocks, key=lambda idx: len(idx), reverse=True)

    for idx in blocks[:4]:  # keep your legacy cap on number of chunks
        if len(idx) < 50:
            continue
        t_seg = time[idx]
        if np.ptp(t_seg)<20:
            continue
        f_seg = flux[idx]
        fe_seg = flux_err[idx] if flux_err is not None else None

        # Run recursive BLS on this segment
        
        accepted_events, _ = using_BLS_search(t_seg, f_seg, flux_err=fe_seg, cfg=cfg, accepted_events=accepted_events)

        # Merge in new ones (dedupe later)

    return accepted_events


def periodic_search(target, *, cfg=PeriodicSearchConfig(), seed_periods=None,
                    run_id=None, run_path=None):
    """
    Periodic search stage.
    Today: loads data + calls your recursive BLS (next step) to build PeriodicEvents,
    then writes periodic_events_raw into the same run_<run_id>.json.
    """
    total_csv = find_total_csv(target.root_dir, cfg.flavour)
    df = pd.read_csv(total_csv).dropna(subset=["FLUX"])

    time = df["TIME"].to_numpy(float)
    flux = df["FLUX"].to_numpy(float)

    if "FLUX_ERR" in df.columns:
        flux_err = df["FLUX_ERR"].to_numpy(float)
    else:
        flux_err = np.full_like(flux, np.nanstd(flux))

    # choose run_id/run_path (finalize should pass these)
    if run_id is None:
        run_id = target.new_run_id()

    run_path = target.candidates_run_path(run_id)    

    # ---- NEXT STEP will replace this stub with your recursive BLS ----

    accepted_events = []
    intransit = np.zeros_like(time, dtype=bool)

    accepted_events, intransit = run_seed_prepass_full_lc(time, flux, flux_err, cfg, seed_periods, accepted_events, intransit)
    periodic_events = run_periodic_full_and_chunked(time, flux, flux_err, cfg, accepted_events=accepted_events, intransit=intransit)
    # Write periodic events into the run JSON (Option 2)
    payload_update = {
        "run_id": run_id,
        "ticid": int(target.ticid),
        "gaia_id": target.gaia_id,
        "total_csv": str(total_csv),
        "periodic_events_raw": [pe.to_dict() for pe in periodic_events],
    }
    upsert_run_json(run_path, payload_update)

    return periodic_events
