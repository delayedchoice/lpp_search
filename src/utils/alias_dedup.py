# utils/alias_dedup.py
from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple, Optional

from core.planet_candidate import PlanetCandidate


# ---------- small helpers ----------
def _get_summary(c: PlanetCandidate, var: str) -> Optional[dict]:
    d = getattr(c, "pymc_summary", None)
    if isinstance(d, dict):
        vv = d.get(var, None)
        return vv if isinstance(vv, dict) else None
    return None

def _median(c: PlanetCandidate, var: str, fallback=None):
    vv = _get_summary(c, var)
    if vv and "median" in vv:
        try:
            return float(vv["median"])
        except Exception:
            pass
    return fallback

def _hdi16(c: PlanetCandidate, var: str, fallback=None):
    vv = _get_summary(c, var)
    if vv and "hdi_16%" in vv:
        try:
            return float(vv["hdi_16%"])
        except Exception:
            pass
    return fallback

def _max_rhat(c: PlanetCandidate) -> Optional[float]:
    d = getattr(c, "pymc_summary", None)
    if not isinstance(d, dict):
        return None
    vals = []
    for vv in d.values():
        if isinstance(vv, dict) and "r_hat" in vv:
            try:
                vals.append(float(vv["r_hat"]))
            except Exception:
                pass
    return max(vals) if vals else None


def _depth_med(c: PlanetCandidate) -> Optional[float]:
    d = _median(c, "depth", getattr(c, "depth", None))
    return None if d is None else float(d)


def _signal_not_zero(c: PlanetCandidate, eps: float = 1e-6) -> bool:
    """
    Your idea: reject periods that 'fit flat light curves'.
    Implemented as: rp_rs (or depth) HDI lower bound must be > 0-ish.
    """
    rp_lo = _hdi16(c, "rp_rs", None)
    if rp_lo is not None:
        return rp_lo > eps
    d_lo = _hdi16(c, "depth", None)
    if d_lo is not None:
        return d_lo > eps
    # fallback: if no HDI, use median depth
    d_med = _depth_med(c)
    return (d_med is not None) and (float(d_med) > eps)

def _snr_med(c: PlanetCandidate) -> float:
    s = _median(c, "SNR", getattr(c, "snr", 0.0))
    try:
        return float(s)
    except Exception:
        return 0.0

def _period(c: PlanetCandidate) -> Optional[float]:
    p = getattr(c, "period_days", None)
    return None if p is None else float(p)

def _t0(c: PlanetCandidate) -> float:
    return float(getattr(c, "t0_days"))



def _phase_offset_days(t: float, t0: float, P: float) -> float:
    """
    Distance (days) between t and nearest epoch implied by (t0,P)
    """
    x = (t - t0 + 0.5 * P) % P - 0.5 * P
    return abs(x)


# ---------- clustering criteria ----------
def _get_observed_transit_times(
    c
) -> Optional[np.ndarray]:
    """
    Returns observed transit times (days) for clustering/dedup:
      - Single: [t0_days]
      - Periodic: transit_times_days if present
    Returns None if not available for periodic.
    """

    # Singles: always just the one observed epoch
    if getattr(c, "ptype", None) == "Single":
        return np.array([float(getattr(c, "t0_days"))], dtype=float)


    # Periodic: otherwise look for a candidate field (if/when you add it)
    arr = getattr(c, "transit_times_days", None)
    if arr:
        vv = np.asarray(arr, dtype=float)
        vv = vv[np.isfinite(vv)]
        return np.sort(vv) if vv.size else None

    return None



def _shared_t0_overlap(a: PlanetCandidate, b: PlanetCandidate, tol_days: float) -> Tuple[int, int]:
    """
    Count overlaps between observed transit-time lists.
    Returns (n_overlap, n_minlist). If either list missing -> (0,0).
    """
    ta = _get_observed_transit_times(a)
    tb = _get_observed_transit_times(b)
    if ta is None or tb is None:
        return 0, 0

    i = j = 0
    overlap = 0
    while i < ta.size and j < tb.size:
        da = ta[i]
        db = tb[j]
        if abs(da - db) <= tol_days:
            overlap += 1
            i += 1
            j += 1
        elif da < db:
            i += 1
        else:
            j += 1

    return overlap, int(min(ta.size, tb.size))

def _depth_consistent(a: PlanetCandidate, b: PlanetCandidate, ratio_max: float = 1.75, floor: float = 5e-5) -> bool:
    da = _depth_med(a)
    db = _depth_med(b)
    if da is None or db is None or (not np.isfinite(da)) or (not np.isfinite(db)):
        return True  # don't block if depth missing
    dmax = max(da, db)
    dmin = max(min(da, db), floor)
    return (dmax / dmin) <= ratio_max


# ---------- winner selection ----------
def _winner_key(c: PlanetCandidate) -> Tuple:
    """
    Higher is better. This encodes what we discussed:
      1) signal_not_zero (your 'flat LC fit' veto)
      2) fit_is_current True
      3) lower rhat (better)  -> negative
      4) higher SNR median
      5) shorter period (tie-breaker; negative period so smaller wins)
    """
    sig = 1 if _signal_not_zero(c) else 0
    fit = 1 if getattr(c, "fit_is_current", False) else 0

    rhat = _max_rhat(c)
    # rhat_ok: 1 if acceptable / 0 if not (tune threshold)
    rhat_ok = 1
    if rhat is not None and float(rhat) > 1.05:
        rhat_ok = 0
    snr = _snr_med(c)
    P = _period(c)
    per_score = 0.0 if P is None else -float(P)  # shorter is better as tie-breaker
    return (sig, fit, rhat_ok, snr, per_score)


# ---------- main entrypoint ----------
def alias_dedup_periodic_candidates(
    periodic_candidates: List[PlanetCandidate],
    *,
    # clustering thresholds
    shared_t0_tol_days: float = 0.05,
    shared_overlap_frac: float = 0.6,
    depth_ratio_max: float = 1.75,
    # ephemeris fallback thresholds
    epoch_tol_scale: float = 0.25,
    epoch_tol_floor_days: float = 0.02,
) -> List[PlanetCandidate]:
    """
    Mutates candidates in-place:
      - sets default=False on duplicates
      - appends a note pointing to the winner candidate_id
    Returns the same list (mutated).
    """

    # only periodic with a period
    cands = [c for c in periodic_candidates if getattr(c, "ptype", None) == "Periodic" and _period(c) is not None]
    n = len(cands)
    if n < 2:
        return periodic_candidates

    # union-find clustering
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # pairwise cluster decision
    for i in range(n):
        for j in range(i + 1, n):
            a, b = cands[i], cands[j]

            # 1) Prefer observed transit-time overlap if available
            overlap, denom = _shared_t0_overlap(a, b, tol_days=shared_t0_tol_days)
            if denom > 0:
                if (overlap / denom) >= shared_overlap_frac and _depth_consistent(a, b, ratio_max=depth_ratio_max):
                    union(i, j)
                continue  # if we had lists, we don't need alias fallback

            # 2) Fallback: harmonic period relation + ephemeris-phase check + depth consistency
            if not _depth_consistent(a, b, ratio_max=depth_ratio_max):
                print('not consistent depth')
                continue

            Pa, Pb = _period(a), _period(b)
            if Pa is None or Pb is None:
                continue

            # stable ordering (also keep the matching t0s aligned with short/long)
            if Pa <= Pb:
                Pshort, Plong = Pa, Pb
                t0short, t0long = _t0(a), _t0(b)
            else:
                Pshort, Plong = Pb, Pa
                t0short, t0long = _t0(b), _t0(a)

            ratio = Plong / Pshort

            # look for small rational ratios up to 5 (tune if needed)
            is_alias = False
            for zz in range(1, 6):
                for yy in range(1, 6):
                    r = zz / yy
                    if abs(ratio - r) <= 0.01:   # 1% tolerance
                        is_alias = True
                        break
                if is_alias:
                    break

            if not is_alias:
                continue

            # define tol (duration-based if available, otherwise floor)
            dura = _median(a, "dur", getattr(a, "duration_days", None))
            durb = _median(b, "dur", getattr(b, "duration_days", None))
            dur_ref = None
            for d in (dura, durb):
                if d is not None and np.isfinite(d):
                    dur_ref = float(d) if dur_ref is None else min(dur_ref, float(d))

            tol = max(
                epoch_tol_floor_days,
                epoch_tol_scale * (dur_ref if dur_ref is not None else epoch_tol_floor_days),
            )

            # phase alignment for aliases: long epoch should land on the short ephemeris
            off = _phase_offset_days(t0long, t0short, Pshort)
            if off <= tol:
                union(i, j)    # build groups
    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    # apply winner/loser marking
    for idx_list in groups.values():
        if len(idx_list) < 2:
            continue
        members = [cands[i] for i in idx_list]
        winner = max(members, key=_winner_key)
        winner_id = winner.candidate_id()

        for m in members:
            if m is winner:
                continue
            m.default = False
            note = f"alias_dedup: duplicate/alias of {winner_id}"
            m.notes = (m.notes + "; " + note) if getattr(m, "notes", "") else note

    return periodic_candidates