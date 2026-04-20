# utils/singles_periodicity.py
import numpy as np

# --- paste your existing functions here ---
def prepping_singles_for_periodic_check(
    t0_singles,
    durations=None, depths=None,
    max_missed_transits=7, P_min=0.25, P_max=None,
    phase_win=None, rel_merge=0.01, min_support=3
):
    t0_all = np.asarray(t0_singles, dtype=float)
    t0, idx = np.unique(t0_all, return_index=True)
    if t0.size < 2:
        return t0, None, None, [], (phase_win if phase_win is not None else 0.05)

    dur = (np.asarray(durations, dtype=float)[idx]
           if (durations is not None and len(durations) == len(t0_singles)) else None)
    dep = (np.asarray(depths, dtype=float)[idx]
           if (depths is not None and len(depths) == len(t0_singles)) else None)

    baseline = float(t0[-1] - t0[0])
    P_max_eff = float(P_max) if (P_max is not None) else max(1.0, baseline if (min_support <= 2) else (baseline / 2.0))

    if phase_win is None:
        if (dur is not None) and np.isfinite(dur).any():
            phase_win_eff = max(0.02, 0.4 * float(np.nanmedian(dur)))
        else:
            phase_win_eff = 0.04
    else:
        phase_win_eff = float(phase_win)

    diffs = np.array([t0[j] - t0[i] for i in range(t0.size) for j in range(i + 1, t0.size)], dtype=float)
    diffs = diffs[diffs > 0]

    cand = []
    for delta in diffs:
        for m in range(1, max_missed_transits + 1):
            P = delta / m
            if (P_min <= P <= P_max_eff):
                cand.append(P)
    if not cand:
        return t0, dur, dep, [], phase_win_eff

    cand = np.sort(np.array(cand, dtype=float))
    groups, cur = [], [cand[0]]
    for v in cand[1:]:
        if abs(v - np.median(cur)) / max(v, 1e-6) <= rel_merge:
            cur.append(v)
        else:
            groups.append(np.array(cur, dtype=float)); cur = [v]
    groups.append(np.array(cur, dtype=float))
    return t0, dur, dep, groups, phase_win_eff

def score_once_modes(
    t0, dur, dep, groups, phase_win,
    min_support=3, use_depth=True, depth_zmax=2.5, depth_ratio_max=1.75, depth_floor=5e-5,
    local_span=0.02, local_n=41
):
    out = []
    if t0.size < min_support or len(groups) == 0:
        return out

    for g in groups:
        P0 = float(np.median(g))
        grid = P0 * np.linspace(1.0 - local_span, 1.0 + local_span, local_n)
        best, best_members = None, None

        for P in grid:
            phases = (t0 - t0[0]) % P
            phases = np.where(phases > P/2, phases - P, phases)
            center = np.median(phases)
            resid  = (phases - center) % P
            resid  = np.where(resid > P/2, resid - P, resid)

            members = np.where(np.abs(resid) <= phase_win)[0]
            support = int(members.size)
            if support < min_support:
                continue

            if use_depth and (dep is not None) and (members.size > 1):
                d_sup = dep[members]
                d_med = float(np.median(d_sup))
                scat  = 1.4826 * np.median(np.abs(d_sup - d_med))  # MAD
                if scat <= depth_floor:
                    dmax, dmin = float(np.max(d_sup)), float(np.min(d_sup))
                    if (dmax / max(dmin, depth_floor)) > depth_ratio_max:
                        continue
                else:
                    z = np.abs(d_sup - d_med) / scat
                    if np.any(z > depth_zmax):
                        continue
                depth_penalty = scat
            else:
                d_med, depth_penalty = (np.nan, 0.0)

            phase_rms = float(np.sqrt(np.mean(resid[members]**2)))
            key = (support, -phase_rms, -depth_penalty)
            if (best is None) or (key > best[0]):
                best = (key, float(P), float(center), support, phase_rms, d_med, depth_penalty)
                best_members = members

        if best is None:
            continue
        _, P_star, center, support, phase_rms, d_med, depth_pen = best
        out.append({
            'P': float(P_star),
            'T0': float(t0[0] + center),
            'support': int(support),
            'members': np.array(best_members, dtype=int),
            'phase_rms': float(phase_rms),
            'depth_med': float(d_med),
            'depth_scat': float(depth_pen) if np.isfinite(depth_pen) else np.nan
        })

    out.sort(key=lambda d: (-d['support'], d['phase_rms']))
    return out

def extract_all_modes_iterative(
    t0_init, dur_init, dep_init,
    min_support=3, prep_kwargs=None, scorer_kwargs=None
):
    prep_kwargs = prep_kwargs or {}
    scorer_kwargs = scorer_kwargs or {}

    accepted = []
    t0 = np.array(t0_init, dtype=float)
    dur = None if dur_init is None else np.array(dur_init, dtype=float)
    dep = None if dep_init is None else np.array(dep_init, dtype=float)

    while True:
        t0_w, dur_w, dep_w, groups, phase_win = prepping_singles_for_periodic_check(
            t0, dur, dep, min_support=min_support, **prep_kwargs
        )
        if (t0_w.size < min_support) or (len(groups) == 0):
            break

        candidates = score_once_modes(
            t0_w, dur_w, dep_w, groups, phase_win,
            min_support=min_support, **scorer_kwargs
        )
        if len(candidates) == 0:
            break

        best = candidates[0]
        accepted.append(best)

        keep = np.ones(t0_w.size, dtype=bool)
        keep[best['members']] = False
        t0 = t0_w[keep]
        dur = None if dur_w is None else dur_w[keep]
        dep = None if dep_w is None else dep_w[keep]

        if t0.size < min_support:
            break

    return accepted


def periodic_modes_from_dt_events(
    events,
    *,
    min_support=3,
    max_missed_transits=7,
    P_min=0.25,
    P_max=None,
    rel_merge=0.01,
    use_depth=True,
    local_span=0.02,
    local_n=41,
):
    """
    Return periodic modes inferred from DT TransitEvent objects.
    Each mode is a dict with keys like P, T0, support, members, phase_rms, ...
    """
    if events is None or len(events) < min_support:
        return []

    t0 = np.array([float(e.t0_days) for e in events], dtype=float)
    dur = np.array([float(e.duration_days) for e in events], dtype=float)
    dep = np.array([float(e.depth) for e in events], dtype=float)

    modes = extract_all_modes_iterative(
        t0, dur, dep,
        min_support=min_support,
        prep_kwargs={
            "max_missed_transits": max_missed_transits,
            "P_min": P_min,
            "P_max": P_max,
            "rel_merge": rel_merge,
            "min_support": min_support
        },
        scorer_kwargs={
            "use_depth": use_depth,
            "local_span": local_span,
            "local_n": local_n
        }
    )
    return modes


def seed_periods_from_dt_events(
    events,
    *,
    top_k=10,
    min_support=3,
    **mode_kwargs
):
    """
    Convenience wrapper: compute modes, then return top_k periods.
    """
    modes = periodic_modes_from_dt_events(events, min_support=min_support, **mode_kwargs)
    return [float(m["P"]) for m in modes[:top_k]]