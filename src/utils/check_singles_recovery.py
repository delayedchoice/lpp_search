import numpy as np
from typing import List
from core.planet_candidate import PlanetCandidate

def check_singles_against_periodic_candidate(
    periodic: PlanetCandidate,
    singles: List[PlanetCandidate],
    tol: float = None
):
    """
    Check whether single candidates correspond to a periodic candidate.

    Parameters
    ----------
    periodic : PlanetCandidate
        Must be ptype="Periodic"
    singles : list of PlanetCandidate
        Must be ptype="Single"
    tol : float or None
        Time tolerance in days. If None, uses 0.5 * duration.

    Returns
    -------
    dict with:
        matched_indices
        unmatched_indices
        unmatched_candidates
        epoch_indices
        n_matched
        n_unmatched
    """

    if periodic.ptype != "Periodic":
        raise ValueError("Input periodic candidate must be ptype='Periodic'")

    P = float(periodic.period_days)
    T0 = float(periodic.t0_days)
    duration = float(periodic.duration_days)

    if duration is None:
        raise ValueError("Periodic candidate must have duration_days")

    if tol is None:
        tol =  duration*2

    singles_t0 = np.array([c.t0_days for c in singles], dtype=float)

    matched = np.zeros(len(singles), dtype=bool)
    epoch_indices = np.zeros(len(singles), dtype=int)

    # --- match singles to nearest periodic epoch ---
    for i, t in enumerate(singles_t0):
        k = int(np.round((t - T0) / P))
        t_expected = T0 + k * P

        if abs(t - t_expected) <= tol:
            matched[i] = True
            epoch_indices[i] = k

    matched_indices = np.where(matched)[0]
    unmatched_indices = np.where(~matched)[0]

    unmatched_candidates = [singles[i] for i in unmatched_indices]

    return {
        "matched_indices": matched_indices,
        "unmatched_indices": unmatched_indices,
        "unmatched_candidates": unmatched_candidates,
        "epoch_indices": epoch_indices,
        "n_matched": len(matched_indices),
        "n_unmatched": len(unmatched_indices),
    }
