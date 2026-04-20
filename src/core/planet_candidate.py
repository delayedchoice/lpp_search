# core/planet_candidate.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal
import math


PType = Literal["Periodic", "Single"]


def _round_or_none(x: Optional[float], ndp: int) -> Optional[float]:
    #ndp = number of decimal places to round to for stable candidate IDs; this is a balance between precision and stability (too many decimals and tiny changes cause ID changes; too few and distinct candidates might merge)
    if x is None:
        return None
    try:
        xf = float(x)
    except Exception:
        return None
    if not math.isfinite(xf):
        return None
    return round(xf, ndp)


@dataclass
class PlanetCandidate:
    """
    One candidate hypothesis for a target star.
    This is the unit we save/load in candidates/run_<run_id>.json.

    Identity is by fingerprint (ptype + rounded period + rounded t0), not by P1/P2.
    """
    # --- identity / hypothesis ---
    ptype: PType
    t0_days: float
    period_days: Optional[float] = None  # None for singles (or if unknown)

    # --- shape / detection ---
    duration_days: Optional[float] = None
    depth: Optional[float] = None

    snr: Optional[float] = None
    sde: Optional[float] = None
    n_transits_obs: Optional[int] = None

    # --- provenance ---
    source: str = ""  # e.g. "BLS", "DT", "SINGLES_CLUSTER", "MANUAL"

    # --- fit bookkeeping ---
    fit_fingerprint: Optional[str] = None #this will strore t0, per and tdur of last fit; if current hypothesis matches, fit is up-to-date; otherwise needs refit
    fit_is_current: bool = False
    pymc_summary: Dict[str, Any] = field(default_factory=dict)  # store summary dicts, i.e. pparams and diagnostics from the last fit, for later consolidation and vetting

    # --- freeform notes / flags ---
    notes: str = ""
    default: bool = True

    # -----------------------------
    # Identity + fit key helpers
    # -----------------------------
    def candidate_id(self, *, p_ndp: int = 6, t0_ndp: int = 5) -> str:
        """
        Stable fingerprint for this hypothesis.
        If period changes later, the ID changes (as it should).
        """
        p = _round_or_none(self.period_days, p_ndp)
        t0 = _round_or_none(self.t0_days, t0_ndp)
        if self.ptype == "Single" or p is None:
            return f"{self.ptype}|t0={t0}"
        return f"{self.ptype}|P={p}|t0={t0}"

    def compute_fit_fingerprint(self, *, p_ndp: int = 6, t0_ndp: int = 5) -> str:
        """
        The key we store at the moment of fitting.
        If anything defining the hypothesis changes, the key will no longer match.
        """
        return self.candidate_id(p_ndp=p_ndp, t0_ndp=t0_ndp)

    def mark_fitted(self) -> None:
        self.fit_fingerprint = self.compute_fit_fingerprint()
        self.fit_is_current = True

    def mark_needs_refit(self) -> None:
        self.fit_is_current = False

    def refresh_fit_status(self) -> None:
        """
        Recompute whether the stored fit_fingerprint matches the current hypothesis.
        Call this after alias resolution / candidate edits.
        """
        if self.fit_fingerprint is None:
            self.fit_is_current = False
            return
        self.fit_is_current = (self.fit_fingerprint == self.compute_fit_fingerprint())

    # -----------------------------
    # JSON (dict) serialization
    # -----------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ptype": self.ptype,
            "t0_days": float(self.t0_days),
            "period_days": None if self.period_days is None else float(self.period_days),
            "duration_days": None if self.duration_days is None else float(self.duration_days),
            "depth": None if self.depth is None else float(self.depth),
            "snr": None if self.snr is None else float(self.snr),
            "sde": None if self.sde is None else float(self.sde),
            "n_transits_obs": None if self.n_transits_obs is None else int(self.n_transits_obs),
            "source": self.source,
            "fit_fingerprint": self.fit_fingerprint,
            "fit_is_current": bool(self.fit_is_current),
            "pymc_summary": self.pymc_summary,
            "notes": self.notes,
            "default": bool(self.default),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlanetCandidate":
        return cls(
            ptype=d.get("ptype", "Single"),
            t0_days=float(d["t0_days"]),
            period_days=(None if d.get("period_days") is None else float(d["period_days"])),
            duration_days=(None if d.get("duration_days") is None else float(d["duration_days"])),
            depth=(None if d.get("depth") is None else float(d["depth"])),
            snr=(None if d.get("snr") is None else float(d["snr"])),
            sde=(None if d.get("sde") is None else float(d["sde"])),
            n_transits_obs=(None if d.get("n_transits_obs") is None else int(d["n_transits_obs"])),
            source=str(d.get("source", "")),
            fit_fingerprint=d.get("fit_fingerprint"),
            fit_is_current=bool(d.get("fit_is_current", False)),
            pymc_summary=dict(d.get("pymc_summary", {})),
            notes=str(d.get("notes", "")),
            default=bool(d.get("default", True)),
        )

        from core.planet_candidate import PlanetCandidate

    def single_candidates_from_dt_events(events, *, source="DT"):
        out = []
        for e in events:
            out.append(PlanetCandidate(
                ptype="Single",
                t0_days=float(e.t0_days),
                period_days=None,
                duration_days=float(e.duration_days),
                depth=float(e.depth),
                snr=None if e.snr is None else float(e.snr),
                source=source,
                fit_is_current=False,
            ))
        return out