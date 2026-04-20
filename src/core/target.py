# core/target.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import numpy as np
import pandas as pd

from datetime import datetime
from core.planet_candidate import PlanetCandidate

class PipelineStage(Enum):
    RAW = auto()
    EXTRACTED = auto()
    MERGED = auto()     # singles gate
    SEARCHED = auto()   # quick singles done
    FITTED = auto()   # MCMC fit run
    REPORTED = auto()   # DV report PDF made; candidate summary ready; ready for vetting and follow-up

class DataSource(Enum):
    TGLC = "TGLC"
    SPOC = "SPOC"
    ELEANOR = "ELEANOR"
    QLP = "QLP"
    CUSTOM = "CUSTOM"

@dataclass
class Target:
    ticid: int
    gaia_id: Optional[str]
    root_dir: Path
    catalog_row: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))

    rho_star: Optional[float] = field(default=None, init=False)
    _catalog: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    pipeline_stage: PipelineStage = field(default=PipelineStage.RAW)
    data_source: DataSource = field(default=DataSource.TGLC)
    source_fits: List[Path] = field(default_factory=list)

    # quick singles state
    dt_prelim_found: Optional[bool] = None
    quick_singles_t0: List[float] = field(default_factory=list)

    # candidate run bookkeeping (pointers only; keep state small)
    last_run_id: Optional[str] = None
    last_candidates_run: Optional[str] = None

    def __post_init__(self):
        self.root_dir = Path(self.root_dir)
        if isinstance(self.catalog_row, pd.Series) and len(self.catalog_row) > 0:
            self._catalog = {str(k): (None if pd.isna(v) else v) for k, v in self.catalog_row.items()}
            for col, val in self._catalog.items():
                setattr(self, col, val)
            self._compute_rho_star_if_possible()

    def _compute_rho_star_if_possible(self):
        try:
            mass = float(getattr(self, "Mass"))
            rad  = float(getattr(self, "Rad"))
        except (AttributeError, TypeError, ValueError):
            return
        if np.isfinite(mass) and np.isfinite(rad) and rad > 0:
            # (3/4π) * M/R^3 ; you can switch to your preferred normalization later
            self.rho_star = (mass / (rad**3)) * (3.0 / (4.0 * np.pi))

    def catalog(self) -> Dict[str, Any]:
        return dict(self._catalog)

    # -------- state I/O --------
    @property
    def state_path(self) -> Path:
        return self.root_dir / "target.state.json"

    def save_state(self) -> None:
        payload = {
            "ticid": self.ticid,
            "gaia_id": self.gaia_id,
            "data_source": self.data_source.value,
            "pipeline_stage": self.pipeline_stage.name,
            "source_fits": [str(p) for p in self.source_fits],
            "rho_star": self.rho_star,
            "catalog": self._catalog,
            "dt_prelim_found": self.dt_prelim_found,
            "quick_singles_t0": list(self.quick_singles_t0),
            "last_run_id": self.last_run_id,
            "last_candidates_run": self.last_candidates_run,
        }
        self.state_path.write_text(json.dumps(payload, indent=2))

    def load_state(self) -> None:
        if not self.state_path.exists():
            return
        pl = json.loads(self.state_path.read_text())

        # enums
        try:
            self.data_source = DataSource(pl.get("data_source", "TGLC"))
        except Exception:
            self.data_source = DataSource.TGLC
        try:
            self.pipeline_stage = PipelineStage[pl.get("pipeline_stage", "RAW")]
        except Exception:
            self.pipeline_stage = PipelineStage.RAW

        self.source_fits = [Path(s) for s in pl.get("source_fits", [])]
        self.rho_star = pl.get("rho_star", self.rho_star)
        self.gaia_id = pl.get("gaia_id", self.gaia_id)

        self.last_run_id = pl.get("last_run_id", self.last_run_id)
        self.last_candidates_run = pl.get("last_candidates_run", self.last_candidates_run)

        if isinstance(pl.get("catalog"), dict):
            self._catalog = {str(k): v for k, v in pl["catalog"].items()}
            for col, val in self._catalog.items():
                setattr(self, col, val)
            self._compute_rho_star_if_possible()

        self.dt_prelim_found = pl.get("dt_prelim_found", self.dt_prelim_found)
        t0_list = pl.get("quick_singles_t0", [])
        if isinstance(t0_list, list):
            try:
                self.quick_singles_t0 = [float(x) for x in t0_list]
            except Exception:
                self.quick_singles_t0 = []

    @property
    def ld_u1_u2(self):
        """
        Quadratic limb-darkening coefficients (u1,u2) for this star.
        Expected to be cached in the catalog row as aLSM/bLSM by DataPrep.ensure_catalog().
        """
        u1 = getattr(self, "aLSM", None)
        u2 = getattr(self, "bLSM", None)
        if u1 is None or u2 is None:
            raise ValueError("Limb darkening not available: missing aLSM/bLSM on Target. Run DataPrep.ensure_catalog().")
        return float(u1), float(u2)
        
    # -------- convenience --------
    def set_stage(self, stage: PipelineStage) -> None:
        self.pipeline_stage = stage
        self.save_state()

    @staticmethod
    def discover_ids_from_dirname(dir_path: Path) -> tuple[int, Optional[str]]:
        name = Path(dir_path).name
        ticid = int(name.split("_")[1].split("-")[1])
        gaia_id = str(name.split("_")[2].split("-")[-1]) if "gaiaID-" in name else None
        return ticid, gaia_id

    @classmethod
    def from_dir(cls, dir_path: Path) -> "Target":
        ticid, gaia_id = cls.discover_ids_from_dirname(dir_path)
        t = cls(ticid=ticid, gaia_id=gaia_id, root_dir=Path(dir_path))
        t.load_state()
        return t

    # -------- candidates (per-run artifacts) --------
    @property
    def candidates_dir(self) -> Path:
        d = self.root_dir / "candidates"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def new_run_id(self) -> str:
        # safe filename-friendly timestamp
        return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    def candidates_run_path(self, run_id: str) -> Path:
        return self.candidates_dir / f"run_{run_id}.json"

    def save_candidates(self, run_id: str, candidates: List[PlanetCandidate]) -> Path:
        path = self.candidates_run_path(run_id)
        payload = {
            "run_id": run_id,
            "ticid": self.ticid,
            "gaia_id": self.gaia_id,
            "candidates": [c.to_dict() for c in candidates],
        }
        path.write_text(json.dumps(payload, indent=2))

        # update pointers in state
        self.last_run_id = run_id
        self.last_candidates_run = str(path.relative_to(self.root_dir))
        self.save_state()
        return path

    def load_candidates(self, path: Optional[Path] = None) -> List[PlanetCandidate]:
        if path is None:
            if not self.last_candidates_run:
                return []
            path = self.root_dir / self.last_candidates_run
        if not Path(path).exists():
            return []
        pl = json.loads(Path(path).read_text())
        cand_list = pl.get("candidates", [])
        out: List[PlanetCandidate] = []
        if isinstance(cand_list, list):
            for d in cand_list:
                if isinstance(d, dict):
                    out.append(PlanetCandidate.from_dict(d))
        return out

        # inside Target class (core/target.py)

    def stage_rank(self) -> int:
        order = {
            PipelineStage.RAW: 0,
            PipelineStage.EXTRACTED: 1,
            PipelineStage.MERGED: 2,
            PipelineStage.SEARCHED: 3,
            PipelineStage.FITTED: 4,
            PipelineStage.REPORTED: 5,
        }
        return order.get(self.pipeline_stage, -1)

    def stage_at_least(self, stage: PipelineStage) -> bool:
        order = {
            PipelineStage.RAW: 0,
            PipelineStage.EXTRACTED: 1,
            PipelineStage.MERGED: 2,
            PipelineStage.SEARCHED: 3,
            PipelineStage.FITTED: 4,
            PipelineStage.REPORTED: 5,
        }
        return order.get(self.pipeline_stage, -1) >= order.get(stage, 10)