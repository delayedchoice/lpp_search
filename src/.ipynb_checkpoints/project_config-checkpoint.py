# config.py
from pathlib import Path
import pandas as pd

# Project roots
PROJECT_ROOT = Path(__file__).resolve().parent

# Data paths
MDWARF_CATALOG = PROJECT_ROOT / "data" / "final_mdwarf_params.csv"
LDC_FILES = PROJECT_ROOT / "data" / "LDC_params" / "table15.dat"

# Physics constants (in your units)
G = 2941.18330364  # R_s^3 M_s^-1 day^−2

# DeepTransit model
MODEL_PATH = PROJECT_ROOT / "model_TESS.pth"

# Pre-load and filter LDC table once
LDC_for_quadratic = pd.read_csv(
    LDC_FILES, header=None, sep=r"\s+", engine="python",
    names=['logg','Teff','z','L/HP','aLSM','bLSM','aFCM','bFCM','SQRT(CHI2)','qsr','PC']
)
LDC_PARAMS_MDWARF = LDC_for_quadratic[LDC_for_quadratic['Teff'] < 4300]


DEFAULT_DATA_SOURCE = "TGLC"
