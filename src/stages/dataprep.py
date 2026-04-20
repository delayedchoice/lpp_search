# stages/dataprep.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits as apf
from astropy.stats import sigma_clip
from astropy import units
from wotan import flatten

from core.target import Target, PipelineStage, DataSource
import config as con

# ---- LDC helpers (trimmed from Functions_all.py) ----
def match_logg_and_teff_for_LDC(df: pd.DataFrame) -> pd.DataFrame:
    ldc = con.LDC_PARAMS_MDWARF.copy()
    a, b = [], []
    for i in range(len(df)):
        Teff = float(df.loc[i, 'Teff'])
        logg = float(df.loc[i, 'logg']) if pd.notna(df.loc[i, 'logg']) else np.nan
        k = min(8, len(ldc))
        idx_teff = np.abs(ldc['Teff'].astype(float) - Teff).argsort()[:k]
        pool = ldc.iloc[idx_teff].reset_index(drop=True)
        if not np.isfinite(logg):
            logg = np.nanmedian(pool['logg'].astype(float))
        j = np.abs(pool['logg'].astype(float) - logg).argsort().iloc[0]
        aLSM, bLSM = pool.loc[j, ['aLSM', 'bLSM']]
        a.append(float(aLSM)); b.append(float(bLSM))
    df['aLSM'] = a; df['bLSM'] = b
    return df  # [

def get_catalog_info(ticid: int, rtrn_df: bool=False) -> pd.DataFrame | tuple:
    mdwarfs = pd.read_csv(con.MDWARF_CATALOG, iterator=True, chunksize=100000)
    new_df = pd.concat([c[c['TICID'].astype(int) == int(ticid)] for c in mdwarfs]).reset_index(drop=True)
    new_df = match_logg_and_teff_for_LDC(new_df)
    return new_df if rtrn_df else (
        new_df[['aLSM','bLSM']].values[0].astype(float),
        float(new_df['Mass']), float(new_df['Rad'])
    )  

# ---- basic cleaning & flattening ----
def remove_outliers(time, flux, sigma_lower=6.0, sigma_upper=3.0, **kwargs):
    return sigma_clip(flux, sigma_lower=sigma_lower, sigma_upper=sigma_upper, **kwargs).mask  

def T14(P, R_star, M_star, R_planet, b=0, i=90*units.deg):
    # Short wrapper; same math as your original
    from astropy import constants as const
    P, R_star, M_star, R_planet = P*units.day, R_star*units.R_sun, M_star*units.M_sun, R_planet*units.R_earth
    a3 = (const.G * M_star * (P.to(units.s))**2 / (4*np.pi**2)).to(units.m**3)
    a = a3**(1/3)
    k = (R_planet / R_star).decompose().value
    i = i.to(units.rad).value
    val = (R_star / a).decompose().value * np.sqrt((1 + k)**2 - b**2) / np.sin(i)
    return ((P.to(units.s) / np.pi) * np.arcsin(val)).to(units.day).value  

def flatten_lc(time, flux, catalog_df=pd.DataFrame({'Rad':[-1],'Mass':[-1]}), maxP=100, R_planet_RE=2):
    if len(catalog_df) > 0:
        M_star = float(catalog_df.iloc[0]['Mass'])
        R_star = float(catalog_df.iloc[0]['Rad'])
    else:
        M_star, R_star = 0.5, 0.5
    if not np.isfinite(M_star) or M_star <= 0: M_star = 0.5
    if not np.isfinite(R_star) or R_star <= 0: R_star = 0.5
    windw = 3 * T14(P=maxP, R_star=R_star, M_star=M_star, R_planet=2*R_planet_RE)
    flat_flux, trend = flatten(time, flux, method='biweight', window_length=windw, return_trend=True)
    return flat_flux, trend  

def extract_data_from_fits_files(fitsFile, PL="", sector=0):
    with apf.open(fitsFile) as hdulist:
        tb = next(h for h in hdulist if 'Table' in str(h))
        data = tb.data
        cols = [c.name.upper() for c in tb.columns]
    df = pd.DataFrame({'TIME': data['TIME']})
    if PL: PL = PL.upper() + '_'
    flux_cols = sorted([c for c in cols if 'FLUX' in c and (not c.startswith(('CAL','K')) or 'X_' in c)], key=len)
    bkg_cols  = sorted([c for c in cols if 'BKG' in c or 'BACKGROUND' in c], key=len)
    cent_cols = [c for c in cols if 'CENTR' in c and 'MOM' not in c]
    qual_cols = [c for c in cols if 'QUAL' in c or 'FLAG' in c]
    if not flux_cols: return
    useful = []
    if len(flux_cols) > 1:
        for c in flux_cols:
            out = c.split('_')[0][:4] + '_FLUX'
            df[out] = data[c]; useful.append((c, out))
    else:
        df['FLUX'] = data[flux_cols[0]]; useful.append((flux_cols[0], 'FLUX'))
    if bkg_cols:
        df['BKG_FLUX'] = data[bkg_cols[0]]; useful.append((bkg_cols[0], 'BKG_FLUX'))
    if cent_cols:
        xs = sorted([c for c in cent_cols if 'X' in c or '1' in c], key=len)
        ys = sorted([c for c in cent_cols if 'Y' in c or '2' in c], key=len)
        if xs: df['CENTROID_X'] = data[xs[0]]; useful.append((xs[0], 'CENTROID_X'))
        if ys: df['CENTROID_Y'] = data[ys[0]]; useful.append((ys[0], 'CENTROID_Y'))
    if qual_cols:
        df['QUALITY'] = np.sum([data[q] for q in qual_cols], axis=0)
    for orig, out in useful:
        err = orig + '_ERR'
        if err in cols:
            df[out + '_ERR'] = data[err]
    outdir = os.path.dirname(fitsFile); os.makedirs(outdir, exist_ok=True)
    outname = f"{outdir}/{PL}{os.path.basename(outdir)}_sector{int(sector):02d}.csv"
    df.to_csv(outname, index=False)
    return df  

def get_data(ticid_directory, flux_type="APER_", PL="TGLC", verbose=False, catalog_df=False):
    files = sorted(glob.glob(f"{ticid_directory}/*_sector*.csv"))
    all_t, all_f, all_fe, all_flat, all_flat_fe, all_trend = [], [], [], [], [], []
    for fil in files:
        df = pd.read_csv(fil)
        flux_col = flux_type + "FLUX" if flux_type + "FLUX" in df.columns else "FLUX"
        if 'QUALITY' in df: df = df[df['QUALITY'] == 0]
        df = df[~np.isnan(df[flux_col])]
        if df.empty: continue
        med = np.nanmedian(df[flux_col])
        ferr = (df[flux_col + '_ERR'] / med) if (flux_col + '_ERR') in df else np.full(len(df), np.std(df[flux_col] / med))
        df[flux_col] /= med
        time, flux = df.TIME.to_numpy(), df[flux_col].to_numpy()
        mask = ~remove_outliers(time, flux)
        t, f, fe = time[mask], flux[mask], ferr[mask]
        flat, trend = (flatten_lc(t, f) if isinstance(catalog_df, bool) else flatten_lc(t, f, catalog_df=catalog_df))
        flat_err = np.full(len(flat), np.std(flat))
        df2 = df.iloc[mask].copy()
        df2[flux_col + '_FLAT'] = flat
        df2[flux_col + '_FLAT_ERR'] = flat_err
        df2[flux_col + '_TREND'] = trend
        df2.to_csv(fil[:-4] + '_flat.csv', index=False)
        if verbose:
            plt.figure(figsize=(20,5)); plt.scatter(time, flux, s=3); plt.plot(t, trend, 'k')
            plt.figure(figsize=(20,5)); plt.scatter(t, flat, s=3); plt.show()
        all_t.extend(t); all_f.extend(f); all_fe.extend(fe)
        all_flat.extend(flat); all_flat_fe.extend(flat_err); all_trend.extend(trend)
    if not all_t:
        return
    idx = np.argsort(all_t)
    out = pd.DataFrame({
        'TIME': np.array(all_t)[idx],
        'RAW_FLUX': np.array(all_f)[idx],
        'RAW_FLUX_ERR': np.array(all_fe)[idx],
        'FLUX': np.array(all_flat)[idx],
        'FLUX_ERR': np.array(all_flat_fe)[idx],
        'FLUX_TREND': np.array(all_trend)[idx]
    })
    outname = f"{ticid_directory}/{os.path.basename(ticid_directory)}_{PL}_{flux_type}total.csv"
    out.to_csv(outname, index=False)
    return out  

# ---- DataPrep class ----
@dataclass
class DataPrep:
    target: Target
    flavour: str = "TGLC"
    source: DataSource = DataSource.TGLC

    def ensure_catalog(self) -> None:
        cat_csv = self.target.root_dir / "tic_star_parameters.csv"
        if cat_csv.exists():
            row = pd.read_csv(cat_csv).iloc[0]
        else:
            catalog_df = get_catalog_info(self.target.ticid, rtrn_df=True)
            cat_csv.parent.mkdir(parents=True, exist_ok=True)
            catalog_df.to_csv(cat_csv, index=False)
            row = catalog_df.iloc[0]
        self.target.catalog_row = row
        self.target._catalog = {str(k): (None if pd.isna(v) else v) for k, v in row.items()}
        for col, val in self.target._catalog.items():
            setattr(self.target, col, val)
        self.target._compute_rho_star_if_possible()

    def find_fits(self) -> List[Path]:
        fits = [Path(p) for p in glob.glob(str(self.target.root_dir / "*.fits"))]
        self.target.source_fits = fits
        self.target.save_state()
        return fits

    def extract_all(self) -> None:
        for fp in self.target.source_fits:
            try:
                sector = int(fp.name.split("-")[2][1:])  # 's13' -> 13
            except Exception:
                sector = 0
            extract_data_from_fits_files(str(fp), sector=sector)
        self.target.set_stage(PipelineStage.EXTRACTED)

    def merge_total(self) -> Path:
        catalog_df = pd.DataFrame([self.target.catalog()])
        get_data(str(self.target.root_dir), catalog_df=catalog_df, PL=self.flavour)
        matches = glob.glob(str(self.target.root_dir / f"*{self.flavour}*_*total.csv"))
        if not matches:
            raise FileNotFoundError(f"No *{self.flavour}*_*total.csv in {self.target.root_dir}")
        total_file = Path(matches[0])
        self.target.set_stage(PipelineStage.MERGED)
        return total_file

    def prepare(self) -> Path:
        self.ensure_catalog()
        self.find_fits()
        self.extract_all()
        return self.merge_total()