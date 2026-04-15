#!/usr/bin/env python
# coding: utf-8

# In[29]:


import batman
# import emcee
import glob
import os
import shutil
import math
import corner
import numba
import itertools
import sys
# import gerbls

import numpy           as np
import pandas          as pd
import time            as tm 
import lightkurve      as lk
import deep_transit    as dt
import multiprocessing as mp
# import mr_forecast as mr
import scipy.stats as sst

import matplotlib                      as mpl
import matplotlib.pyplot               as plt
import matplotlib.gridspec             as gridspec
import matplotlib.ticker               as ticker

from   matplotlib.backends.backend_pdf import PdfPages
import mpl_axes_aligner

import astropy.io.fits    as apf
import astropy.units      as units
from   astropy.stats      import sigma_clip
from   astropy.wcs        import WCS
from   astropy.timeseries import BoxLeastSquares
from   astroquery.mast    import Catalogs
from   astroquery         import svo_fps

from multiprocessing import Pool, Process
from wotan           import flatten
from functools       import partial
from ldtk            import LDPSetCreator, BoxcarFilter, TabulatedFilter, SVOFilter
from ldtk.filters    import tess, sdss_z
from IPython.display import display, HTML
from tqdm.auto       import tqdm
import gc
# from pympler.tracker import SummaryTracker

import pymc as pm
import pytensor as PyT
import pytensor.tensor as pt
import arviz as az
from pytensor.graph import Op, Apply

from pytensor import config as pt_config


from typing import List, Dict, Any, Tuple, Optional, Iterable

# import eleanor

import warnings
warnings.filterwarnings("ignore")
display(HTML("<style>.container { width:95% !important; }</style>"))

import config as con
# In[32]:


def mkdir_if_doesnt_exist(outdir, str_new_dir_name):
    # Written by Mallory Harris
    # Description: creates new directory to save data to if it does not already exist
    # Arguments : outdir             = existing directory in which new directory will be located
    #             str_new_dir_name   = string of the new subdirectory of outdir's name

    if os.path.exists(outdir+str_new_dir_name)==False:
        new_outdir = os.path.join(outdir, str_new_dir_name)
        os.mkdir(new_outdir)


# In[37]:
def mk_target_dir_mv_fits_file(fits_file_with_GAIAid, sector_df):
    gaia_ID = fits_file_with_GAIAid.split('-')[2]
    ticid = int(sector_df[sector_df['GAIA_ID'].astype(str)==gaia_ID]['TICID'])
    mkdir_if_doesnt_exist('../Search_target_data/', 'target_tic-'+str(ticid)+'_gaiaID-'+str(gaia_ID))
    os.rename(fits_file_with_GAIAid, '../Search_target_data/target_tic-'+str(ticid)+'_gaiaID-'+str(gaia_ID)+'/'+fits_file_with_GAIAid.split('/')[-1])
    


LDC_for_quadratic = pd.read_csv('../data/LDC_params/table15.dat', 
                                header = None, 
                                sep="\s+", index_col=None,
                               names = ['logg', 'Teff', 'z','L/HP', 'aLSM', 'bLSM',
                                       'aFCM', 'bFCM', 'SQRT(CHI2)', 'qsr', 'PC'])

LDC_PARAMS_MDWARF = LDC_for_quadratic[LDC_for_quadratic['Teff']<4300]
                  

def match_logg_and_teff_for_LDC(df):
    
    # Written by Mallory Harris
    # Description: uses logg and effective temp to calculate quadratic limb darkening parameters based on given  csv
    # Arguments : df = panda dataframe of TIC parameters, specifically Teff and logg
    # Return    : df = panda dataframe with updated quadratic limb darkening parameters

    a = []
    b = []
    bar_format = "{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} targets | {elapsed}<{remaining}"
    pbar = tqdm(total=len(df), smoothing=0.3,  position=1, leave=True, bar_format=bar_format)
    for i in range(len(df)):

        Teff = np.float128(df['Teff'])[i]
        logg =  np.float128(df['logg'])[i]
        pbar.update(1)

        try:
            int(Teff)
            mdwarf_Teff =LDC_PARAMS_MDWARF[LDC_PARAMS_MDWARF['Teff'] == np.median(LDC_PARAMS_MDWARF.iloc[(LDC_PARAMS_MDWARF['Teff'].astype('float128')-Teff+0.01).abs().argsort()[:8]].reset_index(drop=True)['Teff'])]
#             if i%100 == 0:
#                 if len(set(mdwarf_Teff['Teff']))>1:
#                        print(mdwarf_Teff)
            if not abs(logg)>=0.:
                logg = np.median(mdwarf_Teff['logg'])
            aLSM, bLSM =  mdwarf_Teff.iloc[(mdwarf_Teff['logg'].astype('float128')-logg-1E-8).abs().argsort()].iloc[0][['aLSM', 'bLSM']]
            a.append(aLSM)
            b.append(bLSM) 
        except:
            a.append(np.nan)
            b.append(np.nan)
    pbar.close()

    df['aLSM'] = a
    df['bLSM'] = b
    return df

    
def get_catalog_info(ticid, df = False, rtrn_df = False, gaia_id = False):
    try: 
        new_df = df[df['TICID'].astype(int)==int(ticid)]
#         print('checking df, ', new_df)

    except Exception as err:
        print('this is error: ', err)
        
        ctlfile = '../data/final_mdwarf_params.csv'
        mdwarfs = pd.read_csv(ctlfile, iterator =True, chunksize = 100000, index_col=None, header = 0)
        new_df = pd.concat(
            [chunk[chunk['TICID'].astype(int) == int(ticid)] 
            for chunk in mdwarfs]).reset_index(drop=True)

    new_df = match_logg_and_teff_for_LDC(new_df)

#     if len(new_df) == 0:
# #         print('we have a problem')
        
#         if (type(gaia_id)!=bool) and (type(df)!=bool):
#             new_df = df[df['GAIA_ID'].astype(str) == gaia_id]

    if len(new_df) == 0:
#                 print('we have a serious problem')
        ctlfile = '../data/final_mdwarf_params.csv'
        mdwarfs = pd.read_csv(ctlfile, iterator =True, chunksize = 100000, index_col=None, header = 0)
        new_df = pd.concat(
            [chunk[chunk['TICID'].astype(int) == ticid] 
            for chunk in mdwarfs]).reset_index(drop=True)
    if len(new_df) == 0:
        print('we have a problem')
        
        
    if rtrn_df:
        return new_df
    else: 
        return new_df[['aLSM', 'bLSM']].values[0].astype(float), float(new_df['Mass']), float(new_df['eMass']), float(new_df['eMass']), float(new_df['Rad']), float(new_df['eRad']), float(new_df['eRad'])





# In[39]:


column_names = ['TIME', 'FLUX','FLUX_ERR', 'BKG_FLUX', 'BKG_FLUX_ERR', 'QUALITY', 'CENTROID_X', 
                'CENTROID_X_ERR','CENTROID_Y', 'CENTROID_Y_ERR']

def extract_data_from_fits_files(fitsFile, PL = "", sector = 0):
    hdulist=apf.open(fitsFile) #fits time series
    indxs = [i for i  in range(len(hdulist)) if ('Table' in str(hdulist[i]))] #grab the tabular information, which is the data
    tbdata = hdulist[indxs[0]]
    data = tbdata.data
    all_col_names = [jjj.name.upper() for jjj in tbdata.columns]

    params_df = pd.DataFrame(np.nan, range(len(data)), columns = ['TIME'])
    params_df['TIME'] = data['TIME']

    if len(PL)>0:
        PL = PL.upper()+'_'

    flux_cols     = np.array(sorted([name for name in all_col_names if ('FLUX' in name) & ('X_' not in name) & ('K' != name[0]) & ('CAL' != name[0:3])], key=len))
    bkg_cols      = np.array(sorted([name for name in all_col_names if ('BKG' in name) | ('BACKGROUND' in name)], key = len))
    centroid_cols = np.array([name for name in all_col_names if ('CENTR' in name) & ('MOM' not in name)])
    qual_cols     = np.array([name for name in all_col_names if ('QUAL' in name) | ('FLAG' in name)])

    useful_cols = []
    if len(flux_cols) == 0:
        return
    elif len(flux_cols) >1:
#         print(flux_cols)
        for col in flux_cols:
            new_col = col.split('_')[0][:4]
            params_df[new_col+'_FLUX'] = data[col]
            useful_cols.append((col, new_col+'_FLUX'))

    else:
        flux_col = flux_cols[-1]
#         print(flux_col)
        params_df['FLUX'] = data[flux_col]
        useful_cols.append((flux_col, 'FLUX'))

    if len(bkg_cols)>0:
        params_df['BKG_FLUX']  = data[bkg_cols[0]]
        useful_cols.append((bkg_cols[0], 'BKG_FLUX'))

    if len(centroid_cols)>0:
        x_centr = np.array(sorted([name for name in centroid_cols if ('X' in name) | ('1' in name)], key = len))
        y_centr = np.array(sorted([name for name in centroid_cols if ('Y' in name) | ('2' in name)], key = len))
        params_df['CENTROID_X'] = data[x_centr[0]]
        params_df['CENTROID_Y'] = data[y_centr[0]]
        useful_cols.extend([(x_centr[0], 'CENTROID_X'),( y_centr[0], 'CENTROID_Y')])
        
    if len(qual_cols)>0:
        params_df['QUALITY'] = np.sum([np.array(data[qcol_]) for qcol_ in qual_cols], axis=0) 

    for col_ in useful_cols:
        if col_[0]+'_ERR' in all_col_names:
            params_df[col_[1]+'_ERR'] = data[col_[0]+'_ERR']
 
    new_filename = os.path.dirname(fitsFile)+'/'+PL+fitsFile.split('/')[-2]+'_sector'+str(sector).zfill(2)+'.csv'    
    df = params_df.astype(object)

    # print('data frame', df)
    
    df.to_csv(new_filename, index = False)
    


# In[41]:


####DAX FELIZ WINDOW LENGTH FOR WOTAN CODE


def calculate_semi_major_axis(Period, M_star ,R_star): #will replace SMA_AU_from_Period_to_stellar above
    """
    Calculate the semi-major axis using Kepler's third law.

    Parameters:
    ----------
    period : astropy.Quantity
        Orbital period of the planet (in days).
    M_star : astropy.Quantity, optional
        Stellar mass (default is solar mass).

    Returns:
    -------
    semi_major_axis : astropy.Quantity
        Semi-major axis in units of AU.
    """
    Period = Period.to(units.second)  # Convert period to seconds
    M_star = M_star.to(units.kg)  # Convert stellar mass to kg
    R_star = R_star.to(units.m)  # Convert stellar mass to kg

    from astropy import constants as const
    G = const.G

    a_cubed = (G * M_star * Period**2 / (4 * np.pi**2)).to(units.m**3)
    semi_major_axis = a_cubed**(1/3)

    scaled_SMA = (semi_major_axis / R_star).decompose() #now unitless
    SMA_cm = semi_major_axis.to(units.cm)
    return scaled_SMA, SMA_cm.value



def T14(P, R_star, M_star, R_planet, b=0, i=90*units.deg): #will replace Tdur above
    #instead of using R_planet, i'm going to try the minimum planet radius 
    #where I could potentially get a minimum SNR needed
    
    #I'm also going to have to think about the periods that I am going to try for

    """
    Estimate the total transit duration (T14) for a planet using Kepler's third law and transit geometry.

    Parameters:
    ----------
    P : float
        Orbital period of the planet (in days).
    R_star : float
        Radius of the star (in solar radii).
    M_star : float
        Mass of the star (in solar masses).
    R_planet : float
        Radius of the planet (in Earth radii).
    b : float, optional
        Impact parameter (default is 0 for central transit).
    i : astropy.Quantity, optional
        Orbital inclination (default is 90 degrees for edge-on orbit).

    Returns:
    -------
    transit_duration : astropy.Quantity
        The total transit duration (T14) in days.
    """
    import astropy.units as u
    # add units to inputs
    P, R_star, M_star, R_planet = P*units.day, R_star*units.R_sun, M_star*units.M_sun, R_planet*units.R_earth

    # Convert inclination to radians
    i = i.to(units.radian).value

    # Semi-major axis in AU
    scaled_SMA, SMA_cm = calculate_semi_major_axis(Period=P,M_star=M_star,R_star=R_star)
    a = (SMA_cm*units.cm).to(units.AU)

    # Convert units to meters
    R_star = R_star.to(units.m)
    R_planet = R_planet.to(units.m)

    k = (R_planet / R_star).decompose().value  # Planet-to-star radius ratio

    # Calculate the geometric part of the transit duration (dimensionless)
    piece_A = (P.to(units.second) / np.pi).decompose()  # This will give us a time quantity
    piece_B1 = (R_star / a.to(units.m)).decompose().value  # Dimensionless
    piece_B2 = np.sqrt((1 + k)**2 - b**2)  # Dimensionless
    piece_B3 = 1 / np.sin(i)  # Dimensionless

    # Combine the parts for the full duration calculation
#     arcsin_argument = np.clip(piece_B1 * piece_B2 * piece_B3, -1, 1)  # Ensure arcsin argument is valid
    arcsin_argument = piece_B1 * piece_B2 * piece_B3
    angle_radians = np.arcsin(arcsin_argument)  # Result in radians

    # Multiply by the time factor to get the total transit duration
    T14_seconds = (piece_A * angle_radians)  # Now this is in seconds

    return T14_seconds.to(units.day).value

# window_size_in_days = 5* T14(P=maxP, R_star=R_star, M_star=M_star, R_planet=2*R_planet_RE)


# In[42]:


def remove_outliers(time, flux, sigma_lower=6.0, sigma_upper=3., **kwargs):
    outlier_mask = sigma_clip(data=flux,
#                               sigma=sigma,
                              sigma_lower=sigma_lower,
                              sigma_upper=sigma_upper,
                              **kwargs).mask
    # Second, we return the masked light curve and optionally the mask itself
    return outlier_mask

def flatten_lc(time, flux, catalog_df = pd.DataFrame({'Rad':[-1],  'Mass': [-1]}), maxP = 100, R_planet_RE=2):

    # print('catalog df', catalog_df)
    if len(catalog_df)>0:
        M_star = float(catalog_df['Mass'])
        R_star = float(catalog_df['Rad'])
    if not M_star>0:
        M_star = 0.5
    if not R_star>0:
        R_star = 0.5
        
    windw  =3* T14(P=maxP, R_star=R_star, M_star=M_star, R_planet=2*R_planet_RE)
    
#     print('params breaking', maxP, R_star, M_star, R_planet_RE)
    flat_flux, flux_trend = flatten(
    time,                 # Array of time values
    flux,                 # Array of flux values
    method='biweight',
    window_length=windw, #R_planet_RE = in Earth RE, so = 1 # The length of the filter window in units of ``time``
#     edge_cutoff=0.1,      # length (in units of time) to be cut off each edge.
#     break_tolerance=1,  # Split into segments at breaks longer than that
    return_trend=True,    # Return trend and flattened light curve
#     cval=5.0              # Tuning parameter for the robust estimators
    )
#     print('window size currently used?', windw*24, ' Tdur: ', windw/3*24, ' hours')
    return flat_flux, flux_trend


def get_data(ticid_directory, flux_type='APER_', PL = 'TGLC', verbose = False, catalog_df = False, check_PSF = False):
    
    total_time = []
    total_flux = []
    total_flux_err = []
    
    total_flat_flux = []
    total_flat_flux_err = []
    total_flux_trend = []

    trend = []

    flux_col = flux_type+'FLUX'

    files = glob.glob(ticid_directory+'/*_sector*.csv')
#     print('is there a problem with the files?', files)
    for i in files:
        sec = float(i.split('sector')[-1][:2])
        timeseries_df = pd.read_csv(i, index_col = None)
#         print('timeseries_df', timeseries_df)
        if flux_col not in timeseries_df.columns:
            flux_col = 'FLUX'

        timeseries_df = timeseries_df[timeseries_df['QUALITY']==0].reset_index(drop=True)
        timeseries_df_new = timeseries_df[~np.isnan(timeseries_df[flux_col])]
        
        if flux_col+'_ERR' in timeseries_df_new.columns:
            timeseries_df_new[flux_col+'_ERR'] = timeseries_df_new[flux_col+'_ERR']/np.nanmedian(timeseries_df_new[flux_col])
            flux_err = np.array(timeseries_df_new[flux_col+'_ERR'])

        else:
            flux_err = np.full(len(timeseries_df_new), np.std(timeseries_df_new[flux_col]/np.nanmedian(timeseries_df_new[flux_col])))


        timeseries_df_new[flux_col] = timeseries_df_new[flux_col]/np.nanmedian(timeseries_df_new[flux_col])

        time     = np.array(timeseries_df_new.TIME)
        flux     = np.array(timeseries_df_new[flux_col])
        
        
        if len(time) == 0:
            print('check if flux is all null: ', set(timeseries_df[flux_col]))
            print('sector', sec, 'is no good, may want to search PSF')
            continue
        outlier_mask = remove_outliers(time, flux) 
        
        
        clean_time = time[~outlier_mask]
        clean_flux = flux[~outlier_mask]
        clean_err  = flux_err[~outlier_mask]
        
#         print('outlier mask', outlier_mask)
        new_time_series = timeseries_df_new[~outlier_mask]
#         
        #need to work on my flatter_lc function based on Dax's file
        if type(catalog_df) == bool:
        
            flat_flux, flat_trend = flatten_lc(clean_time, clean_flux)
        else:
            flat_flux, flat_trend = flatten_lc(clean_time, clean_flux, catalog_df = catalog_df)
        
        
#         print('catalog df', catalog_df)
#         print('checking flatthings', set(flat_flux), set(flat_trend))
        
        flat_flux_err = np.full(len(flat_flux), np.std(flat_flux))
        
#         print('checking df', new_time_series)
        
        new_time_series[flux_col+'_FLAT'] = flat_flux
        new_time_series[flux_col+'_FLAT_ERR'] = flat_flux_err
        new_time_series[flux_col+'_TREND'] = flat_trend
        new_time_series.to_csv(i, index=False)

        
        if verbose:
            plt.figure(figsize = (20, 6))

            ax = plt.gca()
            ax.set_facecolor('None')
            ax.scatter(time,flux, color = 'brown', zorder = 2, marker = '.')
            ax.plot(clean_time, flat_trend, color = 'k', zorder = 10,)
            
            plt.figure(figsize = (20, 6))
            ax1 = plt.gca()
            ax1.set_facecolor('None')
            ax1.scatter(clean_time,flat_flux, color = 'crimson', zorder = 2, marker = '.')
#             ax1.plot(clean_time, flat_flux, color = 'k', zorder = 10,)

            plt.show()
            
        total_time.extend(clean_time)
        total_flat_flux.extend(flat_flux)
        total_flat_flux_err.extend(flat_flux_err)
        
        total_flux_trend.extend(flat_trend)
        total_flux.extend(clean_flux)
        total_flux_err.extend(clean_err)

#     print('total time length', len(total_time))
    total_time          = np.array(total_time)
    total_flux          = np.array(total_flux)
    total_flux_err      = np.array(total_flux_err)
    total_flat_flux     = np.array(total_flat_flux)
    total_flat_flux_err = np.array(total_flat_flux_err)
    total_flux_trend    = np.array(total_flux_trend)
        
    if len(total_time)>0:
        new_df = pd.DataFrame({'TIME': total_time[np.argsort(total_time)], 'RAW_FLUX': total_flux[np.argsort(total_time)], 
                               'RAW_FLUX_ERR': total_flux_err[np.argsort(total_time)], 'FLUX': total_flat_flux[np.argsort(total_time)],
                              'FLUX_ERR': total_flat_flux_err[np.argsort(total_time)], 'FLUX_TREND': total_flux_trend[np.argsort(total_time)]})
        
        
        #this is getting rid of likely junk segements
#         indexes_split_unorganize = breaking_up_data(total_time, 0.75)   
#         diff_ary = np.array([max(np.array(total_time)[x])-min(np.array(total_time)[x]) for x in indexes_split_unorganize])

#         all_good_indxs =np.concatenate(list(itertools.compress(indexes_split_unorganize, diff_ary>0.5))).ravel()
#         new_df[all_good_indxs].to_csv(ticid_directory+'/'+ticid_directory.split('/')[-1]+'_'+PL+'_'+flux_type+'total.csv', index = False)


        new_df.to_csv(ticid_directory+'/'+ticid_directory.split('/')[-1]+'_'+PL+'_'+flux_type+'total.csv', index = False)
#     elif len(total_time)==0 and flux_type == 'APER_' and check_PSF==True:
#         print('lets try for PSF flux')
#         get_data(ticid_directory, flux_type='PSF_', PL = 'TGLC', verbose = verbose, catalog_df = catalog_df)
#     return np.array(total_time), np.array(total_flux), np.array(total_flux_err)

    


# In[43]:


model_path = '../model_TESS.pth'


def make_LightKurveObject(time, flux, flux_err):
    """ Convert this object to a lightkurve.lightcurve.TessLightCurve object to use for Deep Transit.

    :return: Data in a lightkurve object
    :rtype: `class` lightkurve.lightcurve.TessLightCurve
    """
    lc = lk.TessLightCurve()
    lc.time = time
    lc.flux = flux
    lc.flux_err = flux_err
    return lc


def calc_rudimentary_snr(depth, Tdur, Ntran=1):
    sigma_1hr_15_Tmag = 6283.6147036936645*1E-6

    A = (Ntran**0.5)/sigma_1hr_15_Tmag
    SNR = A*(depth)*((Tdur*24)**0.5)
    return SNR


def plot_lc_with_bboxes(lc_object, bboxes, ax=None, epoch = 0, **kwargs):
    """
    Plot light curve with bounding boxes

    Parameters
    ----------
    lc_object : `~lightkurve.LightCurve` instance

    bboxes : list or np.ndarray
                Bounding boxes in shape (N, 5)

    ax : `~matplotlib.pyplot.axis` instance
                Axis to plot to. If None, create a new one.
    kwargs : dict
                Additional arguments to be passed to `matplotlib.pyplot.plot`

    Returns
    -------
    ax : `~matplotlib.pyplot.axis` instance
                The matplotlib axes object.
    """
    with plt.style.context('grayscale'):
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(12, 26), constrained_layout=False)
            ax.plot(lc_object.time.value, lc_object.flux.value, color = 'k', zorder = 1E5, **kwargs)
            ax.set_xlabel('Time - T0 (hours)', color = 'k', fontsize = 40)
            ax.set_ylabel('Normalized Flux', color = 'k', fontsize = 20)

#         else:
#             ax.plot(lc_object.time.value, lc_object.flux.value,color = 'k', **kwargs)
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection
        recs = []
        val = 0
        for real_mask in bboxes:
            val+=1
#             print(val)
#             print(real_mask[1] - real_mask[3] / 2)

            new_start = real_mask[1] - epoch
#             print('start', real_mask[1], real_mask[4])

            rec = Rectangle(
                (new_start - (real_mask[3])/2, real_mask[2] - (real_mask[4]/2)),
                real_mask[3],
                real_mask[4],
                facecolor='indianred',       # Transparent fill
                edgecolor='indianred',  # Mistyrose border
                linewidth=30,
                zorder=5
            )
            recs.append(rec)
#             SNR = (np.log(1-real_mask[0])/0.15)*-1 #-> SNR calc from DT
            SNR = calc_rudimentary_snr(real_mask[3], real_mask[4])
            ax.text(
                new_start+ abs(real_mask[3]),
                real_mask[2] + 1/2*abs(real_mask[4]),
                s='snr: '+f"{SNR:.2f}",
                color='r',
                verticalalignment="top",
                bbox=dict(alpha=0.75, color='None'),
                clip_on=True, 
                fontsize = 12, zorder = 1E4
            )

        pc = PatchCollection(recs, lw=0.2, zorder=5, match_original = True)
        collection = ax.add_collection(pc)
        collection.set_zorder(5)
    return ax



def DT_analysis(time, flux, flux_err, confidence, DT_Quite=True, is_flat = True):
    """ 
    @author: D. Dragomir, P.Steimle


    Function for the Deep Transit analysis (Cui et al. 2021) that returns only single transit events

    """

    # create a new dataset for the while loop where transits can be masked out
    
    
#     confidence = 1-np.exp(-0.15*snr)
    print('time len', len(time))
    if DT_Quite == True:
        save_stdout = sys.stdout
        save_stderr = sys.stderr
        sys.stdout = open('.trash.txt', 'w')
        sys.stderr = open('.trash.txt', 'w')

        # do check for transits with DT
        DT_model = dt.DeepTransit(make_LightKurveObject(time, flux, flux_err),  is_flat=is_flat)
        bboxes = DT_model.transit_detection(model_path, confidence_threshold=confidence)

        sys.stdout = save_stdout
        sys.stderr = save_stderr

    else:
        # do check for transits with DT
        DT_model = dt.DeepTransit(make_LightKurveObject(time, flux, flux_err), is_flat=is_flat)
        bboxes = DT_model.transit_detection(model_path, confidence_threshold=confidence)

        # if only 1 or 0 boxes were found, break the loop and continue

    return bboxes
 


# In[44]:


def transit_mask(t, period, duration, T0, buffer=0.2):
    
    # Works with numba, but is not faster
    #all units given in days
#     print('this is evil', 'lim time', min(t), max(t), 'T0', T0, 'duration', duration, 'period', period) 

    mask = np.abs((t - T0 + (0.5 * period)) % period - (0.5 * period)) < duration+buffer
    return mask

def transit_mask_tensors(t, period, duration, T0):
    
    # Works with numba, but is not faster
    #all units given in days
    #this will return a boolean tensor object
    mask = pt.abs(((t - T0 + (0.5 * period)) % period) - (0.5 * period)) < (duration * 2. / 3.)
    return mask

def find_breaks(time, val = 27.):
    time = np.array(time)[np.argsort(time)] 
    t    = np.diff(time)
    inds = np.where( t>val)[0]
    return inds + 1

# def sort_data_by_times(time, flux):
#     return time[np.argsort(time)], flux[np.argsort(time)]

def running_median(data, kernel=25):
    """Returns sliding median of width 'kernel' and same length as data """
    
#     print('kernel', kernel)
    
    idx = np.arange(kernel) + np.arange(len(data) - kernel + 1)[:, None]
    idx = idx.astype(np.int64)  # needed if oversamplinfg_factor is not int
    med = np.nanmedian(data[idx], axis=1)

    # Append the first/last value at the beginning/end to match the length of
    # data and returned median
#     print('length of med (if 0, need to return 0)', len(med))
    if len(med)>0:
        first_values = med[0]
        last_values = med[-1]
        missing_values = len(data) - len(med)
        values_front = int(missing_values * 0.5)
        values_end = missing_values - values_front
        med = np.append(np.full(values_front, first_values), med)
        med = np.append(med, np.full(values_end, last_values))
        med[np.isinf(np.abs(med))] = 0

        return med
    else:
        return np.zeros(len(data))

    
    
def breaking_up_data(time, break_val = 27., min_size = 0.5):
    time = np.array(time)
    brk = np.append(np.append([0], find_breaks(time, break_val)), [len(time)])

    indexes = []
    for i in range(len(brk)-1):
        r = np.arange(brk[i],brk[i+1], 1)

        if len(r)>1:
            if np.ptp(time[r])>min_size:
#                 if min_size!= 0.5:
#                     print(np.ptp(time[r]), ' > ', min_size, '?')
                indexes.append(r)
    return indexes



def find_common_element_indices(arrays):
    if not arrays or len(arrays) < 2:
        return [], {}

    
    all_elements = [element for arr in arrays for element in arr]
    elements, indxs, cnts = np.unique(np.round(all_elements, 2), return_index = True, return_counts = True)
    
    common_elements = elements[np.where(cnts>1)]
    print(common_elements)
    
    result = {}
    for ele in common_elements:
        result[ele] = []
        for i, arr in enumerate(arrays):
            if np.isin(ele, np.round(arr, 2)):
                result[ele].append((i, list(np.round(arr,2)[0]).index(ele)))
    return common_elements, result# Example usage:




def check_multiples(arr):
    """
    Checks if an array contains elements that are multiples of each other.

    Args:
      arr: A list of integers.

    Returns:
      indxs tuples of multiple factors
    """
    indxs = []
    n = len(arr)
    for i in range(n):
        for j in range(n):
            if i != j and arr[i] != 0:
                factor = arr[i]/arr[j]
                
                if np.logical_or(np.abs(factor - np.rint(factor))<0.02, np.abs(1/factor - np.rint(1/factor))<0.02):
                    indxs.append((i, j))
    return indxs





# def checking_BLS_periodicity(per, period_array, t0, t0_array):
#     factors = np.array(period_array)/per
#     print('factors',np.round(factors, 5))
#     factor_indxs = np.where(np.logical_or(np.abs(factors - np.rint(factors))<0.03, np.abs(1/factors - np.rint(1/np.array(factors)))<0.03))[0]
    
#     print('factor indexes', factor_indxs)

#     pop_per = np.nan
#     pop_per_indx = np.nan

#     keep_factor = 1
# #     if 1. in np.round(factors, 7):
# #         keep_factor = -1E5
# #         print('its 1')

# #         return per, keep_factor
#     rep_indxs = np.where(np.rint(factors[factor_indxs])==1.)[0]     
# #     print('indexes of repeated periods', rep_indxs)
#     new_period = per
#     not1_indxs = np.where(np.rint(factors[factor_indxs])!=1.)
#     if len(rep_indxs)>0:
#         keep_factor = -1
#         val = 0
#         while (1. in np.round(period_array/new_period, 1) and val<len(period_array)+1):
#             val+=1                    
#             new_periods = catching_periods_repeated_and_offset(new_period,t0, np.array(period_array), np.array(t0_array), rep_indxs)
#             if len(new_periods)> 1:
#                 new_period  = min(new_periods)
#                 keep_factor = abs(per/new_period)

#                 pop_per = per
                
#                 pop_per_indxs = np.where(np.array(period_array)==pop_per)[0]

#                 if len(pop_per_indxs)>0:
#                     pop_per_indx = pop_per_indxs[0]
#                 else:
#                     pop_per = np.nan
#                     keep_factor = -1
                    
                
#             elif (len(new_periods) == 1) and (new_periods[0] == per):
#                 keep_factor = -1E5
                

#         if val>len(period_array):
#             keep_factor = -2
    

#     elif len(not1_indxs[0])>0:
        
#         factors_not1 = factors[not1_indxs]
# #         factors_not1[factors_not1>1] = 1.

#         factors_not1 = np.unique(factors_not1)
        
#         keep_factor = max(1/factors_not1)

        
#         #note: the following line is assuming that generally the min period is the true period, and multiples are aliases. This allows us to run MCMC on fewer periods. However, if the true period is the longer one, we're in trouble

#         if keep_factor<1:

#             pop_per = float(np.unique(np.array(period_array)[np.where(factors == 1/keep_factor)])[0])
#             if pop_per>0:
#                 pop_per_indx = np.where(np.array(period_array)==pop_per)[0][0]
                
#         else:
#             keep_factor = -1*keep_factor


#     if pop_per_indx != np.nan:
#         print('popping index: ', pop_per_indx, ' or period ', pop_per, 'from periods: ', period_array)
# #     print('what exactly are these? ', pop_per_indx, pop_per)
    
#     print('final of periodicity check; period: ', new_period, ' keep factor', keep_factor)
#     return new_period, keep_factor, pop_per, pop_per_indx
        


# ---------- tiny primitives ----------

def _near_integer(x: float, tol: float = 0.03) -> Tuple[bool, Optional[int]]:
    k = int(np.rint(x))
    if np.isfinite(x) and abs(x - k) <= tol:
        return True, k
    return False, None

def _nearest_epoch_to_time(t_target: float, T0: float, P: float) -> float:
    n = int(np.rint((t_target - T0) / P))
    return T0 + n * P

def _offset_to_ephemeris(t_target: float, T0: float, P: float) -> float:
    return abs(t_target - _nearest_epoch_to_time(t_target, T0, P))

def _index_of_period(periods: Iterable[float], p: float, rel_tol: float = 1e-3) -> Optional[int]:
    arr = np.asarray(list(periods), dtype=float)
    if arr.size == 0:
        return None
    diffs = np.abs(arr - p) / np.maximum(arr, 1e-12)
    i = int(np.argmin(diffs))
    return i if diffs[i] <= rel_tol else None

def checking_BLS_periodicity(
    per, period_array, t0, t0_array,
    ratio_tol=0.03, epoch_tol_perc=0.05, rel_tol=1e-3,
    min_period=0.25, max_period=None
):
    """
    Compare a new candidate (per, t0) against existing (period_array, t0_array).
    Returns (new_period, keep_factor, pop_per, pop_per_indx) with same semantics as before.
    """
    periods = list(map(float, period_array))
    t0s     = list(map(float, t0_array))

    new_period = float(per)
    keep_factor = 1.0
    pop_per = np.nan
    pop_per_indx = np.nan

    for i, (p_old, t0_old) in enumerate(zip(periods, t0s)):
        r = p_old / per
        rel1, k1 = _near_integer(r, tol=ratio_tol)       # p_old ~ k1 * per
        rel2, k2 = _near_integer(1.0 / r, tol=ratio_tol) # per   ~ k2 * p_old
        if not (rel1 or rel2):
            continue

        k = int(max(1, (k1 if rel1 else k2)))
        if k == 1:
            # exact repeat sentinel
            keep_factor = -2
        shorter = min(per, p_old)
        new_is_shorter = (per <= p_old)

        offset_old = _offset_to_ephemeris(t0, t0_old, p_old)
        epoch_tol = max(epoch_tol_perc * shorter, 2.5/24)  # use shorter period scale for tolerance
        aligned = (offset_old <= epoch_tol)

        if aligned:
            if new_is_shorter:
                # Keep new, pop old
                return float(per), (keep_factor + float(k)), float(p_old), i
            # Reject new; keep existing shorter. Use shorter for masking.
            return float(p_old), float(-k), pop_per, pop_per_indx

        # Misaligned alias → try offset masking if offset behaves like a submultiple
        if offset_old > 0:
            r_off = min(p_old, per) / offset_old
            rel1_, kx = _near_integer(r_off, tol=ratio_tol)
            rel2_, ky = _near_integer(1.0 / r_off, tol=ratio_tol)
            if (rel1_ or rel2_):
                k_off = int(max(1, (kx if rel1_ else ky)))
                if (offset_old < max(p_old, per) / 2) and (k_off <10):
                    if ((min_period is None or offset_old >= min_period) and
                        (max_period is None or offset_old <= max_period)):
                        return float(offset_old), (keep_factor + 10.0), pop_per, pop_per_indx

    # Novel period
    return new_period, keep_factor, pop_per, pop_per_indx

def checking_aliases_repeated_periodic_planets(per_ary, t0_ary, q_ary):
    final_indexs = np.full(len(per_ary), True)
    all_t0 = []
    for iii in range(len(per_ary)):
        all_t0.append([t0_ary[iii] + per_ary[iii]*np.arange(-50, 50)])
    rep_t0, indxs_of_rep_t0 = find_common_element_indices(all_t0)
    for t0 in rep_t0:
        indx_t0 = set(np.array(indxs_of_rep_t0[np.round(t0, 2)])[:,0])
#         print('indexes: ', indx_t0)
        bad_ndx = np.where(np.array(q_ary) == min([q_ary[x] for x in indx_t0]))
        final_indexs[bad_ndx] = False
    print('final_indxs', final_indexs)
    return final_indexs


def checking_multiples_and_duplicate_periodic_planets(per_ary, t0_ary, d_ary, q_ary): 
    final_indexs = np.full(len(per_ary), True)
    div_period_ary = np.ones(len(per_ary))
    
    multiple_indxs = check_multiples(per_ary)
    for ndxs in multiple_indxs:
        i, j = ndxs
        if np.abs(per_ary[i]/per_ary[j] - 1) < 0.03 and np.abs(d_ary[i]/d_ary[j] - 1) > 0.1:
            print('likely a binary')
            final_indexs[[i, j]] = False
            
        else:
            
            if np.abs(per_ary[i]/per_ary[j] - 1) >= 0.03:
                bad_ndx = np.where(np.array(q_ary) == min((q_ary[i], q_ary[j])))
#                 print('this should be int val', bad_ndx, 'and it should be one of these 2 vals', i, j)
                final_indexs[bad_ndx] = False
                
            if  np.abs(per_ary[i]/per_ary[j] - 1) < 0.03:
                t0_diff   = (t0_ary[i] - t0_ary[j])/per_ary[i]
                new_indxs = check_multiples([t0_diff, per_ary[i]])
                
                if len(new_indxs) >0: 
                    bad_ndx = np.where(np.array(q_ary) == min((q_ary[i], q_ary[j])))
                    good_ndx = np.where(np.array(q_ary) == max((q_ary[i], q_ary[j])))

                    final_indexs[bad_ndx] = False
                    div_period_ary[good_ndx] = max([t0_diff/per_ary[i], per_ary[i]/t0_diff])
    
    rep_t0_final_indxs = checking_aliases_repeated_periodic_planets(per_ary, t0_ary, q_ary)
                      
    final_indexs = np.logical_and(rep_t0_final_indxs, final_indexs)
    print('keeping these periods', np.array(per_ary)[final_indexs])
    return final_indexs, div_period_ary




def catching_periods_repeated_and_offset(per, t0, per_array, t0_array, rep_indxs):
    
    new_periods = []
    
    diff_tc = np.abs(np.array(t0_array)[rep_indxs] - t0)
    
    for iii in range(len(diff_tc)):
        n = np.ceil(diff_tc[iii]/per)
        min_diff_tc = np.nanmin(np.abs(diff_tc[iii] - np.array([n-1, n, n+1])*per))
        
        if np.rint(min_diff_tc) != 0.:
#             frac_per = per/min_diff_tc
            new_periods.append(min_diff_tc)
        
        else: 
            new_periods.append(per)
    
    if len(new_periods)>0:
        nper, indxs = np.unique(np.round(new_periods, 1), return_index=True)
        new_periods = list(np.array(new_periods)[indxs])
    
    return new_periods
                        



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
    print('max indx', max_indx, ' len power results ', len(power_results))
    if max_indx == 0 or max_indx >= len(power_results):
        return np.arange(len(power_results))
    else:        
        return np.arange(len(power_results)-max_indx)


        

    
def checking_BLS_odd_even_binaries(stats, t0, period, depth):
    if stats['depth_odd'][0]/stats['depth_even'][0] > 10:
        print('keeping odd - would rather keep binary as 2 objects than miss a planet from period alias')
        t0 = t0+period
        period = 2*period
        depth = stats['depth_odd'][0]

    elif stats['depth_even'][0]/stats['depth_odd'][0] > 10:
        print('keeping even - would rather keep binary as 2 objects than miss a planet from period alias')
        t0 = t0
        period = 2*period
        depth = stats['depth_even'][0]
    return t0, depth, period


def check_rules_to_continue_BLS(results, index):
    sorted_pwr = np.sort(np.unique(results.power_final))
#     stdv_pwr = np.nanstd(np.diff(results.power_final))

    stdv_pwr = np.nanstd(np.sort(results.power_final)[:-1])

#     stdv_pwr = np.nanstd(results.power_final)
    

    period = results.period[index]

    rule_1 = np.abs(np.diff(sorted_pwr[[-1, -2]])) > 2.*stdv_pwr

    pwr_copy = np.array(results.power_final).copy()
    pwr_copy[index] = -np.inf

    period_2 = results.period[np.argmax(pwr_copy)]
    
    factor = np.arange(2, 6)
    check_multipls = sorted(list(1/factor)+list(factor))


    if (not rule_1) and np.isin(np.round(period/period_2, 2), np.round(check_multipls, 2)):
        print('double checking rule 1')
        rule_1 = np.abs(np.diff(sorted_pwr[[-1, -3]])) > 2.*stdv_pwr
        pwr_copy = np.array(results.power_final).copy()
        pwr_copy[np.argmax(pwr_copy)] = -np.inf

        period_3 = results.period[np.argmax(pwr_copy)]
        
        if (not rule_1) and  np.isin(np.round(period/period_3, 2), np.round(check_multipls, 2)):
            print('double checking rule 2')
#             stdv_pwr1 = np.nanstd(np.sort(np.diff(results.power_final))[:-3])
            stdv_pwr1 = np.nanstd(np.sort(results.power_final)[:-3])

#             print('standard deviations', stdv_pwr, stdv_pwr1)
#             print('number stdev', np.abs(np.diff(sorted_pwr[[-1, -4]]))[0]/stdv_pwr1)


            rule_1 = np.abs(np.diff(sorted_pwr[[-1, -4]])) > 2.25*stdv_pwr1
    return rule_1


# def using_TLS_to_find_periodic_signals(time, flux, u, verbose = False, show_progress_info = True, save_phaseFold = True,
#                                        intransit = [], periods = [], T0 = [], Tdur = [], 
#                                        depths = [], first=True):
    
#     time_diff = max(time)-min(time)
#     print('time diff', time_diff)
#     max_per = min(time_diff, 100.)
#     if first == True:
#         intransit = np.full(len(time), False)
#         periods = []
#         T0 = []
#         Tdur = []
#         depths = []

#     time_new = np.array(time)[~intransit]
#     flux_new = np.array(flux)[~intransit]
#     if len(time_new)>0:
#         start = tm.time()

#         durations = np.linspace(0.02, 0.5, 75)
        
#         model = transitleastsquares(time_new, flux_new)
#         results = model.power(
#             period_min=0,
#             period_max=max_per,
# #             transit_depth_min=ppm*10**-6,
# #             oversampling_factor=10,
# #             duration_grid_step=1.02,
#             u=ab,
#             limb_dark='quadratic',
# #             M_star = 1,
# #             M_star_max=1.1
#             n_transits_min = 1,
#             show_progress_info = show_progress_info
#             )

                
#         index = np.argmax(results.power)
        
#         period    = results.period
#         val_triangles = min(results.power)-np.std(results.power)

# #         print('period', period, 'index ', index)
#         end = tm.time()
#         if round(results.T0, 4) in [round(x, 4) for x in T0]:
        
#             intransit = np.logical_or(intransit, transit_mask(time, results.period, results.duration, results.T0))
#             print('FOUND THE SAME PLANET: CONTINUING')
#             return using_TLS_to_find_periodic_signals(time, flux, u, intransit=intransit, verbose = verbose,  periods = periods, T0 = T0, Tdur = Tdur, depths = depths, first = False)
#         if verbose:
            

# #             print('plot 1')
#             plt.figure(figsize = (10,6))

#             ax = plt.gca()
#             ax.set_facecolor('None')
#             ax.scatter(time_new,flux_new, color = 'k', zorder = 10, marker = '.')
            
#             faux_intransit = np.logical_or(intransit, transit_mask(time, results.period, results.duration, results.T0))
            
#             ax.scatter(np.array(time)[~faux_intransit],np.array(flux)[~faux_intransit], 
#                        color = 'r', zorder = 11, marker = '.', alpha = 0.3)

#             plt.ylabel(r'N. Flux')#, fontsize = 40)
#             plt.xlabel('Time', fontsize = 40)

#             plt.figure(figsize = (5, 5))

#             ax = plt.gca()
#             ax.set_facecolor('None')
#             ax.scatter(period,val_triangles, color = 'r', marker = '^', s=20, zorder = 10)

#             plt.xlim(np.min(results.periods), np.max(results.periods))
#             for n in range(2, 10):
#                 ax.scatter( n*period,val_triangles, color = 'maroon', marker = '^', s=20, zorder = 10, alpha= 0.8)
#                 ax.scatter(period / n,val_triangles, color = 'maroon', marker = '^', s=20, zorder = 10, alpha= 0.8)


#             plt.ylabel(r'SDE')#, fontsize = 40)
#             plt.xlabel('Period (days)')#, fontsize = 40)
        
        
#             ax.plot(results.periods, results.power, color = 'k', lw=1)
#             ax.xaxis.label.set_color('k')        #setting up X-axis label color to yellow
#             ax.yaxis.label.set_color('k')          #setting up Y-axis label color to blue


#             t0 = results.T0
#             duration = results.duration
            
#             plt.show()
#             plt.close()
            
#             plt.figure(figsize = (5, 5))
#             ax2 = plt.gca()

#             ax2.set_facecolor('None')

#             x = ((time - t0 + 0.5*period) % period) -( 0.5*period)
#             m = np.abs(x) < 0.5
#             ax2.scatter(
#                 x[m],
#                 np.array(flux)[m],
#                 color='gray',
#                 s=5,
#                 alpha=0.8,
#                 zorder=2)

#             x_new = np.linspace(-0.5, 0.5, 1000)
#             f = model.model(x_new + t0, period, duration, t0)

#             ax2.plot(x_new, f, color='grey', lw = 1, alpha = 0.6, zorder = 5)
#             ax2.set_xlabel('Phase')#, color = 'k', fontsize = 40)
#             ax2.set_ylabel('Relative Flux')#, color = '#CC9966', fontsize = 40);
#             plt.show()
# #             print('T0: ', results.T0, 'duration: ', results.duration, 'npoints_dur: ', np.ceil(results.duration/30.))
            
            
#         if not np.abs(np.diff(np.array(sorted(results.power))[[-1, -4]]))>2*np.nanstd(results.power) or len(time_new)==0 :
#             print('FOUND NO PLANET: FINISHING THIS LC')
#             return np.array(periods), np.array(T0), np.array(Tdur), np.array(depths), intransit
        
        
#         else: 
#             intransit = np.logical_or(intransit, transit_mask(time, results.period, (2*results.duration/24)+(1/6), results.T0[0]))
            
#             depths.append(results.depth)
#             periods.append(results.period)
#             T0.append(results.T0)
#             Tdur.append(results.duration)
#             print('FOUND A PLANET: CONTINUING')
#             return using_TLS_to_find_periodic_signals(time, flux, u, intransit=intransit, verbose = verbose,  periods = periods, T0 = T0, Tdur = Tdur, depths = depths, first = False)


# ------------------------
# Helper Functions
# ------------------------

def compute_log_likelihood(flux, model_flux, flux_err):
    residuals = flux - model_flux
    chi2 = np.sum((residuals / flux_err)**2)
    return -0.5 * chi2

def compute_BIC(logL, n, k):
    return k * np.log(n) - 2 * logL

def compute_AIC(logL, k):
    return 2 * k - 2 * logL


def build_box_model(time, t0, duration, depth, period = -1):
    """
    Build a box-shaped transit model in the time domain (not phase-folded).
    
    Parameters:
    ----------
    time : array
        Time array in days.
    period : float
        Orbital period in days.
    t0 : float
        Transit midpoint in days.
    duration : float
        Transit duration in days.
    depth : float
        Transit depth (fractional).
    
    Returns:
    -------
    model_flux : array
        Flux model with transits applied.
    """
    model_flux = np.ones_like(time)
    # Compute phase relative to t0 without folding
    # For each time point, check if it's in transit for any cycle
    if period>-1: 
        phase_offset = (time - t0) % period
    else: 
        phase_offset = (time - t0) 
        
    in_transit = (phase_offset < duration / 2) | (phase_offset > (period - duration / 2))
    # Apply depth to in-transit points
    model_flux[in_transit] -= depth
    return model_flux

# ------------------------
# Main Recursive Function
# ------------------------

def using_BLS_recursive(time, flux, flux_err = None, intransit=None,
                            verbose=True, plot=True, max_planets=10,
                            min_SNR=7, min_SDE = 10,
                            periods=None, T0=None, Tdur=None, depths=None, first=False):
    """
    Recursive multi-planet search using GERBLS pyFastBLS + run_double and BIC/AIC-based model selection.
    """

    if intransit is None:
        intransit = np.zeros_like(time, dtype=bool)
    if flux_err is None:
        flux_err = np.std(flux) * np.ones_like(flux)

    if first:
        periods, T0, Tdur, depths = [], [], [], []
    df = 1E-4
    durations = np.linspace(0.01, 0.5, 50)

    # Mask in-transit points
    time_new, flux_new, flux_err_new = time[~intransit], flux[~intransit], flux_err[~intransit]
    if len(time_new) < 10:
        print("Stopping: insufficient data.")
        return np.array(periods), np.array(T0), np.array(Tdur), np.array(depths), intransit

    # Prepare data for BLS
    freq_fact_prelim = df/min(durations)*(np.nanmax(time_new)-np.nanmin(time_new))**2
    freq_fact_exp = np.ceil(np.log10(freq_fact_prelim))    

    start = tm.time()

    model     = BoxLeastSquares(time_new, flux_new)
    max_per   = np.min([50., (max(time_new)-min(time_new))*4/5])
    max_dur   = np.min([0.5,  (max(time_new)-min(time_new))/2])
        

    results   = model.autopower(durations[durations<max_dur], frequency_factor = np.max([10, (10**(freq_fact_exp-1))/2]), maximum_period=max_per)#, objective='snr', )

    end = tm.time()    

    my_median = running_median(results.power, kernel = min((25, int(len(time_new)/10))))
    
    
    results['power_final'] = results.power - my_median

    check_pwr_final_indxs = checking_last_BLS_power_for_artificial_inflation(results['power_final'])
    index = np.argmax(results.power_final[check_pwr_final_indxs])
#     print('my median ', my_median,  'my indexes ', len(check_pwr_final_indxs), 'power final ', results['power_final'])
    
    
    
    period = results.period[index]
    t0 = results.transit_time[index]
    duration = results.duration[index]
    depth = results.depth[index]
    
    print('depth found', results.depth[index])
    
    sorted_results = np.sort(results['power_final'])

    # Compute SDE
    mad = sst.median_abs_deviation(results['power_final'])
    print('mad', mad)
    if mad == 0.:
        print('mad == 0: standard deviation is', np.std(results['power_final']), ', mad without running median is', sst.median_abs_deviation(results['power']))

        mad = np.nanmax([
            1e-5,
            np.std(results['power_final']),
            sst.median_abs_deviation(results['power'])
        ])
    results['SNR'] =  results['power_final']/(mad/0.67)
    
    sde  = (results['power_final'][index] - np.mean(results['power_final'])) / np.std(results['power_final'])
#     sde2 = (sorted_results[-1] - sorted_results[-2] ) / np.std(sorted_results[:-2])

#     if verbose:
    print(f"Candidate: P={period:.4f} d, SDE={sde:.2f},min_SDE={min_SDE:.2f}, SNR = {results['SNR'][index]:.4f}, min_SNR = {min_SNR:.2f}")
#     print(f"Candidate: P={period:.4f} d, SDE={sde:.2f}, SDE2={sde2:.2f}, min_SDE={min_SDE:.2f}")

    

    mask = results['SNR'] > min_SNR


    if plot: # and np.ceil(results.duration[index]/np.nanmedian(np.diff(time)))>=3:

        plt.figure(figsize = (10, 6))
        val_triangles = min(results.SNR)-np.std(results.SNR)
        ax = plt.gca()
        ax.scatter(period, val_triangles, color = 'r', marker = '^', s=20, zorder = 10)

        plt.xlim(np.min(results.period), np.max(results.period))
        for n in range(2, 10):
            ax.scatter( n*period,val_triangles, color = 'maroon', marker = '^', s=20, zorder = 10, alpha= 0.8)
            ax.scatter(period / n,val_triangles, color = 'maroon', marker = '^', s=20, zorder = 10, alpha= 0.8)
        plt.ylabel(r'SNR')#, fontsize = 40)
        plt.xlabel('Period (days)')#, fontsize = 40)

        ax.plot(results.period, results.SNR, color = 'k', lw=0.65)

        

        plt.show()
        plt.close()
        

        if duration<period:
            plt.figure(figsize = (5, 5))
            ax2 = plt.gca()

            x = ((time_new - t0 + 0.5*period) % period) -( 0.5*period)
            m = np.abs(x) < 0.5
            ax2.scatter(
                x[m],
                flux_new[m],
                color='k',
                s=5,
                alpha=0.8,
                zorder=10)

            x_new = np.linspace(-0.5, 0.5, 1000)

            f = model.model(x_new + t0, period, duration, t0)

            f2 = build_box_model(x_new+t0, t0, duration, depth, period)
            ax2.plot(x_new, f, color='grey', lw = 1, alpha = 0.6, zorder = 5)
#             ax2.plot(x_new, f2, color='violet', lw = 1, alpha = 0.6, zorder = 5)

#             ax2.set_xlim(-0.5, 0.5)
            ax2.set_xlabel('Phase')#, color = 'k', fontsize = 40)
            ax2.set_ylabel('Relative Flux')#, color = 'k', fontsize = 40);
            plt.show()
    if not mask.any() or sde<min_SDE:
        if sde<min_SDE:
            print("Stopping: SDE below threshold.")
        if not mask.any():
            print('Stopping: SNR below threshold.')
        return np.array(periods), np.array(T0), np.array(Tdur), np.array(depths), intransit

    single = False

    try:
        stats = model.compute_stats(period, duration, t0)
        print('number transit times in baseline:', len(stats["transit_times"][stats["per_transit_count"]>0]), ' \nnumber of point in each transit: ', stats["per_transit_count"][stats["per_transit_count"]>0], '\ntransit likelihood:', stats["per_transit_log_likelihood"][stats["per_transit_count"]>0])
        transit_times_all = stats["transit_times"][np.where(stats["per_transit_count"]>0)[0]]
        single = False
        if len(transit_times_all)<2:
            single = True

    except ValueError as err:
        print('getting error: ', err)


    repeat = False
    if (len(periods)>0) and (not single):# and len(factor_indxs)>0:

        new_period, keep_factor, pop_per, pop_per_indx  = checking_BLS_periodicity(period, periods, t0, T0)

        if keep_factor>0:
            repeat=False
            intransit = np.logical_or(intransit, transit_mask(time, new_period, duration, t0))
            period = new_period  
            if pop_per > 0:
                print(f'popping_period: {pop_per} = {np.array(periods)[pop_per_indx]}, and keeping period {period}, as the old is {pop_per/period}x the new')
#                 pop_indx = periods.index(pop_per)
               
                periods.pop(pop_per_indx)
                T0.pop(pop_per_indx)
                Tdur.pop(pop_per_indx)
                depths.pop(pop_per_indx)
                
        if keep_factor < -50 :

            intransit = np.logical_or(intransit,  transit_mask(time, new_period/2, duration, t0, buffer = 0.3))
        else:
            intransit = np.logical_or(intransit, transit_mask(time, new_period, duration, t0, buffer = 0.3))

        if keep_factor<0:
            repeat = True

    # Build models for likelihood
    model_flux = model.model(time_new, period, duration, t0)

    null_flux = np.ones_like(flux_new)
    
    logL_transit = compute_log_likelihood(flux_new, model_flux, flux_err_new)

    logL_null = compute_log_likelihood(flux_new, null_flux, flux_err_new)

    n, k = len(time_new), 3
    bic_transit = compute_BIC(logL_transit, n, 3)

    bic_null = compute_BIC(logL_null, n, 1)
    delta_BIC = bic_null - bic_transit


    if (single) or (repeat):
        print('masking this single transit or repeat detection and continuing')

#             return np.array(periods), np.array(T0), np.array(Tdur), np.array(depths), intransit
#         else:
        if single:
            print("Candidate rejected - masking single event")
        elif repeat:
            print("Candidate rejected - masking repeat discovery transits")
        else:
                print("Candidate rejected: insufficient BIC improvement - will still mask and try again.")
            # Mask transits using your transit_mask
        intransit = np.logical_or(intransit, transit_mask(time, period, duration, t0))

        return using_BLS_recursive(time, flux, intransit=intransit,flux_err=flux_err,periods=periods, T0=T0, Tdur=Tdur, depths=depths, first=False)

    # Accept candidate
    periods.append(period)
    T0.append(t0)
    Tdur.append(duration)
    depths.append(depth)
    print('depths all', depths, 'transt durations: ', Tdur)

    if verbose:
        print(f"Accepted planet: P={period:.4f} d")

    # Mask transits using your transit_mask
    intransit = np.logical_or(intransit, transit_mask(time, period, duration, t0))

    if len(periods) >= max_planets:
        print("Reached max planets.")
        return np.array(periods), np.array(T0), np.array(Tdur), np.array(depths), intransit

    # Recurse
    return using_BLS_recursive(time, flux, intransit=intransit,flux_err=flux_err,periods=periods, T0=T0, Tdur=Tdur, depths=depths, first=False)

    

def fitting_periodic_planets(time, flux, flux_err, pers, t0s, depths, ab, intransit, verbose=True, save_phaseFold = False, total_time = True, data_file = '.', chain_diff = 0):
#             print('period', periods_multis[jjj])
    periods  = []
    T0_vals  = []
    Tdur     = []
    depth    = []
    SNR_vals = []
    
    params_df_all   = []
    
    if type(total_time) == bool:
        total_time = time


    print('len(t0s)', len(t0s))
    for iii in range(len(pers)): 
        if pers[iii]>0.25:

            params_df, conv, conv_attempt = pymc_new_general_function(time, flux, flux_err, t0s[iii], [pers[iii], ab, depths[iii]], 'Periodic')
            
            if len(params_df)>0:

                params_df.loc[len(params_df)] = [None] * len(params_df.columns) 

                params_df_all.append(params_df)

                T0_, period_, depth_, tdur_, SNR = params_df.loc['t0', 'mean'], params_df.loc['Per', 'mean'], params_df.loc['depth', 'mean'], params_df.loc['dur', 'mean'], params_df.loc['SNR', 'mean']
#                 print('params df', params_df)

#                 print('checking convergence 1')
                pd.DataFrame({'TICID':[con.TICID], 't0':[T0_], 'per':[period_], 'depth':[depth_], 'converged': [True], 'conv_on_run':[conv_attempt]}).to_csv('../checking_convergence_output/'+str(con.TICID)+'_'+str(round(T0_, 5))+'_Yconv_per.csv')


            else:
                T0_, period_, tdur_, depth_, SNR = np.nan, np.nan, np.nan, np.nan, 0

#                 print('checking convergence 2')
                pd.DataFrame({'TICID':[con.TICID], 't0':[t0s[iii]], 'per':[pers[iii]], 'depth':[depths[iii]], 'converged': [False], 'conv_on_run':[np.nan]}).to_csv('../checking_convergence_output/'+str(con.TICID)+'_'+str(round(t0s[iii], 5))+'_Nconv_per.csv')


            if not np.isnan(T0_):

                periods.append(period_)
                T0_vals.append(T0_)
                Tdur.append(tdur_)
                depth.append(depth_)
                SNR_vals.append(SNR)


                intransit = np.logical_or(intransit, transit_mask(total_time, period_, float(tdur_), float(T0_), buffer = 0.5))

            gc.collect()

    if len(params_df_all)>0:

        params_df_all = pd.concat(params_df_all)
    

#     print('params df 3', params_df_all)
    return T0_vals, periods, depth, Tdur, SNR_vals, intransit, params_df_all

            
        
def searching_for_periodic_signals(data_file, ab, TLS = False, verbose = True, save_file = True, save_phaseFold = False):
    
    params = []
    periods  = []
    T0_vals  = []
    Tdur     = []
    depth    = []
    SNR_vals = []


    total_time_flux_df = pd.read_csv(data_file).dropna(subset = ['FLUX'])

    
    
    time      = np.array(total_time_flux_df['TIME'].astype(float))
    flux      = np.array(total_time_flux_df['FLUX'].astype(float))
    flux_err  = np.array(total_time_flux_df['FLUX_ERR'].astype(float))
    intransit = np.full(len(time), False)

    print('running search on all data')

    ###running search on whole dataset
    if TLS:
        periods_multis, T0_multis, Tdur_multis, depth_multis, intransit_per = using_TLS_to_find_periodic_signals(time, flux, u = ab, intransit =intransit, verbose = verbose)

    else:

        print('recursive BLS')
        periods_multis, T0_multis, Tdur_multis, depth_multis, intransit_per = using_BLS_recursive(time, flux, intransit = intransit, verbose = verbose, first=True)
        
#     print('depths', depth_multis)
    
    if len(T0_multis)>0:
        
        sort_indices = np.argsort(periods_multis)
        
        nt0_vals, nperiods, ndepth, nTdur, nSNR_vals, intransit, params_df = fitting_periodic_planets(time, flux, flux_err, periods_multis[sort_indices], T0_multis[sort_indices], depth_multis[sort_indices], ab, intransit_per, verbose, save_phaseFold, data_file = data_file)

#         params_df.loc[len(params_df)] = [None] * len(params_df.columns) 
        if len(params_df)>0:
            params.append(params_df)

        periods.extend(nperiods)
        T0_vals.extend(nt0_vals)
        Tdur.extend(nTdur)
        depth.extend(ndepth)
        SNR_vals.extend(nSNR_vals)
    
    
    ###running search on split subsets
    print('running search on chunked data')
    
    
    
    indexes_split_unorganize = breaking_up_data(time)   
    indexes_split = sorted(indexes_split_unorganize, key=lambda x: len(x), reverse=True)
    
    if len(indexes_split) == 1 and (np.any(np.array(periods)<10.) or len(periods)==0):
        indexes_split_unorganize = breaking_up_data(time, break_val = 1.)   
        indexes_split = sorted(indexes_split_unorganize, key=lambda x: len(x), reverse=True)


#     print('split indexes lengths: ', [len(x) for x in indexes_split])
    if len(indexes_split)>1:
        for iii, indxs_s in enumerate(indexes_split):
            if len(indxs_s) == 1:
                print('too few indexes to run again')
                continue
    #         print('len time', len(new_time))
    

            intransit_split = np.array(intransit[indxs_s])
        
        
            split_time     = np.array(time[indxs_s])
            split_flux     = np.array(flux[indxs_s])
            split_flux_err = np.array(flux_err[indxs_s])

            masked_time = split_time[intransit_split]
            masked_flux = split_flux[intransit_split]
            masked_flux_err = split_flux_err[intransit_split]

            if len(split_time)==0:
                print('WHY ISNT THIS WORKING')

                print(indexes_split)
                continue

            else:
                if TLS:
                    periods_multis, T0_multis, Tdur_multis, depth_multis, intransit_small = using_TLS_to_find_periodic_signals(split_time, split_flux, u = ab, verbose = verbose)
                else:
                    periods_multis, T0_multis, Tdur_multis, depth_multis, intransit_small = using_BLS_recursive(masked_time, masked_flux, verbose = verbose, periods = periods.copy(), T0 =T0_vals.copy(), Tdur = Tdur.copy(), depths = depth.copy())

#             intransit[indexes_split[iii]] = np.logical_or(intransit_split, intransit_small)


            toss_bool = np.isin(np.round(periods_multis, 1), np.round(periods, 1))


            print('periods to not run again', np.array(periods_multis)[toss_bool], 'rest to run', np.array(periods_multis)[~toss_bool])

            if len(np.array(periods_multis)[~toss_bool])>0:
                
                try:
                    sub_per, sub_t0, sub_depth = [list(np.array(x)[~toss_bool]) for x in [periods_multis, T0_multis, depth_multis]]
                except Exception as err:
                    print(err)
                    print(f'len(per) {len(periods_multis)}, len(T0) {len(T0_multis)}, len(Depth) {len(depth_multis)}, len(bool) {len(toss_bool)}')
                    toss_bool = np.full(False, len(T0_multis))


                nt0_vals_split, nperiods_split, ndepth_split, nTdur_split, nSNR_vals_split, intransit, nparams_df = fitting_periodic_planets(split_time, split_flux, split_flux_err, sub_per, sub_t0, sub_depth, ab, intransit, verbose, save_phaseFold=False, total_time = time, data_file=data_file)


                if len(nparams_df)>0:
                    params.append(nparams_df)

                periods.extend(nperiods_split)
                T0_vals.extend(nt0_vals_split)
                Tdur.extend(nTdur_split)
                depth.extend(ndepth_split)
                SNR_vals.extend(nSNR_vals_split)

    print('done with multis ', iii+1, ':', len(indexes_split))

#     print('params df 2', params)

    
    only_per_intransit = np.full(len(time), False)

    for iii in range(len(periods)):
#         print('period is', periods[iii])
        new_transit_planet = transit_mask(time, periods[iii], Tdur[iii], T0_vals[iii])
#         print('intransit '+str((iii+1)*7), new_transit_planet, len(np.where(new_transit_planet)[0]), len(new_transit_planet))

        only_per_intransit = np.logical_or(only_per_intransit, new_transit_planet)
        

    intransit_indexes =  np.where(only_per_intransit)
    
    return periods, T0_vals, Tdur, depth, only_per_intransit, SNR_vals, intransit_indexes, params

# ---------- stacked-table block utilities ----------

def split_summary_blocks(summary: pd.DataFrame) -> List[pd.DataFrame]:
    blocks, cur_rows, cur_idx = [], [], []
    for idx, row in summary.iterrows():
        if row.isna().all():
            if cur_rows:
                B = pd.DataFrame(cur_rows, index=cur_idx, columns=summary.columns)
                B.index = [str(r).strip().lower() for r in B.index]
                B.columns = [str(c).strip().lower() for c in B.columns]
                blocks.append(B.copy())
                cur_rows, cur_idx = [], []
        else:
            cur_rows.append(row.values)
            cur_idx.append(str(idx))
    if cur_rows:
        B = pd.DataFrame(cur_rows, index=cur_idx, columns=summary.columns)
        B.index = [str(r).strip().lower() for r in B.index]
        B.columns = [str(c).strip().lower() for c in B.columns]
        blocks.append(B.copy())
    return blocks

def _cell_to_scalar(block: pd.DataFrame, param: str, col: str = "mean") -> Optional[float]:
    try:
        val = float(block.loc[param.lower(), col.lower()])
        return val if np.isfinite(val) else None
    except Exception:
        return None

def extract_planet_params(block: pd.DataFrame) -> Dict[str, Optional[float]]:
    return {
        "P_mean":     _cell_to_scalar(block, "Per",   "mean"),
        "P_sd":       _cell_to_scalar(block, "Per",   "sd"),
        "t0_mean":    _cell_to_scalar(block, "t0",    "mean"),
        "t0_sd":      _cell_to_scalar(block, "t0",    "sd"),
        "depth_mean": _cell_to_scalar(block, "depth", "mean"),
        "depth_sd":   _cell_to_scalar(block, "depth", "sd"),
        "dur_mean":   _cell_to_scalar(block, "dur",   "mean"),
        "dur_sd":     _cell_to_scalar(block, "dur",   "sd"),
        "snr_mean":   _cell_to_scalar(block, "snr",   "mean"),
        "snr_sd":     _cell_to_scalar(block, "snr",   "sd"),
    }


# ---------- alias resolution helpers ----------

def _find_near_integer_aliases(P: np.ndarray, tol_abs: float, use_rel: bool, tol_rel: float) -> List[List[int]]:
    n = len(P)
    adjacency = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            r = max(P[i], P[j]) / min(P[i], P[j])
            ok, k = _near_integer(r, tol=tol_abs)
            if not (ok and (k is not None) and (k >= 1)):
                continue
            if use_rel and (abs(r - k) / max(k, 1) > tol_rel):
                continue
            adjacency[i].append(j)
            adjacency[j].append(i)
    return adjacency

def _grouping_connected_components(adj: List[List[int]]) -> List[List[int]]:
    n = len(adj)
    seen = np.zeros(n, dtype=bool)
    groups = []
    for i in range(n):
        if seen[i]:
            continue
        stack, comp = [i], []
        seen[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        groups.append(sorted(comp))
    return groups

def _alignment_tol_days(
    dur_short: float, z_align: float, dur_frac: float,
    sd_t0_short: float, sd_t0_long: float, Psd_short: float, n_cycles: int
) -> float:
    sigma_align = np.hypot(sd_t0_short, sd_t0_long)
    if (Psd_short > 0) and (n_cycles != 0):
        sigma_align = np.hypot(sigma_align, abs(n_cycles) * Psd_short)
    base = z_align * sigma_align
    floor = (dur_frac * dur_short) if (np.isfinite(dur_short) and dur_short > 0) else 0.0
    return max(base, floor)

def _aligns_by_short_ephemeris(
    i: int, j: int,
    P: np.ndarray, t0: np.ndarray, sd_t0: np.ndarray, Psd: np.ndarray, dur: np.ndarray,
    z_align: float, dur_frac: float
) -> Tuple[bool, Dict[str, float]]:
    # determine short vs long
    if P[i] <= P[j]:
        i_short, j_long = i, j
    else:
        i_short, j_long = j, i

    P_s, t0_s = P[i_short], t0[i_short]
    t_long = t0[j_long]
    sd_t0_s = float(sd_t0[i_short] or 0.0)
    sd_t0_l = float(sd_t0[j_long]  or 0.0)
    Psd_s = float(Psd[i_short] or 0.0)
    dur_s = float(dur[i_short] if np.isfinite(dur[i_short]) else dur[j_long])

    n = int(np.rint((t_long - t0_s) / P_s)) if (P_s > 0) else 0
    tol = _alignment_tol_days(dur_s, z_align, dur_frac, sd_t0_s, sd_t0_l, Psd_s, n)
    off = _offset_to_ephemeris(t_long, t0_s, P_s)
    return (np.isfinite(off) and off <= tol), {"pair": (i, j), "off": float(off), "tol": float(tol)}


def _offset_period_candidate(
    P_short: float, P_long: float,
    t0_short: float, t0_long: float,
    *,
    ratio_tol_abs: float = 0.05,
    kmax: int = 10,
    P_min: float = 0.25,
    P_max: float | None = None
) -> float | None:
    """
    If periods are near-integer related but misaligned, test whether the epoch
    offset to the short ephemeris looks like a 1/k submultiple of the short period.
    Return that offset as a candidate period if plausible; else None.
    """
    if not (np.isfinite(P_short) and np.isfinite(P_long) and P_short > 0 and P_long > 0):
        return None
    off = _offset_to_ephemeris(t0_long, t0_short, P_short)
    if not (np.isfinite(off) and 0 < off < max(P_short, P_long) / 2):
        return None

    r_off = P_short / off
    rel1, k1 = _near_integer(r_off, tol=ratio_tol_abs)
    rel2, k2 = _near_integer(1.0 / r_off, tol=ratio_tol_abs)
    if not (rel1 or rel2):
        return None

    k = int(max(1, (k1 if rel1 else k2)))
    if k >= kmax:
        return None

    if P_max is not None and off > P_max:
        return None
    if off < P_min:
        return None

    return float(off)




def resolve_period_aliases_from_summary(
    summary: pd.DataFrame,
    *,
    ratio_tol_abs: float = 0.05,     # absolute |r - k| tolerance for near-integer ratio
    ratio_rel_use: bool = False,     # if True, also require |r-k|/k <= ratio_tol_rel
    ratio_tol_rel: float = 0.02,     # relative tolerance (2%)
    z_depth: float = 2.0,            # require depth_mean - z_depth*depth_sd > 0
    z_align: float = 2.5,            # sigma multiplier for epoch alignment
    dur_frac: float = 0.25,          # floor as fraction of short duration
    snr_sd_floor: float = 1e-6,      # denom floor in stability score
    # record offset suggestions (not used to choose winners unless you decide to)
    offset_kmax: int = 10,
    offset_P_min: float = 0.25,
    offset_P_max: Optional[float] = None
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Resolve near-integer period aliases between planet candidates in a stacked summary.

    Returns
    -------
    kept_period_by_idx : np.ndarray
        For each candidate (block) i, the period value of the group's winner.
        (You can use this to update other tables; no need for a True/False mask.)
    decisions : list of dict
        One dict per alias group with winner, reason, diagnostics, and any
        offset-as-period suggestions observed for misaligned pairs.
    """
    blocks = split_summary_blocks(summary)
    N = len(blocks)
    params = [extract_planet_params(B) for B in blocks]

    P   = np.array([p["P_mean"]     for p in params], dtype=float)
    t0  = np.array([p["t0_mean"]    for p in params], dtype=float)
    sd0 = np.array([p["t0_sd"]      for p in params], dtype=float)
    dpt = np.array([p["depth_mean"] for p in params], dtype=float)
    sdd = np.array([p["depth_sd"]   for p in params], dtype=float)
    dur = np.array([p["dur_mean"]   for p in params], dtype=float)
    Psd = np.array([p["P_sd"]       for p in params], dtype=float)
    SNRm= np.array([p["snr_mean"]   for p in params], dtype=float)
    SNRs= np.array([p["snr_sd"]     for p in params], dtype=float)

    adj    = _find_near_integer_aliases(P, tol_abs=ratio_tol_abs, use_rel=ratio_rel_use, tol_rel=ratio_tol_rel)
    groups = _grouping_connected_components(adj)

    kept_period = np.array(P, copy=True)
    decisions: List[Dict[str, Any]] = []

    for g in groups:
        if len(g) == 1:
            i = g[0]
            decisions.append({"members": g, "winner_idx": i, "winner_period": float(P[i]),
                              "reason": "no_alias_relation", "details": []})
            kept_period[i] = P[i]
            continue

        cand = sorted(g, key=lambda idx: (P[idx], idx))
        records = []

        for idx in cand:
            depth_ok = (dpt[idx] - z_depth * sdd[idx]) > 0

            align_ok = True
            checks: List[Dict[str, float]] = []
            offset_suggestions: List[float] = []

            for jdx in cand:
                if jdx == idx:
                    continue
                ok, diag = _aligns_by_short_ephemeris(idx, jdx, P, t0, sd0, Psd, dur, z_align, dur_frac)
                checks.append(diag)
                if not ok:
                    align_ok = False
                    # Try offset-as-period suggestion using the shorter ephemeris
                    if P[idx] <= P[jdx]:
                        P_s, P_l = P[idx], P[jdx]
                        t_s, t_l = t0[idx], t0[jdx]
                    else:
                        P_s, P_l = P[jdx], P[idx]
                        t_s, t_l = t0[jdx], t0[idx]
                    offP = _offset_period_candidate(
                        P_s, P_l, t_s, t_l,
                        ratio_tol_abs=ratio_tol_abs, kmax=offset_kmax,
                        P_min=offset_P_min, P_max=offset_P_max
                    )
                    if offP is not None:
                        offset_suggestions.append(offP)

            sm = float(SNRm[idx] if np.isfinite(SNRm[idx]) else 0.0)
            ss = float(SNRs[idx] if np.isfinite(SNRs[idx]) and SNRs[idx] >= 0 else 0.0)
            score = sm / (1.0 + max(ss, snr_sd_floor))

            records.append({
                "idx": idx,
                "period": float(P[idx]),
                "depth_ok": bool(depth_ok),
                "align_ok": bool(align_ok),
                "score": float(score),
                "align_checks": checks,
                "offset_suggestions": sorted(set(offset_suggestions))
            })

        winners = [r for r in records if r["depth_ok"] and r["align_ok"]]
        if winners:
            win = sorted(winners, key=lambda r: (r["period"], -r["score"], r["idx"]))[0]
            reason = "kept shortest passing (alignment & depth)"
        else:
            win = sorted(records, key=lambda r: (-r["score"], r["period"], r["idx"]))[0]
            reason = "no candidate passed both; kept best stability"

        widx = int(win["idx"])
        for i in g:
            kept_period[i] = P[widx]

        decisions.append({
            "members": cand,
            "winner_idx": widx,
            "winner_period": float(P[widx]),
            "reason": reason,
            "details": records
        })

    return kept_period, decisions



def apply_alias_resolution_to_table(
    table: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    ratio_tol_abs: float = 0.05,
    ratio_rel_use: bool = False,
    ratio_tol_rel: float = 0.02,
    z_depth: float = 2.0,
    z_align: float = 2.5,
    dur_frac: float = 0.25,
    snr_sd_floor: float = 1e-6,
    offset_kmax: int = 10,
    offset_P_min: float = 0.25,
    offset_P_max: Optional[float] = None
) -> pd.DataFrame:
    """
    Run resolve_period_aliases_from_summary on `summary` and write results into `table`.

    Assumptions:
      * The number/order of summary blocks matches the subset of `table` rows where Ptype == 'Period'.
      * We do not overwrite the original 'period' column; instead we add 'kept_period'.
      * Winners get Default=True; losers Default=False; Notes are appended.

    Returns: updated copy of `table` with 'kept_period', 'Default', and 'Notes' updated.
    """
    df = table.copy()

    # ensure bookkeeping columns
    if 'Notes' not in df.columns:
        df['Notes'] = ''
    if 'Default' not in df.columns:
        df['Default'] = True

    # subset rows that correspond to the alias resolution blocks
    per_idx = df.index[df['Ptype'] == 'Period'].to_numpy()
    per_view = df.loc[per_idx]

    kept_period, decisions = resolve_period_aliases_from_summary(
        summary,
        ratio_tol_abs=ratio_tol_abs,
        ratio_rel_use=ratio_rel_use,
        ratio_tol_rel=ratio_tol_rel,
        z_depth=z_depth,
        z_align=z_align,
        dur_frac=dur_frac,
        snr_sd_floor=snr_sd_floor,
        offset_kmax=offset_kmax,
        offset_P_min=offset_P_min,
        offset_P_max=offset_P_max
    )

    # sanity: align lengths by truncation (preferably they should match exactly)
    n_map = min(len(per_idx), len(kept_period))
    if len(per_idx) != len(kept_period):
        # If you prefer, raise an exception instead of silently truncating:
        # raise ValueError("Mismatch between number of 'Period' rows and summary blocks.")
        per_idx = per_idx[:n_map]
        kept_period = kept_period[:n_map]

    # write kept_period for 'Period' rows
    df.loc[per_idx, 'kept_period'] = kept_period

    # default all 'Period' rows to True; we will set losers to False per-group
    df.loc[per_idx, 'Default'] = True

    # annotate winners/losers per decision group
    # map block index -> table index
    block_to_table = {bi: int(per_idx[bi]) for bi in range(n_map)}

    for d in decisions:
        members = [bi for bi in d["members"] if bi < n_map]
        if not members:
            continue
        w_block = int(d["winner_idx"])
        if w_block >= n_map:
            # winner outside truncated range; skip
            continue

        w_row = block_to_table[w_block]
        member_rows = [block_to_table[bi] for bi in members]

        # loser rows
        losers = [r for r in member_rows if r != w_row]
        if losers:
            df.loc[losers, 'Default'] = False

        # build compact notes
        # planet numbers if available, else row indices
        def _pnum(irow: int) -> str:
            try:
                return str(int(df.at[irow, 'planet_name']))
            except Exception:
                return f"row{int(irow)}"

        if losers:
            keptP = float(df.at[w_row, 'kept_period']) if 'kept_period' in df.columns else float(df.at[w_row, 'period'])
            loser_note = f"alias of planet {_pnum(w_row)} (kept P={keptP:.6f})"
            df.loc[losers, 'Notes'] = df.loc[losers, 'Notes'].astype(str)
            df.loc[losers, 'Notes'] = np.where(
                df.loc[losers, 'Notes'].str.len() > 0,
                df.loc[losers, 'Notes'] + "; " + loser_note,
                loser_note
            )

        # winner note
        if len(member_rows) > 1:
            others = [r for r in member_rows if r != w_row]
            other_pnums = ",".join(_pnum(r) for r in others)
            win_note = f"kept vs aliases {other_pnums}"
            cur = df.at[w_row, 'Notes']
            df.at[w_row, 'Notes'] = (cur + "; " + win_note) if (isinstance(cur, str) and len(cur) > 0) else win_note

    return df



def executing_total_periodic_search(data_file, ticid, catalog_df = False, TLS = False, verbose = True, save_intrans = True, save_time = True, save_phaseFold = True):

    tm1 = tm.time()
    column_names = ['TICID', 'planet_name', 'period', 'T0', 'Tdur', 'depth', 'SNR']
    params = []
    planet_df = pd.DataFrame(columns=column_names)

    ab, smass, smass_min, smass_max, sradius, sradius_min, sradius_max = get_catalog_info(ticid, df = catalog_df)
    print('PERIODIC SEARCH')
    periods_multi, T0_multi, Tdur_multi, depth_multi, intransit, SNR, n, params = searching_for_periodic_signals(data_file, ab, TLS,  verbose, save_file = save_intrans)
    if len(params)>0:
        params = pd.concat(params)
        
    print('params df', params)
    print('init number of periodic planets', len(periods_multi),  periods_multi)
#     nnn = 0

#     if len(set(np.round(periods_multi, 2)))<len(periods_multi):
    if len(periods_multi)>0:
#         final_indxes, per_div = checking_multiples_and_duplicate_periodic_planets(periods_multi, T0_multi, depth_multi, SNR)   

#         periods_multi = np.array(periods_multi)/per_div
#         print('new periodic planets', len(periods_multi), periods_multi)

#     elif len(periods_multi) == 1:
#         final_indxes = np.array([True])
# #     else: 
# #         final_indxes = range(len(T0_multi)) 
#     else:
#         final_indxes = []
        
        
    

        for jjj in range(len(periods_multi)):
            planet_name = int(jjj+1)
            print('index', jjj)


            planet_df.loc[len(planet_df.index)] = [int(ticid), planet_name, periods_multi[jjj], T0_multi[jjj], Tdur_multi[jjj], min([1-depth_multi[jjj], depth_multi[jjj]]), SNR[jjj]]

        print('done ', ticid, planet_df)

    if save_time:
        tm2 = tm.time()
        dict_time_sectors = {'TICID':[], 'time_to_run':[], 'split_searches':[]}
        dict_time_sectors['TICID'].append(ticid)
        dict_time_sectors['time_to_run'].append(tm2-tm1)
        dict_time_sectors['split_searches'].append(n)

        pd.DataFrame(dict_time_sectors).to_csv(os.path.dirname(data_file)+'/time_periodic_took_to_run.csv', index= False, mode = 'a')
    
    if len(planet_df)>0:
        
        file_name = os.path.dirname(data_file)+'/found_planet_init_params'
        kept = glob.glob(file_name+'*')

        csv_name = '.csv'
        for i in range(len(kept)):
            csv_name = '_'+str(i)+'_'+csv_name[-4:]

        planet_df.to_csv(file_name+csv_name, index = False)
        

    
    return intransit, planet_df, params


# def approximate_common_denominator(float1, float2, precision=10**5):
#     int1 = int(float1 * precision)
#     int2 = int(float2 * precision)

#     gcd = math.gcd(int1, int2)
#     return gcd / precision

# # In[54]:
# def check_if_singles_are_periodic(T0_lst):
#     new_rrr = []
#     T0_vals = []
#     for j in range(len(T0_lst)):
#         diff = np.array(T0_lst)-T0_lst[j]
#         for i in range(len(diff)):
#             for k in range(len(diff)):
#                 if k == j or i == j:
#                     continue
#                 else:
#                     rrr = np.array(approximate_common_denominator(diff[i], diff[k]))
#                     if len(rrr[rrr>2.5])>0:
#                         new_rrr.append(list(rrr[rrr>2.5]))
#                         T0_vals.extend([T0_lst[i], T0_lst[j], T0_lst[k]])


#     set_rrr = [ele for ind, ele in enumerate(new_rrr) if ele not in new_rrr[:ind]]    
#     T0_vals = list(set(T0_vals))
    
#     T0_per_vals = []
#     for val in list(set_rrr):
#         pers = new_rrr[new_rrr == val]
#         T0_min = min(T0_vals)
# #         print(T0_vals)
#         per_max = max(pers)
#         T0_per_vals.append([T0_min, per_max, T0_vals])
#     return T0_per_vals


# ---------- Prep: build Δt/m clusters ----------
# ---------- singles prep and scoring (unchanged behavior; short) ----------

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



# ---------- small dataframe helpers ----------

def _ensure_bookkeeping_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Notes' not in df.columns:
        df['Notes'] = ''
    if 'Default' not in df.columns:
        df['Default'] = True
    return df

def _add_periodic_rows_from_modes(df: pd.DataFrame, singles_df: pd.DataFrame, modes: List[Dict[str, Any]]) -> pd.DataFrame:
    if len(modes) == 0:
        return df

    s_idx  = singles_df.index.to_numpy()
    added_rows = []

    for m in modes:
        if m.get('support', 0) < 3:
            continue

        mem_local = m['members']
        mem_dfidx = s_idx[mem_local]
        mem_nums  = df.loc[mem_dfidx, 'planet_name'].astype(int).tolist()

        next_num = (int(df['planet_name'].max()) + 1) if 'planet_name' in df.columns else 1

        P_new, T0_new = float(m['P']), float(m['T0'])
        Td_new = float(np.nanmedian(df.loc[mem_dfidx, 'Tdur']))  if 'Tdur'  in df.columns else 0.10
        Dp_new = float(np.nanmedian(df.loc[mem_dfidx, 'depth'])) if 'depth' in df.columns else np.nan

        added_rows.append({
            'TICID': df['TICID'].iloc[0] if 'TICID' in df.columns else None,
            'planet_name': next_num,
            'Ptype': 'Period',
            'period': P_new, 'T0': T0_new, 'Tdur': Td_new, 'depth': Dp_new,
            'SNR': np.nan,
            'Notes': f"made of singles {','.join(str(n) for n in mem_nums)}",
            'Default': True
        })

        df.loc[mem_dfidx, 'Default'] = False
        df.loc[mem_dfidx, 'Notes'] = df.loc[mem_dfidx, 'Notes'].astype(str)
        df.loc[mem_dfidx, 'Notes'] = np.where(
            df.loc[mem_dfidx, 'Notes'].str.len() > 0,
            df.loc[mem_dfidx, 'Notes'] + f"; now planet {next_num}",
            f"now planet {next_num}"
        )

    if len(added_rows) > 0:
        df = pd.concat([df, pd.DataFrame(added_rows)], ignore_index=True)
    return df

def _attach_remaining_singles(
    df: pd.DataFrame,
    epoch_tol_scale: float, fixed_epoch_tol: float,
    use_depth_for_attach: bool, depth_ratio_max_attach: float
) -> pd.DataFrame:
    periodic_df = df[df['Ptype'] == 'Period'].copy()
    singles_df  = df[(df['Ptype'] == 'Single') & (df['Default'] == True)].copy()
    if len(singles_df) == 0 or len(periodic_df) == 0:
        return df

    P  = np.asarray(periodic_df['period'], dtype=float)
    T0 = np.asarray(periodic_df['T0'],     dtype=float)
    Td = np.asarray(periodic_df['Tdur'],   dtype=float) if 'Tdur'  in periodic_df else np.full(len(P), np.nan)
    Dp = np.asarray(periodic_df['depth'],  dtype=float) if 'depth' in periodic_df else np.full(len(P), np.nan)

    s_T0 = np.asarray(singles_df['T0'],    dtype=float)
    s_Td = np.asarray(singles_df['Tdur'],  dtype=float) if 'Tdur'  in singles_df else np.full(len(s_T0), np.nan)
    s_Dp = np.asarray(singles_df['depth'], dtype=float) if 'depth' in singles_df else np.full(len(s_T0), np.nan)

    per_idx_map = periodic_df.index.to_numpy()
    sgl_idx_map = singles_df.index.to_numpy()

    attached_to: Dict[int, List[int]] = {}

    for s_local, (t0_s, td_s, d_s) in enumerate(zip(s_T0, s_Td, s_Dp)):
        best_i, best_off = None, None
        for i, (p_i, t0_i, td_i, d_i) in enumerate(zip(P, T0, Td, Dp)):
            if not np.isfinite(p_i) or p_i <= 0:
                continue
            t_near = _nearest_epoch_to_time(t0_s, t0_i, p_i)
            off = abs(t0_s - t_near)
            tol = max(fixed_epoch_tol, epoch_tol_scale * (td_i if np.isfinite(td_i) else td_s))
            if off > tol:
                continue
            if use_depth_for_attach and np.isfinite(d_s) and np.isfinite(d_i):
                dmax, dmin = max(d_s, d_i), max(min(d_s, d_i), 5e-5)
                if (dmax / dmin) > depth_ratio_max_attach:
                    continue
            if (best_off is None) or (off < best_off):
                best_off, best_i = off, i
        if best_i is not None:
            s_global = sgl_idx_map[s_local]
            p_global = per_idx_map[best_i]
            attached_to.setdefault(p_global, []).append(s_global)

    for p_global, s_list in attached_to.items():
        added_nums = df.loc[s_list, 'planet_name'].astype(int).tolist()
        add_note = f"added singles {','.join(str(n) for n in added_nums)}"
        cur = df.at[p_global, 'Notes']
        df.at[p_global, 'Notes'] = (cur + "; " + add_note) if (isinstance(cur, str) and len(cur) > 0) else add_note

        period_num = int(df.at[p_global, 'planet_name'])
        df.loc[s_list, 'Default'] = False
        df.loc[s_list, 'Notes'] = df.loc[s_list, 'Notes'].astype(str)
        df.loc[s_list, 'Notes'] = np.where(
            df.loc[s_list, 'Notes'].str.len() > 0,
            df.loc[s_list, 'Notes'] + f"; now planet {period_num}",
            f"now planet {period_num}"
        )
    return df
def annotate_planet_table_from_singles_numeric(
    df,
    *,
    min_support=3,
    use_iterative=True,
    use_depth=True,
    attach_to_known=True,
    epoch_tol_scale=0.25, fixed_epoch_tol=0.05,
    use_depth_for_attach=True, depth_ratio_max_attach=1.75,
    m_max=7, P_min=0.25, P_max=None, rel_merge=0.01,
    local_span=0.02, local_n=41
):
    """
    Mutates a copy of df:
      1) Adds 'Notes' and 'Default' if missing ('' and True).
      2) Groups singles into new periodic planets (adds new rows).
      3) (Optional) Attaches remaining singles to existing periodic planets (no mask).
    """
    df = _ensure_bookkeeping_cols(df)

    singles_df = df[df['Ptype'] == 'Single'].copy()
    if len(singles_df) >= min_support:
        s_T0 = singles_df['T0'].to_numpy(float)
        s_Td = singles_df['Tdur'].to_numpy(float)  if 'Tdur'  in singles_df else np.full(len(singles_df), np.nan)
        s_Dp = singles_df['depth'].to_numpy(float) if 'depth' in singles_df else np.full(len(singles_df), np.nan)

        t0, dur, dep, groups, phase_win = prepping_singles_for_periodic_check(
            s_T0, durations=s_Td, depths=s_Dp,
            max_missed_transits=m_max, P_min=P_min, P_max=P_max, rel_merge=rel_merge, min_support=min_support
        )

        if use_iterative:
            modes = extract_all_modes_iterative(
                t0, dur, dep, min_support=min_support,
                prep_kwargs={'max_missed_transits': m_max, 'P_min': P_min, 'P_max': P_max, 'rel_merge': rel_merge},
                scorer_kwargs={'use_depth': use_depth, 'depth_zmax': 2.5, 'depth_ratio_max': 1.75,
                               'depth_floor': 5e-5, 'local_span': local_span, 'local_n': local_n}
            )
        else:
            modes = score_once_modes(
                t0, dur, dep, groups, phase_win, min_support=min_support,
                use_depth=use_depth, depth_zmax=2.5, depth_ratio_max=1.75,
                depth_floor=5e-5, local_span=local_span, local_n=local_n
            )

        df = _add_periodic_rows_from_modes(df, singles_df, modes)

    if attach_to_known and (df['Ptype'].eq('Single') & df['Default']).any() and (df['Ptype'].eq('Period')).any():
        df = _attach_remaining_singles(
            df, epoch_tol_scale=epoch_tol_scale, fixed_epoch_tol=fixed_epoch_tol,
            use_depth_for_attach=use_depth_for_attach, depth_ratio_max_attach=depth_ratio_max_attach
        )
    return df


def singles_search(ticid, data_total, intransit = [], catalog_df = False, confidence = 0.5,  verbose = True, run_1 = True, data_file = ''):
    
    print('SINGLES SEARCH')
    column_names = ['TICID', 'planet_name', 'period', 'T0', 'Tdur', 'depth']

    if not run_1:
        column_names+=['SNR']
    planet_df = pd.DataFrame(columns=column_names)
    

    df = pd.read_csv(data_total).dropna(subset = ['FLUX'])
    total_time, total_flux, total_flux_err = [np.array(df[col]) for col in ['TIME', 'FLUX', 'FLUX_ERR']]
#     print('checking time again', len(total_time))

    if len(intransit)>0:
#         print('evil bs', intransit, len(intransit), len(np.where(intransit)[0]))
        total_time = total_time[~intransit]
        total_flux = total_flux[~intransit]
        total_flux_err = total_flux_err[~intransit]

    indexes_split_unorganize = breaking_up_data(total_time, break_val = 0.5, min_size = 1.)  
    all_good_indxs = []

    if len(indexes_split_unorganize)>1:
        diff_ary = np.array([max(np.array(total_time)[x])-min(np.array(total_time)[x]) for x in indexes_split_unorganize])
        all_good_indxs =np.concatenate(list(itertools.compress(indexes_split_unorganize, diff_ary>1))).ravel()
   
    if len(all_good_indxs) == 0: 
        all_good_indxs = list(range(len(total_time)))
        
#     print('all good indexes (i.e., itertools result: ', type(all_good_indxs), len(total_time))
    list_1 = list(all_good_indxs)
    list_2 = []
    for index, value in enumerate(total_time):
        list_2.append(index)
#     print('check differences in good indexes and indexes: ',  [item for item in list_1 if item not in list_2])
#     print('type of time array, ', type(total_time))
    total_time = total_time[all_good_indxs]
    total_flux = total_flux[all_good_indxs]
    total_flux_err =total_flux_err[all_good_indxs]

    ab, smass, smass_min, smass_max, sradius, sradius_min, sradius_max = get_catalog_info(ticid, df = catalog_df)
#     print('val', ab)
    
    params_df = []

    if len(total_time)>0:

        bboxes = DT_analysis(total_time, total_flux, total_flux_err, confidence)
    #     print('ran bboxes', bboxes)

        print('number singles found', len(bboxes))
        t0_singles, dur_singles, depth_singles = [],[],[]
        if len(bboxes)>0:
            n_events = len(bboxes)
            for j, boxes in enumerate(bboxes):
#                 SNR = calc_rudimentary_snr(boxes[3], boxes[4])
#                 if SNR>9:

                t0_singles.append(boxes[1])
                dur_singles.append(boxes[3])
                depth_singles.append(1-boxes[4])

                fig = plt.figure(figsize = (8, 8))

                ax = fig.add_subplot(111)
                ax.set_xlim(boxes[1]-2*boxes[3], boxes[1]+2*boxes[3])

    #             ax.set_ylim(1-3.5*np.median(total_flux_err), 1+3.5*np.median(total_flux_err))
                ax.scatter(total_time, total_flux, color = 'k', marker ='o', zorder = 1E6)
                detrended_lc = make_LightKurveObject(total_time, total_flux, total_flux_err)

                plot_lc_with_bboxes(detrended_lc, bboxes, ms=3, marker='.', lw=0, ax = ax)

                plt.show()


        t0_singles    = np.array(t0_singles)
        dur_singles   = np.array(dur_singles)
        depth_singles = np.array(depth_singles)


   

        new_T0_periodic = []
        planet_name = 0
        params_df = []

        for sss in range(len(t0_singles)):
#             print(sss)
            planet_name+=1

            if run_1:
                planet_df.loc[len(planet_df.index)] = [int(ticid), int(planet_name), np.inf, t0_singles[sss], dur_singles[sss], depth_singles[sss]]
#                 print('planet df singles', planet_df)



            else:

                new_params, conv, conv_attempt = pymc_new_general_function(np.array(total_time), total_flux, total_flux_err, t0_singles[sss], [dur_singles[sss], ab, depth_singles[sss]], 'Single')

                if len(new_params)>0:
                    new_params.loc[len(params_df)] = [None] * len(new_params.columns) 


                    params_df.append(new_params)


                    t0_, period_, depth_, tdur_, q = new_params.loc['t0', 'mean'], new_params.loc['Per', 'mean'], new_params.loc['depth', 'mean'], new_params.loc['dur', 'mean'], new_params.loc['SNR', 'mean']

#                     print('checking convergence 3')
                    if not np.isnan(t0_):
                        planet_df.loc[len(planet_df.index)] = [int(ticid), int(planet_name), np.inf, t0_, tdur_,depth_, q]

                    pd.DataFrame({'TICID':[con.TICID], 't0':[t0_], 'per':[period_], 'depth':[depth_], 'converged': [True], 'conv_on_run':[conv_attempt]}).to_csv('../checking_convergence_output/'+str(con.TICID)+'_'+str(round(t0_, 5))+'_Yconv_single.csv')

                else:
#                     print('checking convergence 4')

                    pd.DataFrame({'TICID':[con.TICID], 't0':[t0_singles[sss]], 'per':[np.nan], 'depth':[depth_singles[sss]], 'converged': [False], 'conv_on_run':[np.nan]}).to_csv('../checking_convergence_output/'+str(con.TICID)+'_'+str(round(t0_singles[sss], 5))+'_Nconv_single.csv')


#                 if not np.isnan(t0_):
#                     planet_df.loc[len(planet_df.index)] = [int(ticid), int(planet_name), np.inf, t0_, tdur_,depth_, q]
        if len(params_df)>0:
            params_df = pd.concat(params_df)
        print('params df singles', params_df)

    return planet_df, params_df





# In[55]:


#my mcmc functions


def lnprob(pars,flux,unc,time,cad,tc, u1, u2):
    t0 = pars[0]
    P = pars[1]
    rp_rs = pars[2]
    cosi = pars[3]
    a = pars[4]
    norm = pars[5]
    b = np.abs(a*cosi)
    #make sure all pars are good

    if np.abs(tc-t0)>0.25: return -np.inf
    if np.max([u1,u2,cosi,rp_rs]) > 1.: return -np.inf
    if np.min([u1,u2,cosi,rp_rs]) < 0.: return -np.inf
    if a < 1: return -np.inf
    if b > 1: return -np.inf

    flux_theo = predict_lc(time,t0,P,rp_rs,cosi,a,u1,u2,cad)
    flux=flux*norm
    unc=unc*norm
    result = 0.-0.5*np.sum(((flux_theo-flux)/unc)**2.)
    if np.isfinite(result) != True:
        #print('bad result')
        return -np.inf
    return result



def easy_data_phaseFold(indxs, *args):
    folded = [arg[indxs] for arg in args if args is not None]
    return np.array(folded)
    
def lnprob_MCMC_global(pars):
    flux,unc,time,cad,tc, u1, u2 = PARAMS
    t0 = pars[0]
    P = pars[1]
    rp_rs = pars[2]
    cosi = pars[3]
    a = pars[4]
    norm = pars[5]
    b = np.abs(a*cosi)
    #make sure all pars are good

    if np.abs(tc-t0)>0.25: return -np.inf
    if np.max([u1,u2,cosi,rp_rs]) > 1.: return -np.inf
    if np.min([u1,u2,cosi,rp_rs]) < 0.: return -np.inf
    if a < 1: return -np.inf
    if b > 1: return -np.inf

    # flux_theo = predict_lc(time,t0,P,rp_rs,cosi,a,u1,u2,cad)

    x = ((time - t0 + 0.5*P) % P) -( 0.5*P)

    flux_theo = predict_lc(x+t0,t0,P,rp_rs,cosi,a,u1,u2,cad)

    m = np.abs(x) < 0.5

    flux=flux*norm
    unc=unc*norm

    flux_theo_phase, flux_phase, unc_phase = easy_data_phaseFold(np.array(m), flux_theo, flux, unc)
    
    result = 0.-0.5*np.sum(((flux_theo_phase-flux_phase)/unc_phase)**2.)
    if np.isfinite(result) != True:
        #print('bad result')
        return -np.inf
    return result


def lnprob_MCMC_global_single(pars):
    flux,unc,time,per,cad,tc, u1, u2= PARAMS
    t0 = pars[0]
    rp = pars[1]
    cosi = pars[2]
    a = pars[3]
    norm = pars[4]
    b = np.abs(a*cosi)
    #make sure all pars are good

    if np.abs(tc-t0)>0.25: return -np.inf
#     if np.max(rp)>0.25: return -np.inf #attempting to set some upper bounds here just for the singles
    if np.max([u1,u2,cosi,rp]) > 1.: return -np.inf
    if np.min([u1,u2,cosi,rp]) < 0.: return -np.inf
    if a < 1: return -np.inf
    if b > 1: return -np.inf

    flux_theo = predict_lc(time,t0,per,rp,cosi,a,u1,u2,cad)
    flux=flux*norm
    unc=unc*norm
    result = 0.-0.5*np.sum(((flux_theo-flux)/unc)**2.)
    if np.isfinite(result) != True:
        #print('bad result')
        return -np.inf
    return result




def predict_lc(time_lc,t0,P,rp_rs,cosi,a,u1,u2,cad):
    oversample = 4
    e = 0.
    omega = np.pi/2.
    inc = np.arccos(cosi)*180./np.pi
    params = batman.TransitParams()
    params.t0  = t0
    params.per = P
    params.rp  = rp_rs
    params.a   = a
    params.inc = inc
    params.ecc = e
    params.w = omega*180./np.pi
    params.u = [u1,u2]
    params.limb_dark = "quadratic"
        
    if not cad>0:
        cad = 30.
    m = batman.TransitModel(params, time_lc ,supersample_factor = oversample, exp_time = cad/24./60.)

    flux_theo = m.light_curve(params)

    return flux_theo




# In[56]:
def extract_summary_dataframe(trace, hdi_prob=0.68):
    """
    Extracts a summary DataFrame from an ArviZ trace object with specified statistics.

    Parameters:
    - trace: ArviZ InferenceData object
    - hdi_prob: float, the probability for the HDI interval (default is 0.68)

    Returns:
    - pandas DataFrame with variables as index and columns: mean, median, sd, hdi_16%, hdi_84%, r_hat
    """
    # Get summary with specified HDI
    summary = az.summary(trace, hdi_prob=hdi_prob)

    # Compute median manually

    
    median_dataset = trace.posterior.median(dim=["chain", "draw"])
    medians = {var: float(median_dataset[var]) for var in median_dataset.data_vars}


    # Merge median into summary
    summary["median"] = medians

    # Reorder and select desired columns
    selected_columns = ['mean', 'median', 'sd', 'hdi_16%', 'hdi_84%', 'r_hat']
    custom_summary_df = summary[selected_columns]
    
#     print('custom summary', custom_summary_df)

    return custom_summary_df


def sample_until_converged(model, max_attempts=3, rhat_threshold=1.1, chains=4,cores=None, mp_context="spawn"):
    # Get all free random variables in the model
    
    cores = min(chains, os.cpu_count() or 1) if cores is None else cores

    free_vars = model.free_RVs
    if not free_vars:
        raise ValueError("No free random variables found for sampling.")
#     print('free vars', free_vars)
    # Use Metropolis for all free RVs
#     step = pm.Metropolis(vars=free_vars)

    step = pm.DEMetropolisZ(vars=free_vars)#, target_accept=0.8) 

    for attempt in range(3, max_attempts + 1):
        print(f"Sampling attempt {attempt-1}...")
        trace = pm.sample(step=step, draws=5000*attempt, tune=2000*attempt, chains=chains, cores = cores, 
            # use a safe, explicit multiprocessing context
            mp_ctx=mp.get_context(mp_context),
            # avoid identical RNG streams across chains
            random_seed=list(range(chains)),
)

        summary = az.summary(trace)
        if (summary['r_hat'] < rhat_threshold).all():
            print(f"Converged on attempt {attempt}")
            return trace, attempt
#         print('checking nans trace', trace.posterior['SNR'])
        print('checking nanas summary', az.summary(trace))
        print(f"Attempt {attempt-1} failed to converge.")

    raise RuntimeError("Model did not converge after multiple attempts.")


def min_relative_ess(idata, total_draws):
    """
    Minimum relative ESS across all posterior variables.
    ArviZ versions differ on `relative=` availability, so handle both.
    """
    try:
        ess_rel = az.ess(idata, method="bulk", relative=True)
        return float(ess_rel.to_array().min())
    except TypeError:
        ess_abs = az.ess(idata, method="bulk")
        return float(ess_abs.to_array().min()) / float(total_draws)




def sample_until_converged_smc(
    model,
    ess_ratio_target=0.70,     # target fraction of draws considered "effective"
    max_attempts=5,            # retries with increasing population size
    base_draws=2000,           # starting population per chain
    chains=4,
    cores=None,
    random_seed=None,          # int or list[int]; if None, will use a simple one-liner
    var_names=None,            # optional quick-look names to summarize
    progressbar=True,
):
    """
    Run SMC repeatedly, increasing draws until min relative ESS meets target.
    Returns (idata, attempt).
    """
    # Cores: one process per chain, capped at available CPUs
    cores = min(chains, os.cpu_count() or 1) if cores is None else cores

    # Seeds: keep it simple and predictable
    if random_seed is None:
        # One-liner Mallory suggested
        seeds = list(2026 + np.array(range(chains)))
    elif isinstance(random_seed, int):
        # Same base int per chain, still simple and reproducible
        seeds = [int(random_seed)] * chains
    else:
        seeds = list(random_seed)  # assume list[int]
        if len(seeds) != chains:
            raise ValueError(f"random_seed list length {len(seeds)} != chains {chains}")

    with model:
        # Names of free RVs once

        for attempt in range(1, max_attempts + 1):
            draws = int(base_draws * attempt)
            print(f"[SMC attempt {attempt}] draws={draws}, chains={chains}, cores={cores}")

            idata = pm.sample_smc(
                draws=draws,
                chains=chains,
                cores=cores,
                random_seed=seeds,              # your per-chain seeds
                threshold=ess_ratio_target,     # ESS fraction for resampling
                return_inferencedata=True,
                progressbar=progressbar,
                compute_convergence_checks=False,
            )

            total_draws = draws * chains

            # 4) Your ESS metric (uses your existing function)
            min_ratio = min_relative_ess(idata, total_draws)
            print(f"  min relative ESS ≈ {min_ratio:.3f}")

            # Optional quick-look summaries
            if var_names:
                try:
                    print(az.summary(idata, var_names=var_names))
                except Exception:
                    pass
            try:
                print(az.summary(idata, var_names=["Per", "rp_rs", "a_rs", "b", "t0"]))
            except Exception:
                pass

            if min_ratio >= ess_ratio_target:
                print(f"Converged by ESS on attempt {attempt}")
                return idata, attempt

    print("Returning last attempt; target ESS not met")
    return idata, attempt

                                


class BatmanOp(Op):
    itypes = [pt.dvector, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar]
    otypes = [pt.dvector]

    def make_node(self, *inputs):
        # Convert all inputs to tensors if they aren't already
        converted_inputs = [pt.as_tensor_variable(inp) for inp in inputs]
        return Apply(self, converted_inputs, [o() for o in self.otypes])

    def perform(self, node, inputs, outputs):
        time, t0, per, rp_rs, a_rs, inc, u1, u2, ecc, cad = inputs
        params = batman.TransitParams()

        params.t0 = float(t0)
        params.per = float(per)
        params.rp = float(rp_rs)
        params.a = float(a_rs)
        params.inc = float(inc)
        params.u = [float(u1), float(u2)]
        params.ecc = float(ecc)
        params.w = 90.0
        params.u = [float(u1), float(u2)]
        params.limb_dark = "quadratic"
        m = batman.TransitModel(params, time, supersample_factor=4, exp_time=cad/24./60.)
        outputs[0][0] = m.light_curve(params)

    def grad(self, inputs, g_outputs):
        # For now, return zeros (no gradient)
        return [pt.zeros(inp.shape, dtype=pt_config.floatX) for inp in inputs]
    
def set_up_variables_for_pymc_fit(time, flux, unc, t0, other_pars, type_fn):
    """
    Returns:
        use_time, use_flux, use_unc, per, u1, u2, depth, cad

    cad is a robust effective cadence in MINUTES.
    """
    mask = np.logical_or(np.logical_or(np.isnan(flux),  np.isnan(time)),  np.isnan(unc))
    time = time[~mask]
    unc  = unc[~mask]
    flux = flux[~mask]

    cad = np.nanpercentile(np.clip(np.diff(np.unique(time))*60.*24., 200/60, 30), 95) #minutes

    if type_fn == 'Single':
        dur, ab, depth = other_pars
        k = np.sqrt(depth)
        per1 = np.max(time)-np.min(time)
        per2 = ((3*np.pi/con.G/con.rho_star)** 0.5 ) * (dur/np.pi/(1+k)) ** 1.5
        per = np.max([per1, per2])
        print('time difference', np.max(time)-np.min(time), 'checking duration units (must be < 0.5 for days, <12 for hours)', dur)
        if per<10:
            per = 27.8
        
        indxes = np.where(np.abs(time-t0)<(1.+dur))
        
        use_time = np.array(time)[indxes]
        use_flux = np.array(flux)[indxes]
        use_unc  = np.array(unc)[indxes]
    elif type_fn == 'Periodic':
        per, ab, depth = other_pars
        use_time = np.array(time)
        use_flux = np.array(flux)
        use_unc  = np.array(unc)
        
        print(per, type(per))
        
        a_smaj_guess = float((con.G * con.rho_star * (per ** 2) / 3 / np.pi) ** (1/3))
        dur =  min([0.5/3, (float(per) / float(a_smaj_guess * np.pi))])  # days

    u1, u2 = ab

    return use_time, use_flux, use_unc, float(per), u1, u2, float(depth), 1.5*float(dur), cad


def median_pytensor(x):
    sorted_x = pt.sort(x)
    n = x.shape[0]
    mid = n // 2
    return pt.switch(
        pt.eq(n % 2, 0),
        (sorted_x[mid - 1] + sorted_x[mid]) / 2.0,
        sorted_x[mid]
    )



def make_windows_from_time_stamps(t, gap_threshold=0.5):
    """
    Convert sorted time stamps (days) into contiguous [start, end] windows.
    gap_threshold is the minimum gap that splits windows (days).
    Pick a value larger than cadence and smaller than any real gap.
    """
    t = np.asarray(t)
    t = t[np.isfinite(t)]
    t = np.sort(t)
    if t.size == 0:
        return np.empty((0, 2))
    gaps = np.diff(t)
    breaks = np.where(gaps > gap_threshold)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends   = np.concatenate((breaks, [t.size - 1]))
    return np.column_stack((t[starts], t[ends]))





def pymc_new_general_function(time, flux, unc, T0, other_pars, type_fn,
                              verbose=True, keep_ld_fixed=True):

    # Unpack and precompute using your existing helper
    time, flux, unc, Per, u1, u2, Depth, pTdur, cad = set_up_variables_for_pymc_fit(
        time, flux, unc, T0, other_pars, type_fn
    )
    
    batman_op = BatmanOp()

    # --- Tightening bounds and simple physics helpers (outside the model) ---

    # Build contiguous observation windows from actual timestamps
    windows = make_windows_from_time_stamps(np.array(time), gap_threshold=0.5)

    # Count how many integer k place transit centers inside these windows
    nobs_est = 0
    for s, e in windows:
        k_low  = np.ceil((s - T0) / Per)
        k_high = np.floor((e - T0) / Per)
        nobs_est += int(max(0, k_high - k_low + 1))

    ecc = 0.

    with pm.Model() as model:

        # Use a duration-sized box for t0 in both modes
        t0 = pm.Uniform("t0", lower=T0 - pTdur, upper=T0 + pTdur)

        if type_fn == 'Single':
            print('single')
            # Keep your Single logic, but give Per a modest bound window
#             per   = pm.Uniform( "Per",  lower=max(0.25, Per * 0.5), upper=Per * 2.5)
#             ecc   = pm.TruncatedNormal("Eccen", mu=0., sigma=0.25, lower=0, upper=1)
            # a/R*: tie to sampled per
            a_rs_from_Per_init = float(((con.G * con.rho_star * (Per ** 2)) / (3.0 * np.pi)) ** (1.0 / 3.0))
#             a_rs_mu = pm.Deterministic("a_rs_mu", (con.G * con.rho_star * (per ** 2) / 3 /np.pi) ** (1/3))
#             a_rs    = pm.TruncatedNormal("a_rs", mu=a_rs_mu, sigma=5.0, lower=1.0)

            a_rs = pm.TruncatedNormal("a_rs", mu=a_rs_from_Per_init, sigma=5.0, lower=1.0,
                              initval=a_rs_from_Per_init)

            per = pm.Deterministic("Per", pt.sqrt((3.0 * np.pi) / (con.G * con.rho_star)) * a_rs ** 1.5)

            fold_this = False
            
        elif type_fn == 'Periodic':
            print('periodic, period =', Per, 'days, transits observed = ', nobs_est)

            # Period bounds:
            #  - many transits: very tight uniform (±1%)
            #  - otherwise: TN with ±20% support and sigma ~ 5% of Per

            if nobs_est > 3:
                P_lower = max(0.25, Per * 0.99)
                P_upper = Per * 1.01
                P_sigma = None
                fold_this = True
                # MANY TRANSITS MODE (DECOUPLED)
                # 1. Keep tight uniform for ephemeris
                per = pm.Uniform("Per", lower=P_lower, upper=P_upper)

                # 2. DO NOT tie a_rs to Per via Kepler’s law
                #    Instead give an uninformative wide prior
                a_rs = pm.Uniform("a_rs", lower=1.0, upper=300.0)

            else:
                P_lower = max(0.25, Per * 0.80)
                P_upper = Per * 1.20
                P_sigma = max(0.1, 0.05 * Per)

                # LOW/FEW TRANSITS MODE (original behavior)
                per = pm.TruncatedNormal("Per", mu=Per, sigma=P_sigma, lower=P_lower, upper=P_upper)
                # original line
                
#                 print('con rho_star', con.rho_star, type(con.rho_star))
                fold_this = False
                a_rs_mu = pm.Deterministic("a_rs_mu", (con.G * con.rho_star * (per ** 2) / 3 /np.pi) ** (1/3))
                a_rs = pm.TruncatedNormal("a_rs", mu=a_rs_mu, sigma=3., lower=1.0)
       
        # Depth and geometry
        rp_rs = pm.TruncatedNormal("rp_rs", mu=pt.sqrt(Depth),
                                   sigma=pt.maximum(0.02, 0.5 * pt.sqrt(Depth)),
                                   lower=0, upper=1)
        b     = pm.TruncatedNormal('b', mu=0, sigma=0.01, lower=0, upper=1)

        depth  = pm.Deterministic('depth', rp_rs**2)

        # 1) Inclination: clip to arccos domain
        cosi = pm.Deterministic("cosi", pt.clip(b / a_rs, -1.0 + 1e-12, 1.0 - 1e-12))
        inc  = pm.Deterministic("inclination", pt.arccos(cosi) * 180.0 / np.pi)

        # 2) Duration terms: guard tiny denominators / underflow
        eps  = 1e-12
        root = pt.sqrt(pt.clip(1.0 - b**2, 1e-12, 1.0))
        T_dur0 = per / ((a_rs + eps) * np.pi)
        tau    = pm.Deterministic('tau', rp_rs * T_dur0 / root)
        dur    = pm.Deterministic('dur', root * T_dur0 + tau)
        win    = pm.Deterministic('win', dur * 2.)



        # Masks
        if type_fn == 'Periodic':
            intran_mask = transit_mask_tensors(time, per, dur, t0)  # boolean PyTensor
        elif type_fn == 'Single':
            intran_mask = pt.abs(time - t0) < (dur / 2.)

        outran_mask = pt.invert(intran_mask)

        # Out-of-transit scatter estimate
        out_flux = flux * outran_mask  # zeros elsewhere
        count = pt.maximum(pt.sum(outran_mask), 1)
        mean_out = pt.sum(out_flux) / count
        std_out = pt.sqrt(pt.sum(outran_mask * (flux - mean_out)**2) / count)

        N_tran = pt.sum(intran_mask)
        uq = pt.ones_like(flux) * std_out
        sigs = pt.switch(N_tran > 0, pt.mean(pt.where(intran_mask, uq, 0)), 1e6)

        # SNR diagnostic
        print('N_intran', pm.draw(N_tran), 'depth', pm.draw(depth), 'sig', pm.draw(sigs))
        SNR_val = pt.switch(pt.gt(N_tran, 0), pt.sqrt(N_tran) * depth / sigs, 0)
        SNR_clipped = pt.clip(SNR_val, 0, 1e4)
        SNR_final = pt.where(pt.eq(SNR_clipped, 1e4), 1, SNR_clipped)
        if not fold_this:
            SNR = pm.Deterministic("SNR", SNR_final)

        norm = pm.Deterministic("norm", median_pytensor(out_flux))

        # Likelihood
        if fold_this:
            folded_phase = ((time - T0 + 0.5 * Per) % Per) - (0.5 * Per)
            
            sort_indx = np.argsort(folded_phase)
            
            phase = folded_phase[sort_indx]
            use_index = np.abs(phase) < min([0.5, 3*pTdur])
            
            dt_minutes_min = np.nanpercentile(np.diff(np.unique(np.sort(time))), 5) * 24.0 * 60.0
            p_cad = float(np.clip(dt_minutes_min, 0.2, 60.0))

            p_flux_model = batman_op(phase[use_index] + T0, t0, per, rp_rs, a_rs, inc, u1, u2, ecc, p_cad)

            intran_mask = transit_mask_tensors(phase + t0, per, dur, t0)
            std_out = pt.sqrt(pt.sum(outran_mask * (flux - mean_out)**2) / count)

            N_tran = pt.sum(intran_mask)
            uq = pt.ones_like(flux) * std_out
            sigs = pt.switch(N_tran > 0, pt.mean(pt.where(intran_mask, uq, 0)), 1e6)
            print('folded: N_intran', pm.draw(N_tran), 'depth', pm.draw(depth), 'sig', pm.draw(sigs))

            SNR_val = pt.switch(pt.gt(N_tran, 0), pt.sqrt(N_tran) * depth / sigs, 0)
            SNR_clipped = pt.clip(SNR_val, 0, 1e4)
            SNR_final = pt.where(pt.eq(SNR_clipped, 1e4), 1, SNR_clipped)
            SNR = pm.Deterministic("SNR", SNR_final)

            pm.Normal("obs", mu=p_flux_model* norm, sigma=unc[sort_indx][use_index], observed=flux[sort_indx][use_index])
        else:
            flux_model = batman_op(time, t0, per, rp_rs, a_rs, inc, u1, u2, ecc, cad)
            pm.Normal("obs", mu=flux_model * norm, sigma=unc, observed=flux)

    # Sampling (use your SMC wrapper; no custom start needed)
    with model:
        try:
            trace, conv_attempt = sample_until_converged(model)
            summary = extract_summary_dataframe(trace)
        except RuntimeError as error:
            return (
                pd.DataFrame(columns=['mean', 'median', 'sd', 'hdi_16%', 'hdi_84%', 'r_hat']),
                False,
                np.nan
            )

    if verbose:
        az.plot_trace(trace)
        az.plot_posterior(trace)
        plt.show()

#     print('summary', summary)
    return summary, True, conv_attempt


def flatten_summary_blocks(F):
    """
    Transforms a summary DataFrame F with np.nan separator rows into a flattened format.
    Each block of rows (separated by NaN rows) becomes a single row in the output,
    with columns named as 'variable_stat'.

    Parameters:
    - F: pandas DataFrame with summary blocks separated by rows of NaNs

    Returns:
    - pandas DataFrame with one row per block and flattened columns
    """
    blocks = []
    current_block = []

    for idx, row in F.iterrows():
        if row.isnull().all():
            if current_block:
                blocks.append(pd.DataFrame(current_block, columns=F.columns, index=[r.name for r in current_block]))
                current_block = []
        else:
            current_block.append(row)

    # Add the last block if it exists
    if current_block:
        blocks.append(pd.DataFrame(current_block, columns=F.columns, index=[r.name for r in current_block]))

    # Flatten each block into a single row
    flattened_rows = []
    for block in blocks:
        flat_row = {}
        for var_name, row in block.iterrows():
            for col in block.columns:
                flat_row[f"{var_name}_{col}"] = row[col]
        flattened_rows.append(flat_row)

    # Create the final DataFrame
    final_df = pd.DataFrame(flattened_rows)

    return final_df
    

# def sort_arrays_by_time(total_time, *args):
# #     for arg in args: 
# #         print(type(arg))
# #         print('arg', len(arg))
# #         print(args[:10])
#     print('len total time', len(total_time), 'len(arguments)', [len(arg) for arg in args])
#     return [np.array(arg)[np.argsort(total_time)] for arg in args] 

def sort_arrays_by_time(total_time, *args):
    total_time = np.asarray(total_time)

    # Skip sorting if already monotonic
    if np.all(np.diff(total_time) >= 0):
        return [np.asarray(arg) for arg in args]

    idx = np.argsort(total_time)
    return [np.asarray(arg)[idx] for arg in args]

def sort_arrays_by_index(index_lst, *args):
    return [[arg[index] for index in index_lst] for arg in args]


# def bin_by_time_many_args(time,time_size_of_bins, **params):
#     time = np.array(time).byteswap().newbyteorder() 

#     interval = time_size_of_bins/60./24.    
    
#     dict_params = {k:np.array(v).byteswap().newbyteorder() for k,v in  params.items()}
#     dict_params['time'] = time        

#     df = pd.DataFrame(dict_params, dtype=object)
#     df = pd.concat([df, df])
#     numbins = np.array(list(range(int(np.ceil((max(time)-min(time))/interval))+1)))
#     bins = np.array([min(time)+x*interval for x in numbins])
#     df['time_bins'] = pd.cut(df.time, bins)
#     new_time = [x for x in df.groupby('time_bins').mean()['time'] if not math.isnan(x)]

#     new_dict = {'time': new_time}
#     for key, value in params.items():
#         new_arg = [x for x in df.groupby('time_bins').mean()[key] if not math.isnan(x)]
#         new_dict[key] = new_arg
# #     print('checking dictionary ', new_dict)
#     return new_time, new_dict

def bin_by_time_many_args(time, time_size_of_bins, **params):
    time = np.asarray(time)

    interval = time_size_of_bins / 60. / 24.  # days

    # Precompute bins
    min_t, max_t = np.min(time), np.max(time)
    nbins = int(np.ceil((max_t - min_t) / interval)) + 1
    bins = min_t + np.arange(nbins + 1) * interval

    # Assign each time point to a bin
    bin_idx = np.digitize(time, bins) - 1

    # Prepare output
    new_dict = {}
    new_time = []

    for b in range(nbins):
        mask = bin_idx == b
        if not np.any(mask):
            continue

        new_time.append(np.mean(time[mask]))

        for key, arr in params.items():
            arr = np.asarray(arr)
            if key not in new_dict:
                new_dict[key] = []
            new_dict[key].append(np.mean(arr[mask]))

    return np.array(new_time), new_dict


def bin_data_with_diff_cadences_many_args(total_time, min_cad = 0, **params):

    # --- NEW: sort once here, not repeatedly ---
    idx = np.argsort(total_time)
    time = np.asarray(total_time)[idx]
    for key, value in params.items():
        params[key] = np.asarray(value)[idx]
    # -------------------------------------------

    new_time = []
    
    indexes_split_unorganize = breaking_up_data(time, 1.)   
    indexes_split = sorted(indexes_split_unorganize, key=lambda x: len(x), reverse=True)
    
    cadences = [min_cad]
    for indx in indexes_split:
        split_time = time[indx]
        med_cadence = np.nanmin(np.diff(split_time))
        cadences.append(med_cadence)
        
    max_cadence = np.nanmax(cadences)
    
    dict_lst = []
    for indx in indexes_split:
        filter_dict = {k:np.array(v)[indx] for k,v in  params.items()}
        cad = np.nanmin(np.diff(time[indx]))

        if np.ceil(cad*60*24) < np.ceil(max_cadence*60*24):
            binned_time, all_params_dict = bin_by_time_many_args(time[indx], max_cadence*60*24, **filter_dict)
            new_time.extend(binned_time)
            dict_lst.append(all_params_dict)
        else:
            new_time.extend(time[indx])
            dict_lst.append(filter_dict)

    binned_dict = {}
    binned_dict['time'] = new_time

    for k in dict_lst[0].keys():
        if k != 'time':
            binned_dict[k] = np.concatenate([d[k] for d in dict_lst])

    # --- sorting AFTER binning is still correct ---
    results = sort_arrays_by_time(np.array(binned_dict['time']), *binned_dict.values())
    return results


def creating_broken_axes_plots_for_DV_report_min_plot(time, flux, err, binned_time, binned_flux, binned_err, gs=False, subplot_val = None, ratios = []):
    

    
    if len(ratios)==0:
        diff_time_arrays = np.array([max(x)-min(x) for x in split_times])
        min_diff_time_arrays = min(diff_time_arrays)
        ratios = diff_time_arrays/min_diff_time_arrays

    if gs == False : 
        fig, axes = plt.subplots(1, len(ratios), figsize = [50, 10], sharey=True, 
                             gridspec_kw={'width_ratios': ratios})

    if gs != False:
        axes = [plt.subplot(gs[subplot_val, x]) for x in range(len(ratios))]
        
        
    indexes_split = breaking_up_data(time)   
    binned_indexes_split = breaking_up_data(binned_time)   

    split_times, split_fluxes, split_err = sort_arrays_by_index(indexes_split, time, flux, err)
    binned_split_times, binned_split_fluxes, binned_split_err = sort_arrays_by_index(binned_indexes_split, binned_time, binned_flux, binned_err)


    
    d = .03 # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=axes[0].transAxes, color='k', clip_on=False)

#     for lll in split_times:
#         print(max(lll))
#         print(min(lll))
    
    for iii in range(len(axes)):

        axes[iii].set_xlim(min(split_times[iii])-1.5, max(split_times[iii])+1.5)
        axes[iii].scatter(split_times[iii], split_fluxes[iii], color = 'lightgrey', s = 5, zorder = 0)
        axes[iii].scatter(binned_split_times[iii], binned_split_fluxes[iii], color = 'k', s = 5, zorder = 1)
        axes[iii].tick_params(labelsize = 7)

        
    for jjj in range(len(axes)-1):
        axes[jjj].spines['right'].set_visible(False)
        
        axes[jjj].plot((1-d/ratios[jjj],1+d/ratios[jjj]),(-d,+d), **kwargs) # top-right diagonal
        axes[jjj].plot((1-d/ratios[jjj],1+d/ratios[jjj]),(1-d,1+d), **kwargs) # bottom-right diagonalaxes[1].plot((1-d,1+d), (-d,+d), **kwargs)

        kwargs.update(transform=axes[jjj+1].transAxes) # switch to the bottom axes

        axes[jjj+1].plot((-d/ratios[jjj+1],d/ratios[jjj+1]),(-d,+d), **kwargs) # top-right diagonal
        axes[jjj+1].plot((-d/ratios[jjj+1],d/ratios[jjj+1]),(1-d,1+d), **kwargs) # bottom-right diagonalaxes[1].plot((1-d,1+d), (-d,+d), **kwargs)

        axes[jjj+1].spines['left'].set_visible(False)
        axes[jjj+1].yaxis.set_ticks_position('none')
        axes[jjj+1].tick_params(axis = 'y', color = 'none', labelcolor = 'none')
    axes[0].tick_params(labelsize = 7)
    axes[0].set_ylabel('Relative Flux', fontsize = 9,rotation='vertical',)
    axes[int(np.floor((len(axes)-1)/2))].set_xlabel('Time (BJD - 2457000)', fontsize = 9)


#     plt.text(0.441, posit[1], 'Time (BJD - 2457000)', va='center', fontsize = 14,  fontweight = 'normal')

#     plt.text(0.09, 0.5, 'Relative Flux', va='center', fontsize = 15,  rotation='vertical', fontweight = 'normal')
#     plt.savefig('evil')
    return axes


# In[57]:




def find_t0_vals_within_time(min_t, max_t, t0, period):
    # Written by Mallory Harris

    # Description: for multi-planet system - uses times of transits previously found to mask when searching for additional planets

    # Arguments : time    = array of time values
    #             t0      = mid-tranist time of previously found planets
    #             Tdur    = transit duration of previously found planets
    #             period  = period of previously found planets

    # Return    : epoch_durations  = list of tuples with start and end times for each tranist in observing window

    num_per_before = 0-int(np.floor(np.abs(t0-min_t)/period))
    num_per_after = int(np.floor(np.abs(t0-max_t)/period))
    
#     print('period range', num_per_before-1, num_per_after+2)
    epochs = t0 + period*np.array(range(num_per_before-1, num_per_after+2))
    epochs = epochs[(epochs<max_t) & (epochs>min_t)]
#     print('should be sorted times', min_t, epochs, max_t)

    return epochs


#All constants that presist throughout the functions are defined in ALL CAPS
NUM_WALKERS = 20
NCHAIN = 5000
BCHAIN = 4000
CADENCES = [20/60, 2., 200/60,  10., 30.]
CAD_PS = 2.
CAD_FFI = 30.


def creating_first_DV_report_page(ticid, data_filename, planet_df, catalog_df, intransit=[], APER=False, eleanor=False, **other_pipelines):

    with PdfPages('../DV_reports/'+os.path.dirname(data_filename).split('/')[-1][:-6]+'.pdf') as pdf: #+str(planet_num)+'.pdf') as pdf:
        print('working on ' + '../DV_reports/'+os.path.dirname(data_filename).split('/')[-1][:-6]+'.pdf')#')#+str(planet_num)+'.pdf')

        df = pd.read_csv(data_filename)
        time, flux, err, trend, raw, raw_err = [np.array(df[col]) for col in ['TIME', 'FLUX', 'FLUX_ERR', 'FLUX_TREND', 'RAW_FLUX', 'RAW_FLUX_ERR']]
        
        print('checking lengths', [len(x) for x in [time, flux, err, trend, raw, raw_err]])
        
        if len(err) != len(flux):
            err = np.full(len(flux), np.std(flux))
            print('Error')
        
        binned_time, binned_flux, binned_err,  binned_trend, binned_raw, binned_rerr = bin_data_with_diff_cadences_many_args(time, flux = flux, err = err, trend = trend, raw = raw, raw_err = raw_err)

        indexes_split = breaking_up_data(time)   

        binned_indexes_split = breaking_up_data(binned_time)   

        split_times, split_fluxes, split_err, split_raw, split_rerr = sort_arrays_by_index(indexes_split, time, flux,
                                                                                           err, raw, raw_err)
        
        binned_split_times, binned_split_fluxes, binned_split_err, binned_split_raw, binned_split_rerr = sort_arrays_by_index(binned_indexes_split, binned_time, binned_flux, binned_err, binned_raw, binned_rerr)

        diff_time_arrays = np.array([max(x)-min(x) for x in split_times])
        print('length of different time arrays: ', diff_time_arrays)
        min_diff_time_arrays = min(diff_time_arrays)
        ratios = diff_time_arrays/min_diff_time_arrays

        num_plots = 3
        if eleanor:
            num_plots+=1
        if APER:
            num_plots+=1
            
        if len(other_pipelines)>0:
            num_plots+=len(other_pipelines)
        
        
        fig0 = plt.figure(figsize=(8.5, 11),constrained_layout=True,dpi=100)
        gs = fig0.add_gridspec(1,2,width_ratios=[4.25, 1], wspace = 0.1) #create grid for subplots - makes it easier to assign where each plot goes
        
        
        gs0 = gs[0].subgridspec(7, len(split_times), wspace=0.02, width_ratios = ratios)
        gs1 = gs[1].subgridspec(1, 1)   
        
        ymin = np.nanmin([np.percentile(raw,1),np.percentile(flux, 0.5),1.-(max(planet_df['Depth']))]) #define y-axis limits by percentages to avoid using outliers 
        ymax = np.nanmax([np.percentile(raw,99.25),np.percentile(flux,99.5)])
        delta_y = np.abs(ymax-ymin)
        ymin = ymin-(delta_y*.05) #make sure ymin allows for all data        

        subplot = 0
        
        axes1 = creating_broken_axes_plots_for_DV_report_min_plot(
            time, raw, raw_err,
            binned_time, binned_raw, binned_rerr,
            gs0, subplot, ratios)
        axes = axes1

        for ax in axes1:
            min_vals, max_vals = ax.get_xlim()
            new_indexes = [(time>min_vals) & (time<max_vals)][0]
            ax.plot(time[new_indexes], trend[new_indexes], color = 'r', lw = 1, zorder = 10)
            ax.set_ylim(ymin, ymax)

        subplot+=1
        
        
        ymin2 = np.nanmin([np.percentile(flux, 0.25)])*0.95#,1.-(max(planet_df['Depth']))]) #define y-axis limits by percentages to avoid using es 
        ymax2 = np.percentile(flux,99.5)*1.05
        delta_y2 = np.abs(ymax2-ymin2)
        ymin2 = ymin2-(delta_y2*.05) #make sure ymin allows for all data        

        per_planets_df = planet_df[planet_df['Ptype']=='Period']

        if len(per_planets_df)>0:
            

            axes2 = creating_broken_axes_plots_for_DV_report_min_plot(
                time, flux, err,
                binned_time, binned_flux, binned_err,
                gs0, subplot, ratios)

            for ax in axes2:
                try:
                    ax.set_ylim(ymin2, ymax2)
                except Exception as e:
                    print(e)
                    print(f'ymin: {ymin2}, ymax: {ymax2}, Error')
                min_vals, max_vals = ax.get_xlim()
                split_time = time[(time>min_vals) & (time<max_vals)]
                cad = np.min(np.diff(split_time))

                for indx, planet in per_planets_df.iterrows():
                    model_time = np.arange(min_vals,max_vals,cad) #creates a uniformly spaced around spanning the length of time measurements taken in 30 minute intervals
                    model_flux = predict_lc(model_time, planet.T0, planet.Period, planet.Rad_p, planet.Cosi, planet.Semi_Maj, planet.u1, planet.u2, cad)#*planet.Norm #create a model of the transit
                    model_flux = model_flux/np.nanmedian(model_flux)
                    ax.plot(model_time, model_flux, color = 'C'+str(indx), lw = 2, alpha = 0.7, zorder = 1E3)
         
            axes = axes+axes2
        subplot+=1

        times_ot, fluxes_ot, err_ot = time[~intransit], flux[~intransit],err[~intransit]
        
        

        binned_time_ot, binned_flux_ot, binned_err_ot = bin_data_with_diff_cadences_many_args(times_ot, flux = fluxes_ot, err = err_ot)
        
        
        single_planet_df = planet_df[planet_df['Ptype']=='Single'].reset_index(drop = True)

        if len(single_planet_df)>0:

            axes3 = creating_broken_axes_plots_for_DV_report_min_plot(
                times_ot, fluxes_ot, err_ot,
                binned_time_ot, binned_flux_ot, binned_err_ot,
                gs0, subplot, ratios
            )
            for ax in axes3:
                min_vals, max_vals = ax.get_xlim()
                split_time = times_ot[(times_ot>min_vals) & (times_ot<max_vals)]
                cad = np.min(np.diff(split_time))

                for indx, planet in single_planet_df.iterrows():
#                     bboxes = DT_analysis(split_time, fluxes_ot[(times_ot>min_vals) & (times_ot<max_vals)], err_ot[(times_ot>min_vals) & (times_ot<max_vals)], confidence = 0.65)
#                     detrended_lc = make_LightKurveObject(times_ot, fluxes_ot, err_ot)
#                 #         print(detrended_lc)
#                     plot_lc_with_bboxes(detrended_lc, bboxes, ms=3, marker='.', lw=0, ax = ax)

                    model_time = np.arange(min_vals, max_vals,cad) 
                    model_flux = predict_lc(model_time, planet.T0, planet.Period, planet.Rad_p, planet.Cosi,planet.Semi_Maj, planet.u1, planet.u2, cad)#*planet.Norm #create a model of the transit
                    
                    model_flux = model_flux/np.nanmedian(model_flux)

                    ax.plot(model_time, model_flux, color = 'C'+str(indx+len(per_planets_df)), lw = 2, alpha = 0.7, zorder = 10)
                    ax.set_ylim(ymin2, ymax2)

        
            axes = axes+axes3
            
            
        subplot +=1

#         if APER:
#             subplot+=1
#             time, flux, err = pd.read_csv(glob.glob(outdir+'*APER*.csv'))
            
#             binned_time, binned_flux, binned_err = bin_data_with_diff_cadences_many_args( time, flux = flux, err = err)
# #             print('binned_times', binned_time)
            
#             indexes_split = breaking_up_data(time)   
#             binned_indexes_split = breaking_up_data(binned_time)   <ma
    
#             split_times, split_fluxes, split_err, split_raw, split_rerr = sort_arrays_by_index(indexes_split, time, flux, err, raw, raw_err)
#             binned_split_times, binned_split_fluxes, binned_split_err, binned_split_raw, binned_split_rerr = sort_arrays_by_index(binned_indexes_split, binned_time, binned_flux, binned_err)
#             axes_n = creating_broken_axes_plots_for_DV_report_min_plot(time, flux, err,binned_time, binned_flux, binned_err, gs0, subplot, ratios)
#             for ax in axes_n:
#                 ax.set_ylim(ymin2, ymax2)
#                 min_vals, max_vals = ax.get_xlim()
#                 split_time = times_ot[(times_ot>min_vals) & (times_ot<max_vals)]
#                 cad = np.min(np.diff(split_time))
#             axes = axes+axes_n

#         if eleanor:
#             subplot+=1

#             time, flux, err = pd.read_csv(glob.glob(outdir+'*eleanor*.csv'))
            
#             binned_time, binned_flux, binned_err = bin_data_with_diff_cadences_many_args( time, flux = flux, err = err)
# #             print('binned_time', binned_time)
            
#             indexes_split = breaking_up_data(time)   
#             binned_indexes_split = breaking_up_data(binned_time)   
    
#             split_times, split_fluxes, split_err, split_raw, split_rerr = sort_arrays_by_index(indexes_split, time, flux, err, raw, raw_err)
#             binned_split_times, binned_split_fluxes, binned_split_err, binned_split_raw, binned_split_rerr = sort_arrays_by_index(binned_indexes_split, binned_time, binned_flux, binned_err)
#             axes_n = creating_broken_axes_plots_for_DV_report_min_plot(time, flux, err,binned_time, binned_flux, binned_err, gs0, subplot, ratios)
#             for ax in axes_n:
#                 ax.set_ylim(ymin2, ymax2)
#                 min_vals, max_vals = ax.get_xlim()
#                 split_time = times_ot[(times_ot>min_vals) & (times_ot<max_vals)]
#                 cad = np.min(np.diff(split_time))
#             axes = axes+axes_n

#         if len(other_pipelines)>0:
#             for pip in other_pipelines:
#                 subplot+=1
#                 time, flux, err = pd.read_csv(glob.glob(outdir+'*'+pip+'*.csv'))
            
#                 binned_time, binned_flux, binned_err = bin_data_with_diff_cadences_many_args( time, flux = flux, err = err)
# #                 print('binned_time', binned_time)

#                 indexes_split = breaking_up_data(time)   
#                 binned_indexes_split = breaking_up_data(binned_time)   

#                 split_times, split_fluxes, split_err, split_raw, split_rerr = sort_arrays_by_index(indexes_split, time, flux, err, raw, raw_err)
#                 binned_split_times, binned_split_fluxes, binned_split_err, binned_split_raw, binned_split_rerr = sort_arrays_by_index(binned_indexes_split, binned_time, binned_flux, binned_err)
#                 axes_n = creating_broken_axes_plots_for_DV_report_min_plot(time, flux, err,binned_time, binned_flux, binned_err, gs0, subplot, ratios)
#                 for ax in axes_n:
#                     ax.set_ylim(ymin2, ymax2)
#                     min_vals, max_vals = ax.get_xlim()
#                     split_time = times_ot[(times_ot>min_vals) & (times_ot<max_vals)]
#                     cad = np.min(np.diff(split_time))
#                 axes = axes+axes_n
    

            

        for ax in axes:
            min_vals, max_vals = ax.get_xlim()
            ymin_, ymax_ = ax.get_ylim()
            delta_y = abs(ymax_ - ymin_)
            for indx, planet in planet_df.iterrows():
                epochs = find_t0_vals_within_time(min_vals, max_vals, planet['T0'], planet['Period'])
#                 print('planet num', planet['Planet_Num'])
                if planet['Ptype']=='Single':
                    ax.scatter(epochs, np.full(len(epochs) ,ymin_ + 0.1*delta_y), marker='^', color = 'C'+str(indx), s=50, zorder = 1000)
                else:
                    ax.scatter(epochs, np.full(len(epochs) ,ymin_ + 0.05*delta_y), marker='^', color = 'C'+str(indx), facecolors='none', s=30, zorder = 5000)

                    
                    
                    
        ax_fin = plt.subplot(gs1[:,-1]) #for the last subplot, print text
        txtstr = 'TICID='+ str(ticid)                              +'\n'\
            +'RA='   + str(round(float(catalog_df.RA), 8))                   +'\n'\
            +'DEC='  + str(round(float(catalog_df.DEC), 8))            +'\n'\
            +'R_*='  + str(round(float(catalog_df.Rad), 5))         +'[R_s]'   +'\n'\
            +'M_*='  + str(round(float(catalog_df.Mass), 5))        +'[M_s]'   +'\n'\
            +'Teff=' + str(round(float(catalog_df.Teff), 2))     +'[K]'     +'\n'\
            +'Tmag=' + str(round(float(catalog_df.Tmag), 3))                   +'\n'\
            +'Vmag=' + str(round(float(catalog_df.Vmag), 3))                   +'\n'\
            +'Jmag=' + str(round(float(catalog_df.Jmag), 3))                   +'\n'\
            +'Cont=' + str(round(float(catalog_df.ContRatio), 3))                   +'\n'\
            +'----- Planet Parmas -----'                              +'\n'\
        
        for indx, planet in planet_df.iterrows():
            txtstr = txtstr + '--' +'Planet Num='+ str(int(planet.Planet_Num))+'--' +'\n'\
            +'Planet Type='+str(planet.Ptype) +'\n'\
            +'R_p='  + '{:3.5}'.format(str(planet.Rad_p*float(catalog_df.Rad)*109.122))    +'[R_e]'   +'\n'\
            +'t0='   + '{:4.9}'.format(str(planet.T0))       +'[TJD]'   +'\n'\
            +'depth='+ '{:1.6}'.format(str(planet.Depth))               +'\n'\
            +'T='    + '{:2.5}'.format(str(planet.Dur))      +'[h]'     +'\n'\
            +'P_c='  + '{:5.6}'.format(str(planet.Period))      +'[d]'     +'\n'


        if len(planet_df)==0:
            txtstr = txtstr + '\n'+'\n'+'\n'+'\n'+'\n'+'\n'+'\n'+'\n'+'\n'
#         plt.axis([0,1,0,1])
        ax_fin.text(0.05, 0.98, txtstr, transform=ax_fin.transAxes, 
        verticalalignment='top', horizontalalignment='left', fontsize = 10)
#         plt.text(0., 0., txtstr,fontsize=8)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        
        pdf.savefig()
        plt.clf()
        plt.close('all')
        

       