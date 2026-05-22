# utils/dv_report.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

from core.target import Target
from core.planet_candidate import PlanetCandidate

from utils.handling_data import (
    bin_by_time_many_args,
    sort_arrays_by_time,
    sort_arrays_by_index,
    predict_lc, _finite,
)
from utils.segments import find_breaks, breaking_up_data
from utils.find_total_csv import find_total_csv



def build_catalog_df_for_target(target):
    import json
    import numpy as np
    import pandas as pd

    def _to_plain_dict(x):
        """Convert dict/Series/DataFrame-row/JSON-string/None into a plain dict."""
        if x is None:
            return {}
        if isinstance(x, dict):
            return x
        # pandas Series / DataFrame row-like
        if hasattr(x, "to_dict"):
            try:
                d = x.to_dict()
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
        # JSON string
        if isinstance(x, str):
            try:
                d = json.loads(x)
                return d if isinstance(d, dict) else {}
            except Exception:
                return {}
        return {}

    # Try common locations, in order
    c = _to_plain_dict(getattr(target, "catalog", None))

    if not c:
        # some Target implementations store state as a dict containing "catalog"
        state = getattr(target, "state", None)
        if isinstance(state, dict):
            c = _to_plain_dict(state.get("catalog"))

    if not c:
        # last-resort: if Target has a to_dict() or similar
        if hasattr(target, "to_dict"):
            try:
                d = target.to_dict()
                if isinstance(d, dict):
                    c = _to_plain_dict(d.get("catalog"))
            except Exception:
                pass

    row = {
        "RA": c.get("RA", np.nan),
        "DEC": c.get("DEC", np.nan),
        "Rad": c.get("Rad", np.nan),
        "Mass": c.get("Mass", np.nan),
        "Teff": c.get("Teff", np.nan),
        "Tmag": c.get("Tmag", np.nan),
        "Vmag": c.get("Vmag", np.nan),
        "Jmag": c.get("Jmag", np.nan),
        "ContRatio": c.get("ContRatio", np.nan),
        "aLSM": c.get("aLSM", np.nan),
        "bLSM": c.get("bLSM", np.nan),
    }
    return pd.DataFrame([row]).iloc[0]

def build_planet_df_from_final_csv(final_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(final_csv)
    df = df[df['default'] == True] 

    out = pd.DataFrame()
    out["Planet_Num"] = np.arange(1, len(df) + 1)


    # normalize ptype naming to what your older report code expects
    out["Ptype"] = df["ptype"].astype(str)

    if len(df[df['ptype']=='periodic'])>1:
        out.loc[out["Ptype"].str.lower().str.startswith("p"), "Ptype"] = "Period"
    if len(df[df['ptype']=='single'])>1:

        out.loc[out["Ptype"].str.lower().str.startswith("s"), "Ptype"] = "Single"

    out["T0"] = df["t0_days"].astype(float)
    out["Period"] = df["period_days"]
    out.loc[out["Ptype"] == "Single", "Period"] = float("inf")

    out["Depth"] = df["depth"].astype(float)
    # your code sometimes treats duration in hours for display; adjust if needed
    out["Dur"] = df["duration_days"].astype(float) * 24.0

    if "rp_rs" in df.columns:
        out["Rad_p"] = df["rp_rs"].astype(float)
    if "cosi" in df.columns:
        out["Cosi"] = df["cosi"].astype(float)
    if "a_smaj" in df.columns:
        out["Semi_Maj"] = df["a_smaj"].astype(float)


    for col in ["Rad_p", "Cosi", "Semi_Maj"]:
        if col not in out.columns:
            out[col] = np.nan

    return out


# ---- Adapter: build catalog “row-like” object the report expects ----


def format_candidate_table(planet_df, ax, max_rows = 10):
    planet_df['Depth'] = planet_df['Depth']*1000
    df_rounded = planet_df.round(4)
    columns = ["#", "Type", "Period [d]", 't0 [TJD]', 'depth [ppt]', 'duration [h]']

    table_data = df_rounded.iloc[:, :len(columns)].values.tolist()
    # columns = df_rounded.columns.tolist()
    # Create the table


        
    table = ax.table(cellText=table_data, colLabels=columns, loc='center')
    # 2. Disable auto-font size and set the new size manually
    # table.auto_set_font_size(False)
    # table.set_fontsize(25)  # Change 14 to your desired size

    return ax


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
    

    if len(np.array([err])) == 0 or err is None:
        err = np.full( len(flux), np.std(flux))
    if len(np.array([binned_err])) == 0 or binned_err is None:
        binned_err = np.full(len(binned_flux), np.std(flux))

    if len(ratios)==0 or binned_err is None:
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


    
    d = 0.02 # how big to make the diagonal lines in axes coordinates
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
CADENCES = [20/60, 2., 200/60,  10., 30.]
CAD_PS = 2.
CAD_FFI = 30.


def creating_first_DV_report_page(target, planet_df, intransit=[]):
    ticid = target.ticid
    if planet_df is None:
        planet_df = build_planet_df_from_final_csv(final_csv)

    total_csv = find_total_csv(target.root_dir, target.data_source.value)
    df = pd.read_csv(total_csv).dropna(subset=["FLUX"])
    time, flux, err, trend, raw, raw_err, bkg = [np.array(df[col]) for col in ['TIME', 'FLUX', 'FLUX_ERR', 'FLUX_TREND', 'RAW_FLUX', 'RAW_FLUX_ERR', 'BKG_FLUX']]
    if intransit is None or (isinstance(intransit, (list, tuple)) and len(intransit) == 0):
        intransit = np.zeros(len(time), dtype=bool)
    else:
        intransit = np.asarray(intransit, dtype=bool)
        if len(intransit) != len(time):
            intransit = np.zeros(len(time), dtype=bool)

    
    catalog_df = target._catalog

    try:
        u1 = float(catalog_df['aLSM'])
        u2 = float(catalog_df['bLSM'])

    except Exception:
        u1, u2 = np.nan, np.nan
    
    print('checking lengths', [len(x) for x in [time, flux, err, trend, raw, raw_err]])
    
    if len(err) != len(flux):
        err = np.full(len(flux), np.std(flux))
        print('Error')
    
    binned_time, binned_flux, binned_err,  binned_trend, binned_raw, binned_rerr = bin_data_with_diff_cadences_many_args(time, flux = flux, err = err, trend = trend, raw = raw, raw_err = raw_err)

    indexes_split = breaking_up_data(time)   

    binned_indexes_split = breaking_up_data(binned_time)   

    split_times, split_fluxes, split_err, split_raw, split_rerr = sort_arrays_by_index(indexes_split, time, flux, err, raw, raw_err)
    
    binned_split_times, binned_split_fluxes, binned_split_err, binned_split_raw, binned_split_rerr = sort_arrays_by_index(binned_indexes_split, binned_time, binned_flux, binned_err, binned_raw, binned_rerr)

    diff_time_arrays = np.array([np.ptp(x) for x in split_times])
    print('length of different time arrays: ', diff_time_arrays)
    min_diff_time_arrays = min(diff_time_arrays)
    ratios = diff_time_arrays/min_diff_time_arrays
            
    fig0 = plt.figure(figsize=(8.5, 11), constrained_layout=True,dpi=100)
    gs = fig0.add_gridspec(1,2,width_ratios=[4.25, 1], wspace = 0.1) #create grid for subplots - makes it easier to assign where each plot goes
    
    
    gs0 = gs[0].subgridspec(5, len(split_times), wspace=0.02, width_ratios = ratios)
    gs1 = gs[1].subgridspec(1, 1)   
    ymin = np.nanmin([np.percentile(raw, 1),
                    np.percentile(flux, 0.5),
                    1.0 - np.nanmax(planet_df["Depth"].to_numpy(float))])
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
    
    
    ymin2 = np.nanmin([np.percentile(flux, 2)])*0.99
    #,1.-(max(planet_df['Depth']))]) #define y-axis limits by percentages to avoid using es 
    ymax2 = np.percentile(flux,98)*1.01
    delta_y2 = np.abs(ymax2-ymin2)
    ymin2 = ymin2-(delta_y2*.01) #make sure ymin allows for all data        

    per_planets_df = planet_df[planet_df['Ptype']=='Periodic']
    print('planet df', planet_df)

    if len(per_planets_df)>0:
        

        axes2 = creating_broken_axes_plots_for_DV_report_min_plot(
            time, flux, err,
            binned_time, binned_flux, binned_err,
            gs0, subplot, ratios)

        for ax in axes2:
            min_vals, max_vals = ax.get_xlim()
            split_time = time[(time>min_vals) & (time<max_vals)]
            cad = np.min(np.diff(split_time))

            for indx, planet in per_planets_df.iterrows():

                if not _finite(planet.Rad_p, u1, u2):
                    print('do we get here', planet.Rad_p, u1, u2)

                    continue
                elif not _finite(planet.Cosi, planet.Semi_Maj):
                    planet.Cosi = 90
                    planet.Semi_Maj = 10
                model_time = np.arange(min_vals,max_vals,cad) #creates a uniformly spaced around spanning the length of time measurements taken in 30 minute intervals

                print('getting model time', len(model_time ))
                model_flux = predict_lc(model_time, planet.T0, planet.Period,
                                        planet.Rad_p, planet.Cosi, planet.Semi_Maj,
                                        u1, u2, cad)
                model_flux = model_flux/np.nanmedian(model_flux)
                ax.plot(model_time, model_flux, color = 'C'+str(indx), lw = 2, alpha = 0.7, zorder = 1E3)
                try:
                    ax.set_ylim(ymin2, ymax2)
                except Exception as e:
                    print(e)
                    print(f'ymin: {ymin2}, ymax: {ymax2}, Error')

        
        axes = axes+axes2
    subplot+=1

    times_oot, fluxes_oot, err_oot = time[~intransit], flux[~intransit],err[~intransit]
    
    

    binned_time_oot, binned_flux_oot, binned_err_oot = bin_data_with_diff_cadences_many_args(times_oot, flux = fluxes_oot, err = err_oot)
    
    
    single_planet_df = planet_df[planet_df['Ptype']=='Single'].reset_index(drop = True)

    if len(single_planet_df)>0:

        axes3 = creating_broken_axes_plots_for_DV_report_min_plot(
            times_oot, fluxes_oot, err_oot,
            binned_time_oot, binned_flux_oot, binned_err_oot,
            gs0, subplot, ratios
        )
        for ax in axes3:
            min_vals, max_vals = ax.get_xlim()
            split_time = times_oot[(times_oot>min_vals) & (times_oot<max_vals)]
            cad = np.min(np.diff(split_time))

            for indx, planet in single_planet_df.iterrows():
#                     bboxes = DT_analysis(split_time, fluxes_oot[(times_oot>min_vals) & (times_oot<max_vals)], err_oot[(times_oot>min_vals) & (times_oot<max_vals)], confidence = 0.65)
#                     detrended_lc = make_LightKurveObject(times_oot, fluxes_oot, err_oot)
#                 #         print(detrended_lc)
#                     plot_lc_with_bboxes(detrended_lc, bboxes, ms=3, marker='.', lw=0, ax = ax)
                if not _finite(planet.Rad_p, planet.Cosi, planet.Semi_Maj, u1, u2):
                    continue
                model_time = np.arange(min_vals, max_vals,cad) 
                model_flux = predict_lc(model_time, planet.T0, planet.Period, planet.Rad_p, planet.Cosi,planet.Semi_Maj, u1, u2, cad)#*planet.Norm #create a model of the transit
                
                
                model_flux = model_flux/np.nanmedian(model_flux)

                ax.plot(model_time, model_flux, color = 'C'+str(indx+len(per_planets_df)), lw = 2, alpha = 0.7, zorder = 10)
                ax.set_ylim(ymin2, ymax2)

    
        axes = axes+axes3
        
        
    subplot +=1

    axes4 = creating_broken_axes_plots_for_DV_report_min_plot(
        time, bkg, None,
        time, bkg, None,
        gs0, subplot, ratios
    )

    ymin3 = np.nanmin([np.percentile(bkg, 0.25)])*0.95
    ymax3 = np.percentile(bkg,99.5)*1.05
    delta_y3 = np.abs(ymax3-ymin3)
    ymin3 = ymin3-(delta_y3*.05) #make sure ymin allows for all data        


    for ax in axes4:
        ax.set_ylim(ymin3, ymax3)


    axes = axes+axes4
    subplot +=1


    ax_tbl = plt.subplot(gs0[subplot, :])  # span all columns for that row
    ax_tbl.axis("off")
    ax_tbl = format_candidate_table(planet_df, ax_tbl, max_rows=10)

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
#                 split_time = times_oot[(times_oot>min_vals) & (times_oot<max_vals)]
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
#                 split_time = times_oot[(times_oot>min_vals) & (times_oot<max_vals)]
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
#                     split_time = times_oot[(times_oot>min_vals) & (times_oot<max_vals)]
#                     cad = np.min(np.diff(split_time))
#                 axes = axes+axes_n


        

    for ax in axes:
        min_vals, max_vals = ax.get_xlim()
        ymin_, ymax_ = ax.get_ylim()
        delta_y = abs(ymax_ - ymin_)
        for indx, planet in planet_df.iterrows():   
            if planet["Ptype"] == "Single":
                epochs = np.array([planet["T0"]], dtype=float)
            else:
                # print('checking ', min_vals, max_vals, planet["T0"], planet["Period"])
                epochs = find_t0_vals_within_time(min_vals, max_vals, planet["T0"], planet["Period"])
#     print('should be sorted times', min_t, epochs, max_t)

            # epochs = find_t0_vals_within_time(min_vals, max_vals, planet['T0'], planet['Period'])
#                 print('planet num', planet['Planet_Num'])
            if planet['Ptype']=='Single':
                ax.scatter(epochs, np.full(len(epochs) ,ymin_ + 0.1*delta_y), marker='^', color = 'C'+str(indx), s=50, zorder = 1000)
            else:
                ax.scatter(epochs, np.full(len(epochs) ,ymin_ + 0.05*delta_y), marker='^', color = 'C'+str(indx), facecolors='none', s=30, zorder = 5000)


                
                
                
    ax_fin = plt.subplot(gs1[:,-1]) #for the last subplot, print text
    txtstr = 'TICID'.strip().ljust(6)+ '='+str(ticid)                              +'\n'\
        +'RA'.strip().ljust(6)   +'='.ljust(2)+ str(round(float(catalog_df['RA']), 6))          +'\n'\
        +'DEC'.strip().ljust(6)  +'='.ljust(2)+ str(round(float(catalog_df['DEC']), 6))       +'\n'\
        +'R_*'.strip().ljust(6)  +'='.ljust(2)+ str(round(float(catalog_df['Rad']), 3)).strip().ljust(5)    +'  [R_s]'   +'\n'\
        +'M_*'.strip().ljust(6)  +'='.ljust(2)+ str(round(float(catalog_df['Mass']), 3)).strip().ljust(5)   +'  [M_s]'   +'\n'\
        +'Teff'.strip().ljust(6) +'='.ljust(2)+ str(round(float(catalog_df['Teff']), 3)).strip().ljust(5)    +'  [K]'     +'\n'\
        +'Tmag'.strip().ljust(6) +'='.ljust(2)+ str(round(float(catalog_df['Tmag']), 3)).strip().ljust(5)     +'\n'\
        +'Vmag'.strip().ljust(6) +'='.ljust(2)+ str(round(float(catalog_df['Vmag']), 3)).strip().ljust(5)             +'\n'\
        +'Jmag'.strip().ljust(6) +'='.ljust(2)+ str(round(float(catalog_df['Jmag']), 3)).strip().ljust(5)           +'\n'\
        +'Cont'.strip().ljust(6) +'='.ljust(2)+ str(round(float(catalog_df['ContRatio']), 3))                   +'\n'\
        # +'----- Planet Parmas -----'                              +'\n'\
    
    # for indx, planet in planet_df.iterrows():
    #     txtstr = txtstr + '--' +'Planet Num='+ str(int(planet.Planet_Num))+'--' +'\n'\
    #     +'Planet Type='+str(planet.Ptype) +'\n'\
    #     +'R_p='  + '{:3.5}'.format(str(planet.Rad_p*float(catalog_df.Rad)*109.122))    +'[R_e]'   +'\n'\
    #     +'t0='   + '{:4.9}'.format(str(planet.T0))       +'[TJD]'   +'\n'\
    #     +'depth='+ '{:1.6}'.format(str(planet.Depth))               +'\n'\
    #     +'T='    + '{:2.5}'.format(str(planet.Dur))      +'[h]'     +'\n'\
    #     +'P_c='  + '{:5.6}'.format(str(planet.Period))      +'[d]'     +'\n'


    # if len(planet_df)==0:
    #     txtstr = txtstr + '\n'+'\n'+'\n'+'\n'+'\n'+'\n'+'\n'+'\n'+'\n'
#         plt.axis([0,1,0,1])
    ax_fin.text(0.05, 0.98, txtstr, transform=ax_fin.transAxes, 
    verticalalignment='top', horizontalalignment='left', fontsize = 10)
#         plt.text(0., 0., txtstr,fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    return fig0

