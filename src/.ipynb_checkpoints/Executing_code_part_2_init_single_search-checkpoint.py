#!/usr/bin/env python
# coding: utf-8

# In[1]:


import batman
import emcee
import glob
import os
import shutil
import math
import corner
import numba
import itertools

import numpy       as np
import pandas      as pd
import time        as tm 
import lightkurve  as lk
# import mr_forecast as mr

import matplotlib                      as mpl
import matplotlib.pyplot               as plt
import matplotlib.gridspec             as gridspec
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


# import eleanor

import warnings
warnings.filterwarnings("ignore")
display(HTML("<style>.container { width:95% !important; }</style>"))

from Functions_all import *


###LET'S DO THIS!!!!
import gc



def main(target):
    # num = 0
# for target in glob.glob('./toi_data/target_*'):
    # num+=1

    # print('running target', num)
    print('target file', target)
    ticid  = int(target.split('/')[-1].split('_')[1].split('-')[1])
    gaiaID = str(target.split('/')[-1].split('_')[2].split('-')[-1])
    catalog_df = False

    if os.path.exists(target+'/tic_star_parameters.csv'):
        print('getting star params')
        catalog_df = pd.read_csv(target+'/tic_star_parameters.csv')

#     else:
#         all_star_data = pd.read_csv('./final_mdwarf_params.csv', header = 0)#
#         print('IDs', ticid, gaiaID)
#         catalog_df = get_catalog_info(ticid, df = all_star_data, rtrn_df = True, gaia_id = gaiaID)
#         if len(catalog_df)>0:
# #         print('catalog info', catalog_df)
#             catalog_df.to_csv(target+'/tic_star_parameters.csv', index = False)
#         else: 
#             catalog_df = False

    # print('catalog: ', catalog_df)
#     for fits in glob.glob(target+'/*.fits'):
#         sector_fits = int(fits.split('/')[-1].split('-')[2][1:])

#         print('fits', fits)
#         print('sector ', sector_fits)
#         # if sector_fits>7:
# #         else:
#         extract_data_from_fits_files(fits, sector = sector_fits)
    
# #     if len(total_file_path)>0:
# #         print('data already detrended')
# #         print(total_file_path)
# #     else:


#         get_data(target, catalog_df = catalog_df) 
    total_file_path = glob.glob(target+'/*TGLC*_total.csv')
    # print('total file', total_file_path)
    if len(total_file_path)==0:
        print('no data file')
        return

    if type(catalog_df)==bool:
        print('try ', ticid, ' again after finish mdwarfs_s0')
        return
#     print('catalog df', catalog_df)
#     print('running first search!!!!')
    
    gc.collect()
    singles_df = singles_search(ticid, total_file_path[0], intransit = [], catalog_df = catalog_df, confidence = 0.5)   
    if len(singles_df)>0:
        os.rename(target, target+'_check')
        print('keeping this ticid: ', ticid)
        # gc.collect()

        return
    else:
        # gc.collect()

        return
        
    

    
if __name__ == "__main__":
    import multiprocessing as mpl
    # from schwimmbad import JoblibPool

    try:
        file_num = int(sys.argv[1])
    except ValueError:
        
        sys.exit(1)

    time1 = tm.time()
    
    target_files = sorted(glob.glob('../new_toi_data/target_*'))
    print('number of files: ', len(target_files), 'file number: ', file_num)
    file = target_files[file_num]
#     print('factor: ', file_factor, ', running files_nums:', file_factor*8 , '-', min(len(target_files), (file_factor+1)*8))
#     pool = mpl.Pool()
    if 'data' in file.split('/')[-1]:
    
    
    
        main(file) 

#         factor_files_max = min(len(target_files), (file_factor+1)*8)
#         factor_files_min = file_factor*8

#         factor_files = target_files[factor_files_min:factor_files_max]
        # for result in pool.imap(main, [file for file in files]):
         #     nfilename = './All_indiv_stars_new/cad_'+str(round(iter_num, 2))+'/yield_vals_'+str(file_num).zfill(4)+'.csv'

        #     result.to_csv(nfilename, index = False, mode='a')

#         with mpl.Pool(8) as pool:
#             pool.map(main, factor_files)

    time_end = tm.time()
    print('time it took: ', (time_end-time1)/60, ' minutes')





