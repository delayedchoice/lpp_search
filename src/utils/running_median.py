# utils/running_median.py

import numpy as np


def running_median(data, kernel=25):
    """Returns sliding median of width 'kernel' and same length as data """
    


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

