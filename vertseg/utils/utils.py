from scipy.io import whosmat, loadmat
from os import environ
import numpy as np

def loadMatData(vol_idx,
                folder = environ['REVEAL_DATA'] + '\\ct_mask_volumes\\',
                data = 'ct'):
    """
    Loads .mat (MATLAB-type) files into system memory.
    Files must have particular naming convention: 'patient1_day1.mat'
    
    @params:
    vol_idx: the volume identifier in the form [p, d] to be loaded.
    folder: the folder containing the 'patient#_day#.mat' files.
    data: the variable to load from the .mat file (saves time?).

    @returns:
    mat: nparray data object containg the volume data.
    """
    mat = loadmat(folder + f'patient{vol_idx[0]}_day{vol_idx[1]}.mat',
                  variable_names = data)[data]
    return mat


def getScanCount(vol_idxs,
                 folder = environ['REVEAL_DATA'] + '\\ct_mask_volumes\\'):
    """
    From a list of vol_idxs determine the number of scans present.
    Used for counting training/validation scans.
    Files must have particular naming convention: 'patient1_day1.mat'

    @params:
    vol_idxs: a list of vol_idx to be inspected
    folder: the folder containing the 'patient#_day#.mat' files.

    @returns:
    total_scans: the number of scans in the volumes (int)
    """
    total_scans = 0
    for vol_idx in vol_idxs:
        vol_file = (folder + f'patient{vol_idx[0]}_day{vol_idx[1]}.mat')
        scans = whosmat(vol_file)[0][1][-1]
        total_scans = total_scans + scans
    return total_scans