import random

import numpy as np
from scipy.io import loadmat

import torch
from torch.utils.data import Dataset
from pathlib import Path
from utils.utils import loadMatData
from os import environ

class VolumeDataset(Dataset):
    """ A Dataset genereated from a [vol_idx] representing a .mat file.
    Files must have particular naming convention: 'patient1_day1.mat'

    @params:
        vol_idx: the [p, d] index to be loaded into the Dataset.
        folder: the folder containing the 'patient#_day#.mat' files.
        data: the variable to load from the .mat file (saves time?).
    """
    def __init__(self,
                 vol_idx,
                 folder = environ['REVEAL_DATA'] + '\\ct_mask_volumes\\',
                 data = ['ct']):
        self.data = data
        self.ct_vol = loadMatData(vol_idx, folder, 'ct')
       
    
    def __getitem__(self, idx):
        ct = torch.from_numpy(self.ct_vol[:, :, idx]).unsqueeze(0).float()
        return {'ct': ct}

    def __len__(self):
        return self.ct_vol.shape[-1]    # length is last dimension of shape


class CTMaskDataset(Dataset):
    """ A dataset representing a map from filenames to .npy scan data. 
    Currently only works for single-class spine segmentation.
    Requires a particular data folder structure:
        data_dir
          |--- ct
          |     |--- 0.npy
          |     |--- 1.npy
          |     |...
          |--- spine
          |...  |--- 0.npy
                |...

    @params:
    data_dir: the path to the data_dir in the diagram above.
    """
    def __init__(self, 
                 data_dir = environ['REVEAL_DATA'] + '\\train_data\\'):
        ct_path = Path(data_dir + '/ct')
        self.ct_files = [file.__str__() for file in list(ct_path.glob('*'))]
        spine_path = Path(data_dir + '/spine')
        self.spine_files = [file.__str__() for file in list(spine_path.glob('*'))]
    
    def __getitem__(self, idx):
        ct = np.load(self.ct_files[idx])
        spine = np.load(self.spine_files[idx])
        
        ct = torch.from_numpy(ct).unsqueeze(0)
        spine = torch.from_numpy(spine).unsqueeze(0)

        return {'image': ct, 
                'target': spine}
    
    def __len__(self):
        return len(self.ct_files)