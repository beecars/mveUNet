import random

import numpy as np
from scipy.io import loadmat

import torch
from torch.utils.data import Dataset
from pathlib import Path
from utils.utils import loadMatData
from os import environ
from utils.augment import augment_dict

class VolumeDataset(Dataset):
    """ A Dataset genereated from a [vol_idx] representing a .mat file.
    Files must have particular naming convention: 'patient1_day1.mat'

    @params:
        vol_idx: the [p, d] index to be loaded into the Dataset.
        folder: the folder containing the 'patient#_day#.mat' files.
        var: the variable to load from the .mat file (saves time?).
    """
    def __init__(self,
                 vol_idx,
                 folder = environ['REVEAL_DATA'] + '\\ct_mask_volumes\\',
                 var = 'ct'):
        self.ct_vol = loadMatData(vol_idx, folder, var)
       
    def __getitem__(self, idx):
        ct = torch.from_numpy(self.ct_vol[:, :, idx]).unsqueeze(0).float()
        return {'ct': ct}

    def __len__(self):
        return self.ct_vol.shape[-1]    # length is last dimension of shape


class CTMaskDataset(Dataset):
    """ A dataset for accessing a folder of 2D image data files in .npy format. 
   
    Requires a particular data folder structure:
    
        data_dir
          |--- ct                 (this structure is automatically 
          |     |--- 0.npy                 generated by "generateNpySlices()")
          |     |--- 1.npy
          |     |...
          |--- spine
          |...  |--- 0.npy
                |...
    
    Currently only works for single-class spine segmentation.
    
    @params:
    data_dir: the path to the data_dir in the diagram above.
    mask_criteria: list of strings identifying the masks that to train with.
    """
    def __init__(self, 
                 data_dir = environ['REVEAL_DATA'] + '\\train_data\\',
                 mask_criteria = ['spine'],
                 augment = True):
        ct_path = Path(data_dir + '/ct')
        self.mask_criteria = mask_criteria
        self.augment = augment
        self.ct_files = [file.__str__() for file in list(ct_path.glob('*'))]
        if 'spine' in mask_criteria:
            spine_path = Path(data_dir + '/spine')
            self.spine_files = [file.__str__() for file in list(spine_path.glob('*'))]
        if 'sternum' in mask_criteria:
            stern_path = Path(data_dir + '/sternum')
            self.stern_files = [file.__str__() for file in list(stern_path.glob('*'))]
        if 'pelvis' in mask_criteria:
            pelvi_path = Path(data_dir + '/pelvis')
            self.pelvi_files = [file.__str__() for file in list(pelvi_path.glob('*'))]
           
    def __getitem__(self, idx):
        # load up the ct data into a dict 
        ct = np.load(self.ct_files[idx])
        data_dict = {'ct': ct}
        # load up the mask data matching the mask_criteria
        if 'spine' in self.mask_criteria:
            spine = np.load(self.spine_files[idx])
            data_dict['spine'] = spine
        if 'sternum' in self.mask_criteria:
            stern = np.load(self.stern_files[idx])
            data_dict['stern'] = stern  
        if 'pelvis' in self.mask_criteria:
            pelvi = np.load(self.pelvi_files[idx])
            data_dict['pelvi'] = pelvi
        # optionally perform augmentation
        if self.augment == True:
            data_dict = augment_dict(data_dict)
        # convert to npy arrays
        for item in data_dict:
            data_dict[item] = torch.from_numpy(data_dict[item]).unsqueeze(0).float()
        
        return data_dict
    
    def __len__(self):
        return len(self.ct_files)   # ct_files always populated at __init__()