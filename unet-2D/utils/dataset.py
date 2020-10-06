import random

import numpy as np
from scipy.io import loadmat

import torch
from torch.utils.data import Dataset
from pathlib import Path

class CTVolumeDataset(Dataset):
    '''
    A CT volume Dataset genereated from one of the following: a 3D numpy 
    array, a MATLAB .mat file, or numpy .npy file.
    @params:
        x = either a 3D np array data OR a filetype, depending on mode.
        mode = 'npy', 'mat', or 'data' (default data)
    '''
    def __init__(self,
                 x,
                 mode = 'data'):
        if mode == 'data':
            self.volume = x
        elif mode == 'mat':                                  
            mat_dict = loadmat(x)                            
            self.volume = list(mat_dict.items())[-1][-1]  # last tuple last dict
        elif mode =='npy':                                  
            self.volume = np.load(x)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.volume[:, :, idx]).unsqueeze(0).float()

    def __len__(self):
        return self.volume.shape[-1]    # length is last dimension of shape


class CTMaskDataset(Dataset):
    ''' 
    fill this
    '''
    def __init__(self, 
                 data_folder):
        ct_path = Path(data_folder + '/ct')
        self.ct_files = [file.__str__() for file in list(ct_path.glob('*'))]
        spine_path = Path(data_folder + '/spine')
        self.spine_files = [file.__str__() for file in list(spine_path.glob('*'))]
    
    def __getitem__(self, idx):
        ct = np.load(self.ct_files[idx])
        spine = np.load(self.spine_files[idx])
        return {'image': ct, 
                'target': spine}
    
    def __len__(self):
        return len(self.data)