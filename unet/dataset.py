import numpy as np
import random
import torch
from torch.utils.data import Dataset

from utils import readBinImage, readUCharImage

class CTMaskDataset(Dataset):
    ''' Single class training Dataset.
    Input data should be a 2D list of: [ct filepath, mask filepath]. 
    Has augment function which randomly crops and flips at __getitem__ retreival.
    '''
    def __init__(self, 
                 ct_mask_data,
                 img_size=(512, 512), 
                 offset=(150,150), 
                 output_size=(320, 320), 
                 augment=True):

        self.data = ct_mask_data
        self.img_size = img_size
        self.offset = offset
        self.output_size = output_size
        self.augment = augment
            
    def __getitem__(self, idx):
        ct_fname, mask_fname = self.data[idx]
        ct = readBinImage(ct_fname)
        mask = readUCharImage(mask_fname)
        Kx, Ky = self.offset
        Lx, Ly = self.output_size
        if self.augment:
            kx = random.randint(0, Kx)
            ky = random.randint(0, Ky)
        else:
            kx, ky = 96, 96
        ct = ct[kx:kx+Lx, ky:ky+Ly]
        mask = mask[kx:kx+Lx, ky:ky+Ly]
        
        if self.augment and np.random.randn() > 0.5:
            ct = np.fliplr(ct).copy()
            mask = np.fliplr(mask).copy()

        ct = torch.from_numpy(ct).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return {'image': ct, 
                'target': mask}
    
    def __len__(self):
        return len(self.data)