import random

import numpy as np
from scipy.io import loadmat

import torch
from torch.utils.data import Dataset
from utils.utils import readUCharImage, readBinImage

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

class CTSequenceDataset(Dataset):
    ''' 
    Dataset for containing a SINGLE CT volume as an ordered list of files 
    representing ordered CT slices. Input data should be an ordrered 1D list of 
    filepath.
    @params: 
        ct_data = a specific data list generated for the REVEAL CT data
                  by using the matchFilesFromPatients() function. 
    '''
    def __init__(self,
                 ct_data):
        
        self.data = ct_data

    def __getitem__(self, idx):
        ct_fname = self.data[idx]
        ct = readBinImage(ct_fname)
        ct = torch.from_numpy(ct).unsqueeze(0).float() 

        return {'image': ct}

    def __len__(self):
        return len(self.data)

class CTMaskDataset(Dataset):
    ''' 
    Single class training Dataset for REVEAL CT data. 
    Input data should be . 
    Has augment function which randomly crops and flips at __getitem__ retreival.
    @params: 
        ct_mask_data = a 2D list of: [ct filepath, mask filepath]. Generated 
                        for the REVEAL CT data using the matchFilesFromPatients() 
                        function
        img_size = the original size of the training images/masks
        offset = the potential offset for translation augmentation in pixels
        output_size = the size returned by the class __getitem__ method
        augment = boolean, whether or not to augment the data upon retreival
    '''
    def __init__(self, 
                 ct_mask_data,
                 img_size = (512, 512), 
                 offset = (150, 150), 
                 output_size = (320, 320), 
                 augment = True):

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


class CTMulticlassDataset(Dataset):
    ''' 
    Multiple class training Dataset.
    Input data should be a 2D list of: [ct_path, mask1_path, mask2_path, ...]. 
    Has augment function which randomly crops and flips at __getitem__ retreival.
    @params: 
        ct_mask_data = a 2D list of: [ct filepath, mask filepath]. Generated 
                        for the REVEAL CT data using the matchFilesFromPatients() 
                        function
        img_size = the original size of the training images/masks
        offset = the potential offset for translation augmentation in pixels
        output_size = the size returned by the class __getitem__ method
        augment = boolean, whether or not to augment the data upon retreival
    '''
    def __init__(self, 
                 ct_multimask_data,
                 img_size=(512, 512), 
                 offset=(150,150), 
                 output_size=(320, 320), 
                 augment=True):

        self.data = ct_multimask_data
        self.img_size = img_size
        self.offset = offset
        self.output_size = output_size
        self.augment = augment
            
    def __getitem__(self, idx):
        ct_fname, spine_fname, sternum_fname, pelvis_fname = self.data[idx]
        ct = readBinImage(ct_fname)
        spine_mask = readUCharImage(spine_fname)
        stern_mask = readUCharImage(sternum_fname)
        pelvi_mask = readUCharImage(pelvis_fname)
        
        Kx, Ky = self.offset
        Lx, Ly = self.output_size
        if self.augment:
            kx = random.randint(0, Kx)
            ky = random.randint(0, Ky)
        else:
            kx, ky = 96, 96
        
        ct = ct[kx:kx+Lx, ky:ky+Ly]
        spine_mask = spine_mask[kx:kx+Lx, ky:ky+Ly]
        stern_mask = stern_mask[kx:kx+Lx, ky:ky+Ly]
        pelvi_mask = pelvi_mask[kx:kx+Lx, ky:ky+Ly]
        
        if self.augment and np.random.randn() > 0.5:
            ct = np.fliplr(ct).copy()
            spine_mask = np.fliplr(spine_mask).copy()
            stern_mask = np.fliplr(stern_mask).copy()
            pelvi_mask = np.fliplr(pelvi_mask).copy()

        ct = torch.from_numpy(ct).unsqueeze(0).float()

        spine_mask = torch.from_numpy(spine_mask).float()
        stern_mask = torch.from_numpy(stern_mask).float()
        pelvi_mask = torch.from_numpy(pelvi_mask).float()
        
        mask_stack = torch.stack([spine_mask, stern_mask, pelvi_mask], 
                                 dim = 0)

        return {'image': ct, 
                'target': mask_stack
               }