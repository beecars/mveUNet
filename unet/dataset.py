from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random

class CTMaskDataset(Dataset):
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
  
    def _readUCharImage(self, fname):
        with open(fname, 'rb') as f:
            rawData = f.read()
            img = Image.frombytes('L', self.img_size, rawData)
            return np.array(img).astype(np.float32)
 
    def _readBinImage(self, fname):
        with open(fname, 'rb') as f:
            rawData = f.read()
            img = Image.frombytes('I', self.img_size, rawData)
            return np.array(img).T.astype(np.float32)
        
    def _readUCharImageToPIL(self, fname):
        with open(fname, 'rb') as f:
            rawData = f.read()
            img = Image.frombytes('L', self.img_size, rawData)
            return img
 
    def _readBinImageToPIL(self, fname):
        with open(fname, 'rb') as f:
            rawData = f.read()
            img = Image.frombytes('I', self.img_size, rawData)
            return img
            
    def __getitem__(self, idx):
        ct_fname, mask_fname = self.data[idx]
        ct = self._readBinImage(ct_fname)
        mask = self._readUCharImage(mask_fname)
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
        
        return {'data': np.expand_dims(ct, axis=0), 
                'label': np.expand_dims(mask, axis=0)}
    
    def __len__(self):
        return len(self.data)