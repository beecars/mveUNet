from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import random

def readUCharImage(fname, im_size=(512,512), as_tensor=False):
    with open(fname, 'rb') as f:
        rawData = f.read()
        img = Image.frombytes('L', im_size, rawData)
        if as_tensor:
            return torch.from_numpy(np.array(img)).float()
        else:
            return np.array(img)
        
def readBinImage(fname, im_size=(512,512), as_tensor=False):
    with open(fname, 'rb') as f:
        rawData = f.read()
        img = Image.frombytes('I', im_size, rawData)
        if as_tensor:
            return torch.from_numpy(np.array(img).T).float()
        else:
            return np.array(img).T

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

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
