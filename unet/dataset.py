from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap

def CenterCrop(height, width):
    crop = iaa.CropToFixedSize(height=height, width=width)
    crop.position = (iap.Deterministic(0.5), iap.Deterministic(0.5))
    pad = iaa.PadToFixedSize(height=height, width=width, pad_mode=ia.ALL, pad_cval=(0, 255))
    pad.position = (iap.Deterministic(0.5), iap.Deterministic(0.5))
    return iaa.Sequential([crop, pad])


class CTTestDataset(Dataset):
    def __init__(self, ct_mask_data, \
                 img_size=(512, 512)):
        self.data = ct_mask_data
        self.img_size = img_size
  
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
        ct_fname = self.data[idx]
        ct = self._readBinImage(ct_fname)
        return {'data': np.expand_dims(ct, axis=0)}
    
    def __len__(self):
        return len(self.data)


class CTMaskDataset(Dataset):
    def __init__(self, ct_mask_data, \
                 img_size=(512, 512), \
                 offset=(150,150), \
                 output_size=(320, 320), \
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
        
    def __getitem_transform__(self, idx):
        ct_fname, mask_fname = self.data[idx]
        ct = self._readBinImageToPIL(ct_fname)
        mask = self._readUCharImageToPIL(mask_fname)
        Kx, Ky = self.offset
        Lx, Ly = self.output_size
        if self.augment:
            kx = random.randint(0, Kx)
            ky = random.randint(0, Ky)
        else:
            kx, ky = 96, 96
        ct = ct[kx:kx+Lx, ky:ky+Ly]
        mask = mask[kx:kx+Lx, ky:ky+Ly]
        
        #if self.augment and np.random.randn() > 0.5:
        #    ct = ct[:, ::-1]
        #    mask = mask[:, ::-1]
        self.transform(ct)
        self.transform(mask)
        
        return {'data': np.expand_dims(ct, axis=0), 
                'label': np.expand_dims(mask, axis=0)}
            
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
        
        #torch.from_numpy(np.flip(x,axis=0).copy())
        
        return {'data': np.expand_dims(ct, axis=0), 
                'label': np.expand_dims(mask, axis=0)}
    
    def __len__(self):
        return len(self.data)


class CTPTMaskDataset(Dataset):
    def __init__(self, ct_pt_mask_data, \
                 img_size=(512, 512), \
                 offset=(150,150), \
                 output_size=(320, 320), \
                 augment=True):
        self.data = ct_pt_mask_data
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
        
    def _readFloatImage(self, fname):
        with open(fname, 'rb') as f:
            rawData = f.read()
            img = Image.frombytes('F', self.img_size, rawData)
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
        ct_fname, pt_fname, mask_fname = self.data[idx]
        ct = self._readBinImage(ct_fname)
        pt = self._readFloatImage(pt_fname).T
        pt[np.isnan(pt)] = 0.0
        mask = self._readUCharImage(mask_fname)
        Kx, Ky = self.offset
        Lx, Ly = self.output_size
        if self.augment:
            kx = random.randint(0, Kx)
            ky = random.randint(0, Ky)
        else:
            kx, ky = 0, 0
        ct = ct[kx:kx+Lx, ky:ky+Ly]
        pt = pt[kx:kx+Lx, ky:ky+Ly]
        mask = mask[kx:kx+Lx, ky:ky+Ly]
        
        if self.augment and np.random.randn() > 0.5:
            ct = np.fliplr(ct).copy()
            pt = np.fliplr(pt).copy() 
            mask = np.fliplr(mask).copy()
        
        #torch.from_numpy(np.flip(x,axis=0).copy())
        ct = np.expand_dims(ct, axis=0)
        pt = np.expand_dims(pt, axis=0)
        data = np.vstack((ct, pt))
        
        return {'data': data, 
                'label': np.expand_dims(mask, axis=0)}
    
    def __len__(self):
        return len(self.data)


class CTMultiClassDataset(Dataset):
    def __init__(self, ct_pt_mask_data, \
                 img_size=(512, 512), \
                 offset=(150,150), \
                 output_size=(320, 320), \
                 augment=True):
        self.data = ct_pt_mask_data
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
        
    def _readFloatImage(self, fname):
        with open(fname, 'rb') as f:
            rawData = f.read()
            img = Image.frombytes('F', self.img_size, rawData)
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
        ct_fname, spine_mask_fname, sternum_mask_fname, pelvis_mask_fname = self.data[idx]
        ct = self._readBinImage(ct_fname)
        spine_mask = self._readUCharImage(spine_mask_fname)
        sternum_mask = self._readUCharImage(sternum_mask_fname)
        pelvis_mask = self._readUCharImage(pelvis_mask_fname)
        Kx, Ky = self.offset
        Lx, Ly = self.output_size
        if self.augment:
            kx = random.randint(0, Kx)
            ky = random.randint(0, Ky)
        else:
            kx, ky = 0, 0
        ct = ct[kx:kx+Lx, ky:ky+Ly]
        spine_mask = spine_mask[kx:kx+Lx, ky:ky+Ly]
        sternum_mask = sternum_mask[kx:kx+Lx, ky:ky+Ly]
        pelvis_mask = pelvis_mask[kx:kx+Lx, ky:ky+Ly]
        
        if self.augment and np.random.randn() > 0.5:
            ct = np.fliplr(ct).copy()
            spine_mask = np.fliplr(spine_mask).copy() 
            sternum_mask = np.fliplr(sternum_mask).copy()
            pelvis_mask = np.fliplr(pelvis_mask).copy()
            
        if self.augment and np.random.randn() > 0.5:
            ct = np.flipud(ct).copy()
            spine_mask = np.flipud(spine_mask).copy() 
            sternum_mask = np.flipud(sternum_mask).copy() 
            pelvis_mask = np.flipud(pelvis_mask).copy() 
            
        if self.augment and np.random.randn() > 0.5:
            ct = np.rot90(ct).copy()
            spine_mask = np.rot90(spine_mask).copy() 
            sternum_mask = np.rot90(sternum_mask).copy() 
            pelvis_mask = np.rot90(pelvis_mask).copy() 
        
        #torch.from_numpy(np.flip(x,axis=0).copy())
        ct = np.expand_dims(ct, axis=0)
        spine_mask = np.expand_dims(spine_mask, axis=0)
        sternum_mask = np.expand_dims(sternum_mask, axis=0)
        pelvis_mask = np.expand_dims(pelvis_mask, axis=0)
        masks = np.vstack((spine_mask, sternum_mask, pelvis_mask))
        
        return {'data': ct, 
                'label': masks}
    
    def __len__(self):
        return len(self.data)


class CTMultiClassDatasetImgaug(Dataset):
    def __init__(self, ct_pt_mask_data, \
                 img_size=(512, 512), \
                 offset=(150,150), \
                 output_size=(320, 320), \
                 augment=True):
        self.data = ct_pt_mask_data
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
        
    def _readFloatImage(self, fname):
        with open(fname, 'rb') as f:
            rawData = f.read()
            img = Image.frombytes('F', self.img_size, rawData)
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
        ct_fname, spine_mask_fname, sternum_mask_fname, pelvis_mask_fname = self.data[idx]
        ct = self._readBinImage(ct_fname)
        spine_mask = self._readUCharImage(spine_mask_fname)
        sternum_mask = self._readUCharImage(sternum_mask_fname)
        pelvis_mask = self._readUCharImage(pelvis_mask_fname)
        Kx, Ky = self.offset
        Lx, Ly = self.output_size
        if self.augment:
            kx = random.randint(0, Kx)
            ky = random.randint(0, Ky)
        else:
            kx, ky = 0, 0
        ct = ct[kx:kx+Lx, ky:ky+Ly]
        spine_mask = spine_mask[kx:kx+Lx, ky:ky+Ly]
        sternum_mask = sternum_mask[kx:kx+Lx, ky:ky+Ly]
        pelvis_mask = pelvis_mask[kx:kx+Lx, ky:ky+Ly]
        
        if self.augment and np.random.randn() > 0.5:
            ct = np.fliplr(ct).copy()
            spine_mask = np.fliplr(spine_mask).copy() 
            sternum_mask = np.fliplr(sternum_mask).copy()
            pelvis_mask = np.fliplr(pelvis_mask).copy()
            
        if self.augment and np.random.randn() > 0.5:
            ct = np.flipud(ct).copy()
            spine_mask = np.flipud(spine_mask).copy() 
            sternum_mask = np.flipud(sternum_mask).copy() 
            pelvis_mask = np.flipud(pelvis_mask).copy() 
            
        if self.augment and np.random.randn() > 0.5:
            ct = np.rot90(ct).copy()
            spine_mask = np.rot90(spine_mask).copy() 
            sternum_mask = np.rot90(sternum_mask).copy() 
            pelvis_mask = np.rot90(pelvis_mask).copy() 
        
        #torch.from_numpy(np.flip(x,axis=0).copy())
        ct = np.expand_dims(ct, axis=0)
        spine_mask = np.expand_dims(spine_mask, axis=0)
        sternum_mask = np.expand_dims(sternum_mask, axis=0)
        pelvis_mask = np.expand_dims(pelvis_mask, axis=0)
        masks = np.vstack((spine_mask, sternum_mask, pelvis_mask))
        
        return {'data': ct, 
                'label': masks}
    
    def __len__(self):
        return len(self.data)


class CTMultiClassDatasetImgaug(Dataset):
    def __init__(self, \
                 input_target_pairs, \
                 img_size=(512, 512), \
                 transform=None,
                 input_only=None):
        self.data = input_target_pairs
        self.img_size = img_size
        self.transform = transform
        self.input_only = input_only
  
    @staticmethod
    def readUCharImage(fname, img_size=(512, 512)):
        with open(fname, 'rb') as f:
            rawData = f.read()
            img = Image.frombytes('L', img_size, rawData)
            #return np.array(img).astype(np.float32)
            #return np.array(img)[..., np.newaxis].astype(np.uint8)
            return np.array(img).astype(np.uint8)
 
    @staticmethod
    def readBinImage(fname, img_size=(512, 512)):
        with open(fname, 'rb') as f:
            rawData = f.read()
            img = Image.frombytes('I', img_size, rawData)
            return np.array(img).T.astype(np.int16)

    @staticmethod
    def readFloatImage(fname, img_size=(512, 512)):
        with open(fname, 'rb') as f:
            rawData = f.read()
            img = Image.frombytes('F', img_size, rawData)
            return np.array(img).T.astype(np.float32)

    def _activator_masks(self, images, augmenter, parents, default):
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default

    def __getitem__(self, idx):
        ct_fname, spine_mask_fname, sternum_mask_fname, pelvis_mask_fname = self.data[idx]
        ct_image = self.readBinImage(ct_fname)
        spine_mask = self.readUCharImage(spine_mask_fname)
        sternum_mask = self.readUCharImage(sternum_mask_fname)
        pelvis_mask = self.readUCharImage(pelvis_mask_fname)

        if self.transform:
            det_tf = self.transform.to_deterministic()
            ct_image = det_tf.augment_image(ct_image)
            #[ct, spine_mask, sternum_mask, pelvis_mask] = det_tf.augment_images([ct, spine_mask, sternum_mask, pelvis_mask],
            #               hooks=ia.HooksImages(activator=self._activator_masks))
            spine_mask = det_tf.augment_image(
                            spine_mask,
                            hooks=ia.HooksImages(activator=self._activator_masks))
            sternum_mask = det_tf.augment_image(
                            sternum_mask,
                            hooks=ia.HooksImages(activator=self._activator_masks))
            pelvis_mask = det_tf.augment_image(
                            pelvis_mask,
                           hooks=ia.HooksImages(activator=self._activator_masks))

        ct = np.expand_dims(ct_image, axis=0).astype(np.float32)
        spine_mask = np.expand_dims(spine_mask, axis=0)
        sternum_mask = np.expand_dims(sternum_mask, axis=0)
        pelvis_mask = np.expand_dims(pelvis_mask, axis=0)
        masks = np.vstack((spine_mask, sternum_mask, pelvis_mask)).astype(np.float32)

        return {'data': ct, 
                'label': masks}

    def __len__(self):
        return len(self.data)