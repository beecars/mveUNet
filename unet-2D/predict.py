import numpy as np
from numpy.core.fromnumeric import shape
import torch
import torch.nn.functional as F
from tqdm.std import tqdm
from scipy.ndimage import rotate as imrot

from utils.utils import loadMatData

def predict_vol_from_vol(net,
                         device,
                         vol_idx,
                         threshold = True,
                         p_threshold = 0.5):
    """ Takes a vol_idx in the form [patient_idx, day_idx] and predicts a
    full-volume segmentation on a CNN model.
    
    @params:
    net : pytorch convnet model.
    device : pytorch device for computation.
    vol_idx : identifier for a patient data volume in the form [p, d].
    threshold : boolean for whether or not to threshold the output.
    p_threshold : probability above which prediction is considered True.
    
    @return:
    pred_volume : a prediction volume w/ shape: [n_classes, H, W, Z] 
    """
    net.eval()
    volume = loadMatData(vol_idx, data = 'ct')
    vol_shape = volume.shape
    n_cts = volume.shape[-1]

    pred_volume = torch.empty(net.n_classes, vol_shape[0], vol_shape[1], vol_shape[2])

    with tqdm(total = n_cts,   # progress bar
              desc = f'Predicting Volume', 
              unit = 'scans',
              ascii = True,
              leave = False,
              bar_format = '{l_bar}{bar:30}{r_bar}{bar:-10b}') as pbar:
    
        with torch.no_grad():
            for idx in range(n_cts):
                ct = torch.Tensor(volume[:, :, idx]).unsqueeze(0).unsqueeze(0)
                ct = ct.to(device=device, dtype=torch.float32)

                pred = net(ct) # output shape: (1, Classes, H, W)
                pred = torch.squeeze(pred) # out shape: (Classes, H, W)

                if net.n_classes > 1:
                    pred = F.softmax(pred, dim = 0)
                else:
                    pred = torch.sigmoid(pred)

                pred_volume[:, :, :, idx] = pred
                
                pbar.update()
            
        if threshold == True:
            pred_volume = pred_volume > p_threshold

    return pred_volume.numpy().astype(float)