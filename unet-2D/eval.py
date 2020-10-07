import numpy as np
import torch
import torch.nn.functional as F

from predict import predict_vol_from_vol
from utils.utils import loadMatData

def eval_volume(net,
                device,
                vol_idx,
                p_threshold,
                mask = 'spine_mask'):
    """Takes a vol_idx in the form [patient_idx, day_idx] and evaluates a 
    prediction from a convnet model against the ground truth.

    Not yet working for multi-class evaluation. 
    
    @params:
    net: pytorch convnet model.
    device: pytorch device for computation.
    vol_idx: identifier for a patient data volume in the form [p, d].
    p_threshold: probability above which prediction is considered True.
    
    @return:
    dice, iou: returns both the dice and iou score.
    """
    mask_vol = loadMatData(vol_idx, data = mask)
    
    # create prediction volume
    pred_volume = predict_vol_from_vol(net,
                                       device,
                                       vol_idx,
                                       p_threshold = p_threshold)
    
    # calculate intersection over union between volumes
    intersection = np.logical_and(pred_volume, mask_vol).astype(int)
    union = np.logical_or(pred_volume, mask_vol).astype(int)
    iou = np.sum(intersection)/np.sum(union)

    # calculate dice coefficient between volumes
    a_volume = np.sum(pred_volume)
    b_volume = np.sum(mask_vol)
    dice = (2 * np.sum(intersection)) / (a_volume + b_volume)

    return dice, iou

def eval_volumes(net,
                device,
                vol_idxs,
                p_threshold = 0.5):
    """
    Wraps eval_volume to perform multiple evaluations given a list of vol_idxs.
    
    @params:
    net: pytorch convnet model.
    device: pytorch device for computation.
    vol_idx: identifier for a patient data volume in the form [p, d].
    p_threshold: probability above which prediction is considered True.
    
    @return:
    dice, iou: returns the average of the dice and iou for all vol_idxs.
    """
    dice_sum = 0
    iou_sum = 0
    for vol_idx in vol_idxs:
        dice, iou = eval_volume(net,
                                device,
                                vol_idx,
                                p_threshold)
        dice_sum = dice_sum + dice
        iou_sum = iou_sum + iou

    return dice_sum/len(vol_idxs), iou_sum/len(vol_idxs)