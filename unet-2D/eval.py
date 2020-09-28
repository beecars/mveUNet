import numpy as np
import torch
import torch.nn.functional as F
from tqdm.std import tqdm

from predict import predict_vol_from_seq
from utils.dataset import readUCharImage
from losses import dice_coeff, iou


def eval_volume(net,
                device,
                ct_mask_data,
                p_threshold = 0.5):
    '''
    From a sequence of CT images, predicts a segmented "volume" and compares
    it with the ground truth. Returns <metrics>.
    @params:
        ct_mask_data = a specific data list generated for the REVEAL CT data
                        by using the matchFilesFromPatients() function. 
        p_threshold = the value above which a pixel is classified as true.
    @returns: 
        a tuple of (dice, iou) metrics.
    '''
    ct_mask_data = np.array(ct_mask_data)
    ct_data = ct_mask_data[:,0]
    mask_data = ct_mask_data[:,1]

    # create prediction volume
    pred_volume = predict_vol_from_seq(net,
                                       device,
                                       ct_data,
                                       p_threshold = p_threshold)
    
    # create ground-truth volume
    mask_shape = readUCharImage(mask_data[0]).shape
    test_volume = np.zeros((mask_shape[0], mask_shape[1], len(mask_data)))
    for i, file in enumerate(mask_data):
        test_volume[:, :, i] = readUCharImage(file)
    test_volume = test_volume.astype(int)

    # calculate intersection over union between volumes
    intersection = np.logical_and(pred_volume, test_volume).astype(int)
    union = np.logical_or(pred_volume, test_volume).astype(int)
    iou = np.sum(intersection)/np.sum(union)

    # calculate dice coefficient between volumes
    a_volume = np.sum(pred_volume)
    b_volume = np.sum(test_volume)
    dice = (2 * np.sum(intersection)) / (a_volume + b_volume)

    return dice, iou

def eval_volumes(net,
                 device,
                 ct_mask_data,
                 volume_idxs,
                 p_threshold = 0.5):
    '''
    From an ordered list representing sequences of CT images of more than one
    CT volume, computes metrics over the individual volumes represented
    in the input data list and returns the average of those metrics.
    @params:
        ct_mask_data = a specific data list generated for the REVEAL CT data
                        by using the matchFilesFromPatients() function. 
        volume_idx = a specific list of (patient, day) tuples generated for 
                        REVEAL CT data by using the matchFilesFromPatients() 
                        function.
        p_threshold = the value above which a pixel is classified as true.
    @returns: 
        a tuple of (dice, iou) metrics.
    '''
    dice_sum = 0
    iou_sum = 0
    for volume_idx in volume_idxs:
        temp_data = []
        for line in ct_mask_data:                                   # build
            if ('P_' + '{0:0=2d}'.format(volume_idx[0]) in line[0]  # individual
                and f'Day_{volume_idx[1]}' in line[0]):             # volume
                temp_data.append(line)                              # subset
            
        dice, iou = eval_volume(net,
                                device,
                                temp_data,
                                p_threshold = p_threshold)
        dice_sum = dice_sum + dice
        iou_sum = iou_sum + iou

    return dice_sum/len(volume_idxs), iou_sum/len(volume_idxs)