import numpy as np
import torch
import torch.nn.functional as F

from predict import predict_vol_from_vol
from utils.utils import loadMatData

def jaccard(prediction_vol, truth_vol):
    """Calculate jaccard/intersection-over-union between mask volumes."""
    intersection = np.logical_and(prediction_vol, truth_vol).astype(int)
    union = np.logical_or(prediction_vol, truth_vol).astype(int)
    iou = np.sum(intersection)/np.sum(union)
    return iou

def dice_coeff(prediction_vol, truth_vol):
    """Calculate dice coefficient between mask volumes."""
    intersection = np.logical_and(prediction_vol, truth_vol).astype(int)
    a_volume = np.sum(prediction_vol)
    b_volume = np.sum(truth_vol)
    dice = (2 * np.sum(intersection)) / (a_volume + b_volume)
    return dice

def sensitivity(prediction_vol, truth_vol):
    """Calculate the sensitivity between a prediction mask and ground truth.
    a.k.a. True Positive Rate
    a.k.a. Recall
    """
    true_positives = np.sum(np.logical_and(prediction_vol, truth_vol).astype(int))
    truth_positives = np.sum(truth_vol.astype(int))
    return true_positives/truth_positives

def specificity(prediction_vol, truth_vol):
    """Calculate the specificity between a prediction mask and ground truth.
    a.k.a. True Negative Rate
    a.k.a. Selectivity
    """
    inverse_truth = np.logical_not(truth_vol).astype(int)
    true_negatives = np.sum(np.logical_and(prediction_vol, inverse_truth).astype(int))
    truth_negatives = np.sum(inverse_truth)
    return true_negatives/truth_negatives

def get_metrics(prediction_vol, truth_vol):
    metrics_dict = {}
    metrics_dict['jaccard'] = jaccard(prediction_vol, truth_vol)
    metrics_dict['dice_coeff'] = dice_coeff(prediction_vol, truth_vol)
    metrics_dict['sensitivity'] = sensitivity(prediction_vol, truth_vol)
    metrics_dict['specificity'] = specificity(prediction_vol, truth_vol)
    return metrics_dict

def eval_volume(net,
                device,
                vol_idx,
                p_threshold = 0.5,
                mask = 'spine_mask'): 
    """ Takes a vol_idx in the form [patient_idx, day_idx] and evaluates a 
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
    iou = jaccard(pred_volume, mask_vol)

    # calculate dice coefficient between volumes
    dice = dice_coeff(pred_volume, mask_vol)

    return dice, iou

def eval_volumes(net,
                device,
                vol_idxs,
                p_threshold = 0.5):
    """ Wraps eval_volume to perform multiple evaluations given a list of 
    vol_idxs.
    
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