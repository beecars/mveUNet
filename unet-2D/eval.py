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
                mask_names,
                p_threshold = 0.5): 
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
    # create prediction volume
    pred_volume = predict_vol_from_vol(net,
                                       device,
                                       vol_idx,
                                       p_threshold = p_threshold)
    
    # create dicts to store performance metrics across classes
    ious = {mask_name : 0 for mask_name in mask_names}
    dices = {mask_name : 0 for mask_name in mask_names}
    # iterate through the classes and evaluate
    for i, mask_name in enumerate(mask_names):
        if net.n_classes > 1:
            # pred_vol[i+1] is the prediction of the true masks[i] for multiclass
            class_idx = i + 1
        else:
            class_idx = 0
        true_mask = loadMatData(vol_idx, data = mask_name)
        # calculate intersection over union between volumes from all classes
        iou = jaccard(pred_volume[class_idx], true_mask)
        ious[mask_name] = iou
        # calculate dice coefficient between volumes
        dice = dice_coeff(pred_volume[class_idx], true_mask)
        dices[mask_name] = dice

    return ious, dices

def eval_volumes(net,
                device,
                vol_idxs,
                mask_names = ['spine_mask', 'stern_mask', 'pelvi_mask'],
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
    # intialize dicts to hold scores for averaging
    dice_sum = {mask_name : 0 for mask_name in mask_names}
    dice_avgs = {mask_name : 0 for mask_name in mask_names}
    iou_sum = {mask_name : 0 for mask_name in mask_names}
    iou_avgs = {mask_name : 0 for mask_name in mask_names}
    for vol_idx in vol_idxs:
        dices, ious = eval_volume(net,
                                device,
                                vol_idx,
                                mask_names,
                                p_threshold = p_threshold)
        # add the score of each evaluated volume to the sum
        for mask_name in mask_names:
            dice_sum[mask_name] = dice_sum[mask_name] + dices[mask_name]
            iou_sum[mask_name] = iou_sum[mask_name] + ious[mask_name]
        # compute the average scores
        for mask_name in mask_names:
            dice_avgs[mask_name] = dice_sum[mask_name]/len(vol_idxs)
            iou_avgs[mask_name] = iou_sum[mask_name]/len(vol_idxs)

    return dice_avgs, iou_avgs