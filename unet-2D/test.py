import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
from tqdm.std import tqdm

from unet import UNet
from predict import predict_vol_from_seq
from utils.dataset import CTVolumeDataset, readUCharImage
from utils.utils import matchFilesFromPatient

def test_volume(net,
                device,
                ct_mask_data,
                p_threshold = 0.5):
    '''From a sequence of CT images, predicts a segmented "volume" and compares
    it with the ground truth. Returns <metrics>.
    '''
    ct_mask_data = np.array(ct_mask_data)
    ct_data = ct_mask_data[:,0]
    mask_data = ct_mask_data[:,1]

    pred_volume = predict_vol_from_seq(net,
                                       device,
                                       ct_data)
    
    mask_shape = readUCharImage(mask_data[0]).shape
    voxel_count = mask_shape[0] * mask_shape[1] * len(mask_data)
    test_volume = np.zeros((mask_shape[0], mask_shape[1], len(mask_data)))

    for i, file in enumerate(mask_data):
        test_volume[:, :, i] = readUCharImage(file)

    test_volume = test_volume.astype(int)

    xnor_volume = np.logical_not(np.logical_xor(pred_volume, test_volume)).astype(int)
    accuracy = np.sum(xnor_volume)/voxel_count

    intersection = np.logical_and(pred_volume, test_volume).astype(int)
    union = np.logical_or(pred_volume, test_volume).astype(int)
    iou = np.sum(intersection)/np.sum(union)

    return {'acc': accuracy,
            'iou': iou}


def get_args():
    parser = argparse.ArgumentParser(description='Predict mask volume from input volume',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--patient', '-p', metavar='INPUT', type=int,
                        help="Specify the patient index.", required = True)
    parser.add_argument('--day', '-d', metavar='INPUT', type=int,
                        help="Specify the day index.", required=True)
    parser.add_argument('--classes', '-cl', metavar='INPUT', default=1,
                        help="Specify the number of classes.", required=False)
    parser.add_argument('--channels', '-ch', metavar='INPUT', default=1,
                        help="Specify the number of channels.", required=False)
    parser.add_argument('--state_dict', '-s', metavar='INPUT', required=True,
                        help="Filename of state dict (.pth)")
    parser.add_argument('--filename', '-f', metavar='INPUT', required=False,
                        help="Filename of ouput file (.npy)", default='volume.npy')
    parser.add_argument('--mask-threshold', '-t', metavar = 'INPUT', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5, required = False)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(message)s")

    ct_mask_data = matchFilesFromPatient(args.patient, args.day, mode = 'CT_SPINE')
    
    net = UNet(args.channels, args.classes, bilinear = False).cuda()
    state = torch.load(args.state_dict)
    net.load_state_dict(state)
    device = torch.device('cuda')
    logging.info(f"Calculating metrics from patient {args.patient} day index {args.day}")
    logging.info(f"Loading state dict {args.state_dict} to UNet model")
    logging.info(f"Predicting {args.classes} classes from {args.channels} channels")
    logging.info(f"Using device {device}")
    
    metrics = test_volume(net,
                          device,
                          ct_mask_data,
                          p_threshold = args.mask_threshold)

    logging.info(f"Results: {metrics}")