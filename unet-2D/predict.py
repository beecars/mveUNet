import argparse
import logging
import os

import numpy as np
from scipy.io import loadmat
import torch
import torch.nn.functional as F
from tqdm.std import tqdm

from unet import UNet
from utils.dataset import CTSequenceDataset, CTVolumeDataset
from utils.utils import matchFilesFromPatient


def predict_vol_from_seq(net,
                         device,
                         ct_data,
                         threshold = True,
                         p_threshold = 0.5):
    ''' 
    Takes a sequence (list) of CT scans, runs them through a CNN model, and 
    assembles and returns a prediction "volume" data object.
    @params:
        ct_data = a specific data list generated for the REVEAL CT data
                  by using the matchFilesFromPatients() function. 
        threshold = boolean, if False volume will be returned as float 
                    prediction values. If true, volume will be thresholded and
                    returned as a binary prediction mask.
        p_threshold = the threshold value above which a pixel is classified as 
                      true.
    @returns:
        a binary prediction volume or a probabily prediction volume as a nparray
    '''
    net.eval()

    volume = CTSequenceDataset(ct_data)
    n_cts = len(volume)
    img_shape = volume[0]['image'].size()[1:3]
    probs_volume = torch.empty(img_shape[0], img_shape[1], n_cts)
    
    with tqdm(total = n_cts, 
                  desc = f'Predicting Volume', 
                  unit = 'scans',
                  ascii = True,
                  leave = False,
                  bar_format = '{l_bar}{bar:30}{r_bar}{bar:-10b}') as pbar:

        with torch.no_grad():
            for i, scan in enumerate(volume):
                
                ct = scan['image'].unsqueeze(0)
                ct = ct.to(device)

                output = net(ct)
                output = torch.squeeze(output)

                if net.n_classes > 1:
                    probs = F.softmax(output, dim = 1)
                else:
                    probs = torch.sigmoid(output)

                probs_volume[:,:, i] = probs
                
                pbar.update()

        probs_volume = probs_volume > p_threshold

    return probs_volume.numpy().astype(float)
    
def predict_vol_from_vol(net,
                         device,
                         x,
                         mode = 'data',
                         threshold = True,
                         p_threshold = 0.5):
    '''
    Predicts a CT volume from a volume data source of one of the following: 
    a 3D numpy data array, a MATLAB .mat file, or a numpy .npy file.
    @params:
        x = either a 3D np array data OR a filepath, depending on mode.
        mode = 'npy', 'mat', or 'data' (default is data)
        threshold = boolean, if False volume will be returned as float 
                    prediction values. If true, volume will be thresholded and
                    returned as a binary prediction mask.
        p_threshold = the threshold value above which a pixel is classified as 
                      true.
    @returns:
        a binary prediction volume or a probabily prediction volume as a nparray
    '''
    volume = CTVolumeDataset(x, mode)
    n_cts = len(volume)
    img_shape = volume[0].shape
    probs_volume = torch.empty(img_shape[1], img_shape[2], n_cts)
    
    with tqdm(total = n_cts, 
                  desc = f'Predicting Volume', 
                  unit = 'scans',
                  ascii = True,
                  leave = False,
                  bar_format = '{l_bar}{bar:30}{r_bar}{bar:-10b}') as pbar:

        with torch.no_grad():
            for i, scan in enumerate(volume):
                
                ct = scan.unsqueeze(0)
                ct = ct.to(device)

                output = net(ct)
                output = torch.squeeze(output)

                if net.n_classes > 1:
                    probs = F.softmax(output, dim = 1)
                else:
                    probs = torch.sigmoid(output)

                probs_volume[:,:, i] = probs
                
                pbar.update()

        probs_volume = probs_volume > p_threshold

    return probs_volume.numpy().astype(float)
    
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
                        default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    ############################################################################
    ### SET PREDICTION VOLUME SAVE DIRECTORY
    dir_save = 'unet-2D/.predictions/'
    ############################################################################
    try:
        os.makedirs(dir_save)
    except OSError:
        pass

    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(message)s")

    ct_data = matchFilesFromPatient(args.patient, args.day, mode = 'CT_ONLY')
    
    net = UNet(args.channels, args.classes, bilinear = False).cuda()
    state = torch.load(args.state_dict)
    net.load_state_dict(state)
    device = torch.device('cuda')

    logging.info(f"State dict {args.state_dict} loaded to UNet model")
    logging.info(f"Predicting {args.classes} classes from {args.channels} channels")
    logging.info(f'Using device {device}')
    logging.info("Starting volume prediction...")

    volume = predict_vol_from_seq(net, device, ct_data)
    np.save(dir_save + args.filename, volume)

    logging.info(f"Prediction saved to {dir_save}{args.filename}")