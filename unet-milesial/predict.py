import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils import data
from torchvision import transforms
from tqdm.std import tqdm

from unet import UNet
from utils.dataset import CTVolumeDataset
from utils.utils import matchFilesFromPatient


def predict_volume(net,
                   device,
                   ct_data,
                   p_threshold = 0.5):
    net.eval()

    volume = CTVolumeDataset(ct_data)
    n_cts = len(volume)
    img_shape = volume[0]['image'].size()[1:3]
    probs_volume = torch.empty(n_cts, img_shape[0], img_shape[1])
    
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

                probs_volume[i] = probs
                
                pbar.update()

            mask_volume = probs_volume > p_threshold

    return mask_volume.numpy().astype(int)


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
    ### SET PREDICTION VOLUME DIRECTORY
    dir_save = 'unet-milesial/.predictions/'
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

    volume = predict_volume(net, device, ct_data)
    np.save(dir_save + args.filename, volume)

    logging.info(f"Prediction saved to {dir_save}{args.filename}")