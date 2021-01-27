import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from predict import predict_vol_from_vol
from utils.utils import loadMatData, plotSomeImages
from unet.unet_model import UNet


def plot_patient_SUV_3_scans_spine_stern_pelvi(patient_idx, net, device):

    pt_scan1 = loadMatData([patient_idx, 1], data = 'pt')
    pt_scan2 = loadMatData([patient_idx, 2], data = 'pt')
    pt_scan3 = loadMatData([patient_idx, 3], data = 'pt')

    vol_mask1 = predict_vol_from_vol(net, device, [patient_idx, 1]).astype(bool)
    vol_mask2 = predict_vol_from_vol(net, device, [patient_idx, 2]).astype(bool)
    vol_mask3 = predict_vol_from_vol(net, device, [patient_idx, 3]).astype(bool)

    spine_suvs = {'Scan 1': pt_scan1[vol_mask1[1]],
                  'Scan 2': pt_scan2[vol_mask2[1]],
                  'Scan 3': pt_scan3[vol_mask3[1]]}

    stern_suvs = {'Scan 1': pt_scan1[vol_mask1[2]],
                  'Scan 2': pt_scan2[vol_mask2[2]],
                  'Scan 3': pt_scan3[vol_mask3[2]]}

    pelvi_suvs = {'Scan 1': pt_scan1[vol_mask1[3]],
                  'Scan 2': pt_scan2[vol_mask2[3]],
                  'Scan 3': pt_scan3[vol_mask3[3]]}

    fig = plt.figure(figsize = (20, 6))
    gs = fig.add_gridspec(ncols = 3, nrows = 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    fig.suptitle("SUV Distributions for ROI's (Patient " + str(patient_idx) + ")", size = 16)

    sns.violinplot(ax = ax1, data = [spine_suvs[key] for key in spine_suvs], width = .5, scale = "width")
    ax1.set_title("Spine ROI", size = 14)
    ax1.set_xticklabels(["Scan 1", "Scan 2", "Scan 3"], size = 12)

    sns.violinplot(ax = ax2, data = [stern_suvs[key] for key in spine_suvs], width = .5, scale = "width")
    ax2.set_title("Sternum ROI", size = 14)
    ax2.set_xticklabels(["Scan 1", "Scan 2", "Scan 3"], size = 12)

    sns.violinplot(ax = ax3, data = [pelvi_suvs[key] for key in spine_suvs], width = .5, scale = "width")
    ax3.set_title("Pelvis ROI", size = 14)
    ax3.set_xticklabels(["Scan 1", "Scan 2", "Scan 3"], size = 12)