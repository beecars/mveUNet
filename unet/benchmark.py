import sys
from PIL import Image
import numpy as np
import scipy.io as scio
import glob
import torch

import torch.nn as nn
import torch.nn.functional as F
import os
import random
import librosa
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
# import matplotlib.pyplot as plt


from utils import get_ct_mask_data
from dataset import CTMaskDataset
from model import UNet
from losses import FocalLoss

from datetime import datetime
import pickle

train_patient_idxs = [1, 2, 4, 5]
train_dev_data = []
for idx in train_patient_idxs:
    for day_selection in range(1,4):
        patient_data = get_ct_mask_data(idx, day_selection)
        train_dev_data.extend(patient_data)


test_patient_idxs = [3, 6]
test_data = []
for idx in test_patient_idxs:
    for day_selection in range(1,4):
        patient_data = get_ct_mask_data(idx, day_selection)
        test_data.extend(patient_data)


seed = 544
K = int(0.1 * len(train_dev_data))
np.random.shuffle(train_dev_data)
dev_data = train_dev_data[:K]
train_data = train_dev_data[K:]
print('train: {}, dev: {}, test: {}'.format(len(train_data), len(dev_data), len(test_data)))


batch_size = 8
train_dataset = CTMaskDataset(train_data)
train_generator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
dev_dataset = CTMaskDataset(dev_data)
dev_generator = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, num_workers=6)



def train(model, device, train_loader,
          criterion, optimizer,
          epoch, log_interval=100, print_log=False):
    model.train()
    train_loss = []
    for batch_idx, batch_data in enumerate(train_loader):
        cts = batch_data['data']
        cts = cts.to(device)

        labels = batch_data['label']
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(cts)
        #print(outputs.size())
        masks_probs = F.sigmoid(outputs)
        masks_probs_flat = masks_probs.view(-1)
        true_masks_flat = labels.view(-1)
        #print(masks_probs_flat.size(), true_masks_flat.size())

        loss = criterion(masks_probs_flat, true_masks_flat)

        loss.backward()
        optimizer.step()


        train_loss.append(loss.item())

        if print_log == True and batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    model_lr_scheduler.step()
    return train_loss

def test(model, device, test_loader,  criterion, epoch, print_log=False):
    model.eval()
    test_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            cts = batch_data['data']
            cts = cts.to(device)

            labels = batch_data['label']
            labels = labels.to(device)

            outputs = model(cts)
            masks_probs = F.sigmoid(outputs)
            masks_probs_flat = masks_probs.view(-1)
            true_masks_flat = labels.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    if print_log == True:
        print('\tTest set: Average loss: {:.4f}\n'.format(test_loss))
    return test_loss




#gammas = [0.0, 0.5, 1.0, 2.0, 4.0]
gamma = float(sys.argv[1])

train_results = dict()

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
lr_model = 0.001
decay_step_size = 500

#gamma = 0.1
model = UNet(1,1)
model.to(device)
criterion = FocalLoss(gamma=gamma, alpha=4.0)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_model)
# Decay LR by a factor of 0.1 every step_size epochs
model_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_step_size, gamma=0.1)

num_epochs = 40
ckpt_save_interval = 2

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gamma_train_loss = []
gamma_val_loss = []
print('Gamma: {}'.format(gamma))
for epoch in range(num_epochs):
    train_epoch_loss = train(model, device, train_generator,
                             criterion, optimizer, epoch, log_interval=10)

    val_epoch_loss = test(model, device, dev_generator, criterion, epoch)

    #if epoch % ckpt_save_interval == 0:
    #    torch.save(model.state_dict(), 'ckpt.model-{}.pt'.format(epoch))

    gamma_train_loss.append(train_epoch_loss)
    gamma_val_loss.append(val_epoch_loss)
    train_results[str(gamma)] = (gamma_train_loss, gamma_val_loss)


f = open("train-gamma-{}.pkl".format(gamma),"wb")
pickle.dump(train_results,f)
f.close()
