from PIL import Image
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from utils import match_files_from_patient
from model import UNet
from losses import FocalLoss, MixedLoss, dice, IoU

import imgaug as iaa
from datetime import datetime
import pickle

def train_net(model, 
              device, 
              train_generator,
              batch_size, 
              criterion, 
              optimizer,
              learning_scheduler,
              epoch, 
              log_interval=100, 
              print_log=False):
          
    model.train()
    train_loss = []

    for batch_idx, batch_data in enumerate(train_generator):
        
        cts = batch_data['data']
        cts = cts.to(device)
        
        labels = batch_data['label']
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(cts)
        
        masks_probs_flat = outputs.view(-1)
        true_masks_flat = labels.view(-1)
        
        loss = criterion(masks_probs_flat, true_masks_flat)

        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())

        # print log message in terminal
        if print_log == True and batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_generator.dataset),
                100. * batch_idx / len(train_generator), loss.item()))
            
    learning_scheduler.step()

    return train_loss
###


def test_net(model, 
             device, 
             test_generator,  
             print_log=False):
    
    model.eval()
    
    dice_score = 0.0
    iou_score = 0.0
    correct = 0.0
    
    with torch.no_grad():
        for i, batch_data in enumerate(test_generator):
            cts = batch_data['data']
            cts = cts.to(device)

            labels = batch_data['label']
            labels = labels.to(device)

            outputs = model(cts)

            masks_probs = outputs

            dice_ = dice(masks_probs, labels)
            iou_ = IoU(masks_probs, labels)

            dice_score += dice_.item()
            iou_score += iou_.item()
            
    dice_score /= len(test_generator)
    iou_score /= len(test_generator)
    
    if print_log == True:
        print('\tTest set: dice score: {:.4f}, iou score: {:.4f}\n'.format(dice_score, iou_score))
    
    return dice_score, iou_score
###

if __name__ == '__main__':
    # code to run as callable training script goes below...'
    pass
