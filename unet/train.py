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

            #print(cts.shape, labels.shape)

            outputs = model(cts)
            #masks_probs = F.sigmoid(outputs)
            masks_probs = outputs
            #masks_probs_flat = masks_probs.view(-1)
            #true_masks_flat = labels.view(-1)

            #loss = criterion(masks_probs_flat, true_masks_flat)
            dice_ = dice(masks_probs, labels)
            iou_ = IoU(masks_probs, labels)
            #test_loss += loss.item()
            dice_score += dice_.item()
            iou_score += iou_.item()

    dice_score /= len(test_generator)
    iou_score /= len(test_generator)

    if print_log == True:
        print('\tTest set: dice score: {:.4f}, iou score: {:.4f}\n'.format(dice_score, iou_score))

    return dice_score, iou_score
###

if __name__ == '__main__':

    train_patient_idxs = [1, 2, 4, 5]
    train_dev_data = []
    for idx in train_patient_idxs:
        for day_selection in range(1,4):
            #if idx == 1 and day_selection == 2:
            #    pass
            #elif idx == 2 and day_selection == 1:
            #    pass
            if idx == 2 and day_selection == 2:
                pass
            elif idx == 2 and day_selection == 3:
                pass
            #elif idx == 4 and day_selection == 2:
            #    pass
            #elif idx == 5 and day_selection == 2:
            #    pass
            #elif idx == 5 and day_selection == 3:
            #    pass
            else:
                patient_data = get_ct_spine_mask_sternum_mask_pelvis_mask_data(idx, day_selection)
                train_dev_data.extend(patient_data)

    test_patient_idxs = [3, 6]
    test_data = []
    for idx in test_patient_idxs:
        for day_selection in range(1,4):
            if idx == 3 and day_selection == 2:
                pass
            elif idx == 6 and day_selection == 2:
                pass
            elif idx == 6 and day_selection == 3:
                pass
            else:
                patient_data = get_ct_spine_mask_sternum_mask_pelvis_mask_data(idx, day_selection)
                test_data.extend(patient_data)

    for line in train_dev_data:
        spine_fname = os.path.basename(line[1])
        sternum_fname = os.path.basename(line[2])
        if sternum_fname != 'empty.uchar' and spine_fname == 'empty.uchar':
            print(line)

    for line in test_data:
        spine_fname = os.path.basename(line[1])
        sternum_fname = os.path.basename(line[2])
        if sternum_fname != 'empty.uchar' and spine_fname == 'empty.uchar':
            print(line)

    augs = iaa.Sequential([
                        CenterCrop(height=320, width=320),
                        iaa.Fliplr(0.5),
                        iaa.Flipud(0.5),
                        iaa.Affine(rotate=(-45, 45),
                                   translate_percent={"x": (-0.2, 0.2), 
                                                      "y": (-0.2, 0.2)}),
                        iaa.GaussianBlur(sigma=(0, 1.0), name='gaussian-blur'),
                        iaa.AdditiveGaussianNoise(scale=(0, .05*255),
                                                  name='gaussian-noise'),
                        iaa.Add((-40, 40), per_channel=0.5, name="color-jitter")
                     ])

    dev_augs = iaa.Sequential([CenterCrop(height=320, width=320)])

    batch_size = 16
    #train_dataset = CTMultiClassDataset(train_dev_data)
    train_dataset = CTMultiClassDatasetImgaug(train_dev_data, img_size=(512, 512), transform=augs,
                                            input_only=['gaussian-blur', 'gaussian-noise', 'color-jitter'])
    train_generator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    dev_dataset = CTMultiClassDatasetImgaug(test_data, img_size=(512, 512), transform=dev_augs)
    dev_generator = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

    ##########################################
    current_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_directory = 'logs-' + current_datetime
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    lr_model = 0.001
    decay_step_size = 50 # decay every decay_step_size epochs.

    gamma = 2.0
    focal_gain = 10.0

    channel_in, num_classes = 1, 3
    ###########################################

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    model = UNet(channel_in, num_classes, large_model=False)
    model.to(device)
    criterion = MixedLoss(focal_gain, gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_model, weight_decay=1e-6)
    # Decay LR by a factor of 0.1 every step_size epochs
    model_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_step_size, gamma=0.1)

    num_epochs = 200
    ckpt_save_interval = 2

    train_losses = []
    val_losses = []
    best_score = 0.0
    for epoch in range(num_epochs):
        train_epoch_loss = train(model, \
                                model_lr_scheduler, \
                                train_generator, 
                                criterion, \
                                optimizer, \
                                epoch, \
                                device, \
                                log_interval=20, \
                                print_log=True)

        val_epoch_loss = test(model, \
                            dev_generator, \
                            criterion, \
                            epoch, \
                            device, \
                            print_log=True)

        if val_epoch_loss[0] > best_score:
            best_score = val_epoch_loss[0]
            torch.save(model.state_dict(), '{}/ckpt.model-{}-{}.pt'.format(output_directory, epoch, best_score))
        #if epoch % ckpt_save_interval == 0:
        #    torch.save(model.state_dict(), 'ckpt.model-{}.pt'.format(epoch))

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)


    with open('val-dice-iou-{}.pkl'.format(output_directory), 'wb') as f:
        pickle.dump(train_losses, f)
        pickle.dump(val_losses, f)