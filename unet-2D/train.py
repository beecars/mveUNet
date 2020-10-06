import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_volumes
from unet import UNet
from losses import FocalLoss

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import CTMaskDataset, CTMulticlassDataset
from utils.utils import matchFilesFromPatients, generateCrossvalidationSets
from torch.utils.data import DataLoader


def train_net(net,
              device,
              train_data,
              val_data,
              val_idxs,
              epochs = 10,
              batch_size = 1,
              lr = 0.0001,
              save_cp = True,
              folds = 1,
              current_split = 0):

    split = current_split + 1
    if net.n_classes > 1:   # for multiclass training
        train_dataset = CTMulticlassDataset(train_data)
        val_dataset = CTMulticlassDataset(val_data)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.RMSprop(net.parameters(), 
                                  lr = lr, 
                                  weight_decay = 1e-8, 
                                  momentum = 0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         'min', 
                                                         patience = 5)
    else:   # for single class training
        train_dataset = CTMaskDataset(train_data)
        val_dataset = CTMaskDataset(val_data)
        
        criterion = nn.BCEWithLogitsLoss()
        # criterion = FocalLoss(alpha = 1, gamma = 2)
        # criterion = MixedLoss(alpha = 10, gamma = 2)
        optimizer = optim.RMSprop(net.parameters(), 
                                  lr = lr, 
                                  weight_decay = 1e-8, 
                                  momentum = 0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         'max', 
                                                         patience = 5)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=1, 
                              pin_memory=True)
    # val_loader... fear not, the val_loader lives inside the eval_volumes 
    #               fucntion. why?
    
    n_train = len(train_dataset)
    n_val = len(val_dataset)
    writer = SummaryWriter(log_dir = dir_logging)
    
    if split == 1:
        logging.info(f'Training initialization:\n'
                     f'\tEpochs:          {epochs}\n'
                     f'\tBatch size:      {batch_size}\n'
                     f'\tLoss Function:   {criterion.__class__.__name__}\n'
                     f'\tOptimizer:       {optimizer.__class__.__name__}\n'
                     f'\tScheduler:       {scheduler.__class__.__name__}\n'
                     f'\tLearning rate:   {lr}\n'
                     f'\tCheckpoints:     {save_cp}\n'
                     f'\tDevice:          {device.type}\n'
                     f'\tTraining size:   {n_train}\n'
                     f'\tValidation size: {n_val}')
    if folds > 1: # if running as "leave some out" these wont appear
        logging.info(f'\tCross-Validation Split {split}/{folds}')
        if split > 1: # uninspired logic to avoid info redundancy in 1st split
            logging.info(f'\tTraining size:   {n_train}\n'
                         f'\tValidation size: {n_val}')

    global_step = 0

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total = n_train,    # progress bar
                  desc = f'Epoch {epoch + 1}/{epochs}', 
                  unit = 'img',
                  ascii = True,
                  leave = False,
                  bar_format = '{l_bar}{bar:60}{r_bar}{bar:-10b}') as pbar:

            for i, batch in enumerate(train_loader):
                imgs = batch['image']
                true_masks = batch['target']

                # load image and mask to device
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                # forward pass image through model
                masks_pred = net(imgs)
                
                # calcululate and log loss
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar(f'split_{split}/train_loss', 
                                  loss.item(), 
                                  global_step)
                pbar.set_postfix(**{'loss (batch)': round(loss.item(), 5)})

                # propogate the loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                # update the progress bar
                pbar.update(imgs.shape[0])
                global_step += 1
                
                if ((i == (len(train_loader) - 1))
                    or (i == round(len(train_loader)/3))
                    or (i == round(len(train_loader)*2/3))):
                    # validation round
                    dice, iou = eval_volumes(net, 
                                            device,
                                            val_data,
                                            val_idxs,
                                            p_threshold = 0.5)
                    # step through learning sheduler
                    scheduler.step(iou)
                    
                    # log learning rate
                    writer.add_scalar(f'split_{split}/learning_rate', 
                                      optimizer.param_groups[0]['lr'], 
                                      global_step)
                    
                    # log validation metrics
                    if net.n_classes > 1:
                        writer.add_scalar(f'split_{split}/validation/cross_entropy', 
                                          iou, 
                                          global_step)
                    else:
                        writer.add_scalar(f'split_{split}/validation/dice', 
                                          dice, 
                                          global_step)
                        writer.add_scalar(f'split_{split}/validation/iou', 
                                          iou, 
                                          global_step)
    if save_cp:
        torch.save(net.state_dict(),
                   dir_logging + f'model_state_split{split}.pth')
        logging.info(f'Saved model state at end of split {split}')
    writer.close()


if __name__ == '__main__':
    ############################################################################
    ### SET LOGGING DIRECTORY 
    ### Model checkpoint and interrupt also saved here.
    subfolder = 'test'
    dt_string = datetime.now().strftime('%Y-%m-%d_%H.%M')
    dir_logging = 'unet-2D/.runs/{}/{}/'.format(subfolder, 
                                                          dt_string)
    try:
        os.makedirs(dir_logging)
    except OSError:
        pass

    logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(dir_logging + "INFO.log"),
                                logging.StreamHandler()])
    ############################################################################
    ### FOR CROSS VALIDATION
    # val_datas, val_idxs, train_datas = generateCrossvalidationSets()
    # folds = 7
    ############################################################################
    ### FOR LEAVE ONE/SOME OUT
    train_patients = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                      18, 19, 20, 21]
    val_patients = [17]
    train_datas, _ = matchFilesFromPatients(train_patients, range(1,4))
    val_datas, val_idxs = matchFilesFromPatients(val_patients, range(1,4))
    train_datas, val_datas, val_idxs = [train_datas], [val_datas], [val_idxs]
    folds = 1
    ############################################################################

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=1, n_classes=1, bilinear=False)
    # save state for model re-initialization during cross-validation
    torch.save(net.state_dict(), dir_logging + 'intial_state.pt')
    
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    net.to(device=device)

    validation_scores = []
    for split in range(len(val_datas)):
        # reload initial weights
        state = torch.load(dir_logging + 'intial_state.pt')
        net.load_state_dict(state)
        try:
            train_net(net = net,
                      device = device,
                      train_data = train_datas[split],
                      val_data = val_datas[split],
                      val_idxs = val_idxs[split],
                      epochs = 10,
                      batch_size = 6,
                      lr = 0.0003,
                      folds = folds,
                      current_split = split)
            
        except KeyboardInterrupt:
            torch.save(net.state_dict(), dir_logging + 'INTERRUPTED.pt')
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
    # remove inital state
    os.remove(dir_logging + 'intial_state.pt')
        