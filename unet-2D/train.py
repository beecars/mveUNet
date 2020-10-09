import argparse
import logging
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_volumes
from unet import UNet
from losses import FocalLoss

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import CTMaskDataset
from utils.utils import generateNpySlices, generateSplits, getScanCount
from torch.utils.data import DataLoader


def train_net(net,
              device,
              train_idxs,
              val_idxs,
              epochs = 10,
              batch_size = 1,
              lr = 0.0003,
              save_cp = True,
              folds = 1,
              current_split = 0):

    generateNpySlices(train_idxs)

    split = current_split + 1
    if net.n_classes > 1:   # for multiclass training     
        train_dataset = CTMaskDataset()      
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.RMSprop(net.parameters(), 
                                  lr = lr, 
                                  weight_decay = 1e-8, 
                                  momentum = 0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         'min', 
                                                         patience = 5)
    else:   # for single class training
        train_dataset = CTMaskDataset()     
        criterion = nn.BCEWithLogitsLoss()
        # criterion = FocalLoss(alpha = 1, gamma = 2)
        optimizer = optim.AdamW(net.parameters(), 
                                lr = lr, 
                                weight_decay = 0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         'max', 
                                                         patience = 5)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=8, 
                              pin_memory=True)
    # val_loader... fear not, the val_loader lives inside the eval_volumes 
    #               fucntion. why? that's a long story.
    
    n_train = len(train_dataset)
    n_val = getScanCount(val_idxs)
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
                imgs = batch['ct']
                true_masks = batch['spine']

                # load image and mask to device
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                # forward pass image through model
                pred_masks = net(imgs)
                
                # calcululate and log loss
                loss = criterion(pred_masks, true_masks)
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
                    net.eval()  
                    dice, iou = eval_volumes(net, 
                                             device,
                                             val_idxs,
                                             p_threshold = 0.5)
                    # step through learning sheduler
                    scheduler.step(iou)
                    # set net back to train mode
                    net.train()
                   
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
    dir_logging = 'unet-2D/.runs/{}/{}/'.format(subfolder, dt_string)
    try:
        os.makedirs(dir_logging)
    except OSError:
        pass

    logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(dir_logging + "INFO.log"),
                                logging.StreamHandler()])
    ############################################################################
    ### MAKE TRAINING/VALIDATION SPLIT
    all_idxs = [[a , b] for b in range(1,4) for a in range(1,23)]
    val_idxs, trn_idxs = generateSplits(all_idxs)
    # trn_idxs = [[17, 3], [9, 3], [1, 1]]
    # val_idxs = [[16, 3]]
    # for compatibility with the cross training nature...
    val_idxs, trn_idxs = [val_idxs], [trn_idxs]
    splits = 1
    logging.info('TESTING... ATTENTION PLEASE')
    logging.info('Validataion Volumes: ' + str(val_idxs))
    logging.info('Training Volumes: ' + str(trn_idxs))
    ############################################################################

    device = torch.device('cuda')
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
    for split in range(splits):
        # reload initial weights
        state = torch.load(dir_logging + 'intial_state.pt')
        net.load_state_dict(state)
        try:
            train_net(net,
                      device,
                      trn_idxs[split],
                      val_idxs[split],
                      epochs = 10,
                      batch_size = 6,
                      lr = 0.0003,
                      save_cp = True,
                      folds = 1,
                      current_split = 0)
            
        except KeyboardInterrupt:
            torch.save(net.state_dict(), dir_logging + 'INTERRUPTED.pt')
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
    # remove inital state
    os.remove(dir_logging + 'intial_state.pt')
        