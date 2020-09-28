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
from losses import FocalLoss, MixedLoss

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import CTMaskDataset, CTMulticlassDataset
from utils.utils import matchFilesFromPatients
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
                                                         patience = 8)
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
                                                         patience = 8)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=8, 
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=8, 
                            pin_memory=True, 
                            drop_last=True)
    
    n_train = len(train_dataset)
    n_val = len(val_dataset)
    writer = SummaryWriter(log_dir = dir_logging)
    
    if split == 1:
        logging.info(f'Training initialization:\n'
                     f'\tCross-Validation Split {split}/{folds}\n'
                     f'\tEpochs:          {epochs}\n'
                     f'\tBatch size:      {batch_size}\n'
                     f'\tLoss Function:   {criterion.__class__.__name__}\n'
                     f'\tOptimizer:       {optimizer.__class__.__name__}\n'
                     f'\tScheduler:       {scheduler.__class__.__name__}\n'
                     f'\tLearning rate:   {lr}\n'
                     f'\tCheckpoints:     {save_cp}\n'
                     f'\tDevice:          {device.type}')
    logging.info(f'\tCross-Validation Split {split}/{folds}\n'
                    f'\tTraining size:   {n_train}\n'
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
                
                if i == (len(train_loader) - 1):
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

# args to use if this file is called as a stand-alone script
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=10,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0003,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--folder', dest='folder', type=str, default='default',
                        help='Subfolder to output logs and model ckpts.')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    ############################################################################
    ### SET LOGGING DIRECTORY 
    ### Model checkpoint and interrupt also saved here.
    subfolder = args.folder
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
    ### GET DATA FOR THE CROSS-VALIDATION DATASETS.
    ### Cross-validation & training sets should be divided by patient.
    ### This could be done much more elegantly but I didn't want to spend 4 
    ### hours generalizing it...
    fold1 = [1, 2]            # 5 total volumes w/ spine seg.
    fold2 = [3, 6]            # 5 total volumes w/ spine seg.
    fold3 = [4, 7, 8]         # 5 total volumes w/ spine seg.
    fold4 = [5, 10, 11]       # 5 total volumes w/ spine seg.
    fold5 = [9, 12, 13]       # 5 total volumes w/ spine seg.
    fold6 = [14, 16, 17]      # 5 total volumes w/ spine seg.
    fold7 = [18, 19, 20]      # 5 total volumes w/ spine seg.

    val_data1, val1_idxs = matchFilesFromPatients(fold1, range(1,4))
    val_data2, val2_idxs = matchFilesFromPatients(fold2, range(1,4))
    val_data3, val3_idxs = matchFilesFromPatients(fold3, range(1,4))
    val_data4, val4_idxs = matchFilesFromPatients(fold4, range(1,4))
    val_data5, val5_idxs = matchFilesFromPatients(fold5, range(1,4))
    val_data6, val6_idxs = matchFilesFromPatients(fold6, range(1,4))
    val_data7, val7_idxs = matchFilesFromPatients(fold7, range(1,4))
    
    trn_split1 = fold2 + fold3 + fold4 + fold5 + fold6 + fold7
    trn_split2 = fold1 + fold3 + fold4 + fold5 + fold6 + fold7
    trn_split3 = fold1 + fold2 + fold4 + fold5 + fold6 + fold7
    trn_split4 = fold1 + fold2 + fold3 + fold5 + fold6 + fold7
    trn_split5 = fold1 + fold2 + fold3 + fold4 + fold6 + fold7
    trn_split6 = fold1 + fold2 + fold3 + fold4 + fold5 + fold7
    trn_split7 = fold1 + fold2 + fold3 + fold4 + fold5 + fold6

    train_data1, _ = matchFilesFromPatients(trn_split1, range(1,4))
    train_data2, _ = matchFilesFromPatients(trn_split2, range(1,4))
    train_data3, _ = matchFilesFromPatients(trn_split3, range(1,4))
    train_data4, _ = matchFilesFromPatients(trn_split4, range(1,4))
    train_data5, _ = matchFilesFromPatients(trn_split5, range(1,4))
    train_data6, _ = matchFilesFromPatients(trn_split6, range(1,4))
    train_data7, _ = matchFilesFromPatients(trn_split7, range(1,4))

    val_datas = [val_data1, val_data2, val_data3, val_data4, 
                 val_data5, val_data6, val_data7]
    
    val_idxs = [val1_idxs, val2_idxs, val3_idxs, val4_idxs, 
                val5_idxs, val6_idxs, val7_idxs]

    train_datas = [train_data1, train_data2, train_data3, train_data4, 
                   train_data5, train_data6, train_data7]
    ############################################################################
    ### For training on all data (testing only)
    # trn_all = fold1 + fold2 + fold3 + fold4 + fold5 + fold6 + fold7
    # train_data, _ = matchFilesFromPatients(trn_all, range(1,4))
    # train_datas = [train_data]
    # val_data, val_idxs = matchFilesFromPatients(fold1, range(1,4))
    # val_datas = [val_data]
    # val_idxs = [val_idxs]
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
                      epochs = args.epochs,
                      batch_size = args.batchsize,
                      lr = args.lr,
                      folds = 7,
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
        