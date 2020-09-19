import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import random
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet
from losses import MixedLoss

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import CTMaskDataset
from utils.utils import matchFilesFromPatient
from torch.utils.data import DataLoader, random_split


def train_net(net,
              device,
              epochs = 10,
              batch_size = 1,
              lr = 0.0001,
              val_percent = 0.1,
              save_cp = True):

    dataset = CTMaskDataset(ct_data)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=8, 
                              pin_memory=True)
    val_loader = DataLoader(val, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=8, 
                            pin_memory=True, 
                            drop_last=True)
    
    global_step = 0
    writer = SummaryWriter(log_dir = dir_logging)
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    optimizer = optim.RMSprop(net.parameters(), 
                              lr=lr, 
                              weight_decay=1e-8, 
                              momentum=0.9)
    
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         'min', 
                                                         patience = 8)
    else:
        criterion = nn.BCEWithLogitsLoss()
        # criterion = MixedLoss(alpha = 10, gamma = 2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         'max', 
                                                         patience = 8)
    best_val_score = 0.0
    save_this_epoch = False

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, 
                  desc=f'Epoch {epoch + 1}/{epochs}', 
                  unit='img',
                  ascii = True,
                  leave = False,
                  bar_format='{l_bar}{bar:60}{r_bar}{bar:-10b}') as pbar:

            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['target']

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': round(loss.item(), 5)})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (5 * batch_size)) == 0:
                    ''' LOGGING OF WEIGHTS: DISABLED
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, 
                                             value.data.cpu().numpy(), 
                                             global_step)
                        writer.add_histogram('grads/' + tag, 
                                             value.grad.data.cpu().numpy(), 
                                             global_step)
                    '''
                    val_score, iou_score = eval_net(net, val_loader, device)
                    if val_score > best_val_score:
                        best_val_score = val_score
                        save_this_epoch = True

                    scheduler.step(val_score)

                    ''' LOGGING OF BASIC METRICS: ENABLED '''
                    writer.add_scalar('learning_rate', 
                                      optimizer.param_groups[0]['lr'], 
                                      global_step)
                    if net.n_classes > 1:
                        logging.info(f'Validation cross entropy: {val_score}')
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info(f'Validation Dice Coeff: {val_score}')
                        writer.add_scalar('Dice/test', val_score, global_step)
                        logging.info(f'Validation IoU Score: {iou_score}')
                        writer.add_scalar('IoU/test', iou_score, global_step)
                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', 
                                          torch.sigmoid(masks_pred) > 
                                                               0.5, global_step)

        if save_cp and save_this_epoch:
            torch.save(net.state_dict(),
                       dir_logging + f'CP_epoch.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
            save_this_epoch = False
    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=10,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    ############################################################################
    ### GET DATA FOR THE TRAINING AND VALIDATION DATASETS
    patient_idxs = range(1,22)
    ct_data = []
    for idx in patient_idxs:
        for day_selection in range(1,4):
            matched_data = matchFilesFromPatient(idx, 
                                                day_selection, 
                                                mode = 'CT_SPINE',
                                                no_empties = True)
            try:
                ct_data.extend(matched_data)
            except:
                print('[WARN] Did not find matching image data for Patient',
                      f'{idx} on Day Index {day_selection}')
                pass

    random.shuffle(ct_data)
    ############################################################################
    ### SET LOGGING AND MODEL CKPT DIRECTORIES
    subfolder = 'bce_spine_all'
    dt_string = datetime.now().strftime('%Y-%m-%d_%H.%M')
    dir_logging = 'unet-milesial/.runs/{}/{}/'.format(subfolder, 
                                                          dt_string)
    ############################################################################
    try:
        os.makedirs(dir_logging)
    except OSError:
        pass
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(dir_logging + "INFO.log"),
                                  logging.StreamHandler()])
    
    
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=1, n_classes=1, bilinear=False)
    
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), dir_logging + 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)