import sys
import os
import logging
from shutil import rmtree
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from eval import eval_volumes
from unet import UNet
from losses import FocalLoss

from utils.dataset import CTMaskDataset
from utils.utils import generateNpySlices, generateSplits, getScanCount


def train_net(net,
              device,
              train_idxs,
              val_idxs,
              epochs = 10,
              batch_size = 1,
              lr = 0.0003,
              save_cp = True,
              splits = 1,
              current_split = 0):

    # generateNpySlices(train_idxs, plane = 'axial')

    split = current_split + 1
    if net.n_classes > 1:   # for multiclass training
        train_dataset = CTMaskDataset()      
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.RMSprop(net.parameters(), 
                                  lr = lr, 
                                  weight_decay = 1e-8, 
                                  momentum = 0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         'max', 
                                                         patience = 8)
    else:   # for single class training
        train_dataset = CTMaskDataset()     
        # criterion = nn.BCEWithLogitsLoss()
        criterion = FocalLoss()
        optimizer = optim.AdamW(net.parameters(), 
                                lr = lr, 
                                weight_decay = .0)
        # optimizer = optim.RMSprop(net.parameters(), 
        #                           lr = lr, 
        #                           weight_decay = 0, 
        #                           momentum = 0)
        # lrmultiply = lambda epoch: 10**(epoch/5)
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer, 
        #                                         lr_lambda = lrmultiply)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
        #                                                  'max',
        #                                                  patience = 8)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=8, 
                              pin_memory=True)

    
    
    n_train = len(train_dataset)
    n_val = getScanCount(val_idxs)
    writer = SummaryWriter(log_dir = dir_logging)
    
    if split == 1:
        logging.info(f'Training initialization:\n'
                     f'\tEpochs:                {epochs}\n'
                     f'\tBatch size:            {batch_size}\n'
                     f'\tLoss Function:         {criterion.__class__.__name__}\n'
                     f'\tOptimizer:             {optimizer.__class__.__name__}\n'
                     f'\tOptimizer Args:        {optimizer.defaults}\n'
     
                     f'\tLearning rate:         {lr}\n'
                     f'\tCheckpoints:           {save_cp}\n'
                     f'\tDevice:                {device.type}\n'
                     f'\tTraining size:         {n_train}\n'
                     f'\tValidation size:       {n_val}\n'
                     f'\tValidataion Volumes:   {val_idxs}\n'
                     f'\tTraining Volumes:      {train_idxs}')
    if splits > 1:
        logging.info(f'Cross-Validation Split   {split}/{splits}\n'
                        f'\tTraining size:         {n_train}\n'
                        f'\tValidation size:       {n_val}\n'
                        f'\tValidataion Volumes:   {val_idxs}\n'
                        f'\tTraining Volumes:      {train_idxs}')
    
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
                writer.add_scalar(f'split_{split}/train_loss', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': round(loss.item(), 5)})

                # propogate the loss
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                # update the progress bar
                pbar.update(imgs.shape[0])
                global_step += 1
                
                if i in [round(len(train_loader)*batch_num/10) for batch_num in range(1,10)]:
                    # validation round
                    net.eval()  
                    dice, iou = eval_volumes(net, 
                                             device,
                                             val_idxs,
                                             p_threshold = 0.5)
                   
                    # step through learning sheduler
                    # scheduler.step()

                    # log learning rate
                    writer.add_scalar(f'split_{split}/learning_rate', 
                                        optimizer.param_groups[0]['lr'], 
                                        global_step)

                    # set net back to train mode
                    net.train()
                
                    # log validation metrics
                    if net.n_classes > 1:
                        writer.add_scalar(f'split_{split}/validation/cross_entropy', iou, global_step)
                    else:
                        writer.add_scalar(f'split_{split}/validation/dice', dice, global_step)
                        writer.add_scalar(f'split_{split}/validation/iou', iou, global_step)
                    
    if save_cp:
        torch.save(net.state_dict(),
                   dir_logging + f'model_state_split{split}.pth')
        logging.info(f'Saved model state at end of split {split}')
    writer.close()


if __name__ == '__main__':
    
    ############################################################################
    ### SET LOGGING DIRECTORY 
    ### Model checkpoint and interrupt also saved here.
    subfolder = 'hyperparam_tuning'
    ############################################################################
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
    ### ENTER DETAILED DESCRIPTION OF EXPERIMENT
    logging.info('hyperparameter tuning for reveal data in UNet')
    ############################################################################

    ############################################################################
    ### MAKE TRAINING/VALIDATION SPLITS
    ### Can do cross-validation with lists of training and validation splits.
    # all_idxs = [[a , b] for b in range(1,4) for a in range(1,23)]
    # val_idxs, trn_idxs = generateSplits(all_idxs)
    val_idxs = [[[1, 2]]]
    trn_idxs = [[[6, 3], [19, 3], [17, 3], [4, 3], [12, 3], [8, 3], [2, 3], 
                 [16, 1], [21, 3], [19, 2], [3, 1], [14, 3], [11, 3], [5, 2], 
                 [18, 3], [7, 3], [9, 1], [13, 3], [10, 3]]]
    # for compatibility with the cross training nature...
    # val_idxs_splits, trn_idxs_splits = generateCrossValidationSplits(all_idxs)
    splits = 1
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
                      epochs = 32,
                      batch_size = 6,
                      lr = 1e-4,
                      save_cp = True,
                      splits = splits,
                      current_split = split)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), dir_logging + 'INTERRUPTED.pt')
            rmtree(os.environ['REVEAL_DATA'] + '/train_data/')
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
        
    # remove inital state
    os.remove(dir_logging + 'intial_state.pt')
