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

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def train_net(net,
              device,
              train_idxs,
              val_idxs,
              mask_names,
              epochs = 10,
              batch_size = 1,
              lr = 0.0003,
              save_cp = True,
              splits = 1,
              current_split = 0):

    split = current_split + 1
    if net.n_classes > 1:   # for multiclass training
        train_dataset = CTMaskDataset(mask_criteria=mask_names)      
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(net.parameters(), 
                                lr = lr, 
                                weight_decay = .0)
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
                              num_workers=1, 
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
                     f'\tDevice:                {device.type}\n'
                     f'\tClass masks:           {mask_names}\n'
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
                imgs = batch['ct'] # (N, Channel, H, W)
                if net.n_classes > 1:
                    true_mask = batch['target'].squeeze(1) # (N, H, W)
                else:
                    true_mask = batch['target'] # (N, Channel, H, W)

                # load image and mask to device
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_mask = true_mask.to(device=device, dtype=mask_type)

                # forward pass image through model
                pred_masks = net(imgs)

                # calcululate and log loss
                loss = criterion(pred_masks, true_mask)
                epoch_loss += loss.item()
                writer.add_scalar(f'split_{split}/train_loss', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': round(loss.item(), 5)})

                # backpropogate the loss
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
                    dices, ious = eval_volumes(net, 
                                               device,
                                               val_idxs,
                                               mask_names,
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
                    writer.add_scalars(f'split_{split}/dice', dices, global_step)
                    writer.add_scalars(f'split_{split}/iou', ious, global_step)
                    
    if save_cp:
        torch.save(net.state_dict(),
                   dir_logging + f'model_state_split{split}.pth')
        logging.info(f'Saved model state at end of split {split}')
    writer.close()


if __name__ == '__main__':
    
    ############################################################################
    ### SET LOGGING DIRECTORY 
    ### Model checkpoint and interrupt also saved here.
    subfolder = 'multiclass testing'
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
    val_idxs = [[[2, 1], [2, 3]]]
    trn_idxs = [[[1, 2], [3, 3], [5, 3], [5, 2], [4, 2], [5, 1], [3, 2], [4, 1],
                 [6, 1], [6, 3], [1, 3], [3, 1], [1, 1], [4, 3]]]
    mask_names = ['spine_mask', 'stern_mask', 'pelvi_mask']
    # generateNpySlices(trn_idxs, mask_criteria=['ct', 'spine_mask', 'pelvi_mask', 'stern_mask'])
    splits = 1
    ############################################################################

    device = torch.device('cuda')

    net = UNet(n_channels=1, n_classes=4, bilinear=False)
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
                      mask_names,
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
