import sys
import os

from torch.nn.modules.loss import CrossEntropyLoss
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
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
from losses import FocalLossBCE, FocalLossCE

from utils.dataset import CTMaskDataset
from utils.utils import generateNpySlices, generateSplits, getScanCount

def train_net(net,
              device,
              train_idxs,
              val_idxs,
              mask_names,
              epochs = 10,
              batch_size = 1,
              grad_clip = 1,
              lr = 0.0001,
              save_cp = True,
              init_weights = True):

    if net.n_classes > 1:   # for multiclass training
        train_dataset = CTMaskDataset()
        if init_weights == True:
            class_weights = torch.Tensor([.1, .3, .3, .3]).to(device)
        else:
            class_weights = torch.Tensor([.25, .25, .25, .25]).to(device)
        criterion = nn.CrossEntropyLoss(weight = class_weights)
        optimizer = optim.AdamW(net.parameters(), 
                                lr = lr, 
                                weight_decay = .0)
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size = 16,
                                              gamma = 0.1)
    else:   # for single class training
        train_dataset = CTMaskDataset()
        criterion = FocalLossBCE()
        optimizer = optim.AdamW(net.parameters(), 
                                lr = lr, 
                                weight_decay = .0)
        # lrmultiply = lambda epoch: 10**(epoch/5)
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer, 
        #                                         lr_lambda = lrmultiply)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
        #                                                  'max',
        #                                                  patience = 8)
    n_train = len(train_dataset)
    n_val = getScanCount(val_idxs)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle = True,
                              num_workers = 12,
                              pin_memory = True)
    
    writer = SummaryWriter(log_dir = dir_logging)
    logging.info(f'Training initialization:\n'
                 f'\tEpochs:                {epochs}\n'
                 f'\tBatch size:            {batch_size}\n'
                 f'\tLoss Function:         {criterion.__class__.__name__}\n'
                 f'\tOptimizer:             {optimizer.__class__.__name__}\n'
                 f'\tClass Weights:         {class_weights}\n'
                 f'\tOptimizer Args:        {optimizer.defaults}\n'
                 f'\tLearning rate:         {lr}\n'
                 f'\tGrad Clip Val:         {grad_clip}\n'
                 f'\tDevice:                {device.type}\n'
                 f'\tClass masks:           {mask_names}\n'
                 f'\tTraining size:         {n_train}\n'
                 f'\tValidation size:       {n_val}\n'
                 f'\tValidataion Volumes:   {val_idxs}\n'
                 f'\tTraining Volumes:      {train_idxs}')

    global_step = 0

    for epoch in range(epochs):
        
        # for running in "reduced initial bg weight" mode
        if net.n_classes > 1 and init_weights == True and epoch == 4:
            class_weights = torch.Tensor([.25, .25, .25, .25]).to(device)
            criterion = nn.CrossEntropyLoss(weight = class_weights)
        
        epoch_loss = 0
        with tqdm(total = n_train,    # progress bar
                  desc = f'Epoch {epoch + 1}/{epochs}', 
                  unit = 'img',
                  ascii = True,
                  leave = False,
                  bar_format = '{l_bar}{bar:60}{r_bar}{bar:-10b}') as pbar:

            for batch in enumerate(train_loader):
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

                # compute and log the loss
                loss = criterion(pred_masks, true_mask)
                epoch_loss += loss.item()
                writer.add_scalar(f'train_loss', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': round(loss.item(), 5)})

                # backpropogate the loss
                optimizer.zero_grad()
                loss.backward()
                
                # gradient clipping
                if grad_clip != 0:
                    nn.utils.clip_grad_value_(net.parameters(), grad_clip)
                optimizer.step()

                # update the progress bar
                pbar.update(imgs.shape[0])
                global_step += 1
                
            # validation round
            net.eval()
            with torch.no_grad():
                dices, ious = eval_volumes(net, 
                                            device,
                                            val_idxs,
                                            mask_names,
                                            p_threshold = 0.5)
            # log validation metrics
            writer.add_scalars(f'dice', dices, global_step)
            writer.add_scalars(f'iou', ious, global_step)

            # set net back to train mode
            net.train()
        
            # step through learning scheduler
            scheduler.step()
            writer.add_scalar(f'learning_rate', 
                              optimizer.param_groups[0]['lr'], 
                              global_step)
                
    if save_cp:
        torch.save(net.state_dict(),
                   dir_logging + f'model_state.pth')
        logging.info(f'Saved model state file to logging folder!')
    writer.close()

if __name__ == '__main__':
    ############################################################################
    ### TRAINING SET UP
    ### log folder / description / train & validation volumes / masks
    ### subfolder name and description for run logs
    subfolder = 'multiclass_testing'
    run_description = 'FocalLossCE Test'
    
    ### mask names defining the class masks (see README)
    mask_names = ['spine_mask', 'stern_mask', 'pelvi_mask']
    n_classes = len(mask_names)+1 if len(mask_names) > 1 else 1
    
    ### training/validation splits by patient vol_idx (see README)
    val_idxs = [[2, 1], [2, 3]]
    trn_idxs = [[1, 2], [3, 3], [5, 3], [5, 2], [4, 2], [5, 1], [3, 2], [4, 1], 
                [6, 1], [6, 3], [1, 3], [3, 1], [1, 1], [4, 3]]
    generateNpySlices(trn_idxs, mask_names = mask_names)
    ############################################################################
    
    dt_string = datetime.now().strftime('%Y-%m-%d_%H.%M')
    dir_logging = 'unet2D/.runs/{}/{}/'.format(subfolder, dt_string)
    try:
        os.makedirs(dir_logging)
    except OSError:
        pass

    # logging setup
    logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(dir_logging + "INFO.log"),
                                logging.StreamHandler()])
    logging.info(run_description)

    # UNet setup
    net = UNet(n_channels=1, n_classes=n_classes, bilinear=False)
    device = torch.device('cuda')
    net.to(device=device) # model to GPU
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
    
    # call training loop
    try:
        train_net(net,
                  device,
                  trn_idxs,
                  val_idxs,
                  mask_names,
                  epochs = 48,
                  batch_size = 6,
                  lr = 1e-4,
                  save_cp = True)
        
    except KeyboardInterrupt:
        torch.save(net.state_dict(), dir_logging + 'INTERRUPTED.pt')
        rmtree(os.environ['DATA'] + '/train_data/')
        logging.info('Saved interrupted state...')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)