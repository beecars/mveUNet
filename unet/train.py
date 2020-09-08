import numpy as np
import torch
from losses import dice, IoU

def train_net(model, 
              device, 
              train_loader,
              batch_size, 
              criterion, 
              optimizer,
              learning_scheduler,
              epoch, 
              print_log=False):
          
    model.train()
    train_loss = []

    for batch_idx, batch_data in enumerate(train_loader):
        
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
        if print_log == True and batch_idx % 100 == 0:
            print('Train Epoch {}:  [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch+1, 
                  batch_idx * batch_size, 
                  len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), 
                  loss.item())
            )
            
    learning_scheduler.step()

    return train_loss
###

if __name__ == '__main__':
    # code to run as callable training script goes below...'
    pass
