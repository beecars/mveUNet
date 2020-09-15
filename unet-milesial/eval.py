import torch
import torch.nn.functional as F
from tqdm import tqdm

from losses import dice_coeff, iou

def eval_net(net, loader, device):
    '''Evaluation without the densecrf with the dice coefficient'''
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    dice_tot = 0
    iou_tot = 0
    with tqdm(total=n_val, 
              desc='Validation', 
              unit='batch', 
              leave=False,
              ascii = True,
              bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}') as pbar:
              
        for batch in loader:
            imgs, true_masks = batch['image'], batch['target']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                dice_tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                dice_tot += dice_coeff(pred, true_masks).item()
                iou_tot += iou(pred, true_masks).item()
            pbar.update()

    net.train()
    return (dice_tot / n_val), (iou_tot / n_val)
