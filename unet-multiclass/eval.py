import torch
from losses import dice, IoU

def eval_net(model, 
             device, 
             test_generator,  
             print_log=False):
    
    model.eval()
    
    dice_score = 0.0
    iou_score = 0.0
    
    with torch.no_grad():
        for i, batch_data in enumerate(test_generator):
            cts = batch_data['image']
            cts = cts.to(device)

            labels = batch_data['target']
            labels = labels.to(device)

            outputs = model(cts)

            masks_probs = outputs

            dice_ = dice(masks_probs, labels)
            iou_ = IoU(masks_probs, labels)

            dice_score += dice_.item()
            iou_score += iou_.item()
            
    dice_score /= len(test_generator)
    iou_score /= len(test_generator)
    
    if print_log == True:
        print('\tTest set: dice score: {:.4f}, iou score: {:.4f}\n'.format(dice_score, iou_score))
    
    return dice_score, iou_score
###