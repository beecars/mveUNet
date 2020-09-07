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

            targets = batch_data['target']
            targets = targets.to(device)

            predictions = model(cts)

            dice_score = dice(predictions, targets)
            dice_score += dice_score.item()
            
            iou_score = IoU(predictions, targets)
            iou_score += iou_score.item()
             
    dice_score /= len(test_generator)
    iou_score /= len(test_generator)
    
    if print_log == True:
        print('\tTest set: dice score: {:.4f}, iou score: {:.4f}\n'.format(dice_score, iou_score))
    
    return dice_score, iou_score
    

def eval_net_multiclass(model,
                        device,
                        test_generator,
                        print_log = False):
    
    model.eval()

    dice_score = 0.0

    with torch.no_grad():
        for i, batch_data in enumerate(test_generator):
            cts = batch_data['image']
            cts = cts.to(device)

            targets = batch_data['target']
            targets = targets.to(device)

            predictions = model(cts)

            dice_score = dice_coef_multiclass(predictions, targets, classes = 3)
            dice_score += dice_score.item()

    dice_score /= len(test_generator)

    if print_log == True:
        print('\tTest set: dice score: {:.4f}'.format(dice_score))

    return dice_score

            

