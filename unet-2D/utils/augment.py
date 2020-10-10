import numpy as np
import albumentations as albu

def augment_dict(image_dict):

    augmentation = albu.Compose([
        albu.ShiftScaleRotate(shift_limit=0.2, rotate_limit=5, p=1),
        albu.CenterCrop(320, 320),
        albu.HorizontalFlip(p=0.5)
        ])

    image = np.expand_dims(image_dict['ct'], axis=2)
    mask =  np.expand_dims(image_dict['spine'], axis=2)
    
    aug_imgs = augmentation(image = image, mask = mask)
    for item in aug_imgs:
        aug_imgs[item] = np.squeeze(aug_imgs[item])
    
    aug_imgs['ct'] = aug_imgs.pop('image')
    aug_imgs['spine'] = aug_imgs.pop('mask')

    return aug_imgs