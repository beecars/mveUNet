# Multi-View Ensemble Multiclass UNet
Convnets are widely used for medical image segmentation tasks, and UNet has proven to be an effective model when paired with data augmentation on smaller training datasets. The current best-in-class convnets for semantic segmentation of volume data use computationally expensive 3D convolutional layers. The acceleration hardware used for the convnet training and inference in this thesis is a Nvidia GTX 1070, which only has 8GB of VRAM available for parallel computation. This puts the 3D semantic segmentation convnets just out of reach. Instead, I use a multi-view ensemble of 2D UNets to increase the accuracy of the segmentation. In this ensemble method, the results of three different 2D UNets are combined to form a single prediction volume. Each of the three UNets has been trained on scans from a different “view” – in this case the views are the sagittal, axial, and coronal anatomical planes. The system can also segment the pelvis and sternum.

## U-Net Model
The U-Net model used in this repository (in the `"/mveUNet/unet2D/unet/"` folder) is based off of this work: [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet). It is the same architecture as the original [U-Net](https://arxiv.org/abs/1505.04597) but with added batch normalization layers. 

## Dataset
---
For this repository to function, it is required to accompany it with a very specfic patient dataset.

The "rules" for this dataset are as follows:

* The dataset files are matlab ".mat" files.
* The ".mat" files are named in this convention: `"patient1_day1.mat", "patient1_day2.mat", ... , "patient22_day3.mat"`
* The ".mat" files live in this folder: `os.environ['DATA']/ct_pt_volumes/` (where `os.environ['DATA']` is a system or user-level environment variable on the local machine)

These volume data files are accessed throughout this project by arrays in the form `[patient_idx(int), day_idx(int)]`. For example, passing `[2, 1]` to the `eval_volume()` function as a `vol_idx` argument will evaluate from the `"patient2_day1.mat"` data file: 
```
                         maps to
    vol_idx = [2, 1]      ---->       "patient2_day1.mat"
```

The ".mat" files should contain at least `'ct'` and `'pt'` variables representing CT and PET volumes. 
For training they will also contain mask volumes in the form of `'spine_mask'`, `'pelvis_mask'`, `'sternum_mask'`.

The masks of the .mat's will be accessed in this project by using a 'mask_names' list. List elements must be the same strings as the name of the mask variable in the "patient.mat" file associated with 
the vol_idx. This list must be ordered by the class number as it appears in the target, where background is automatically assumed zero:
```
-----------------------------------------------> implies target bg = 0
mask_names = ['spine_mask', 'stern_mask']  ----> implies spine pixels = 1
-----------------------------------------------> implies sternum pixels = 2 
```

Note that not all ".mat" volume files need to have the same mask data, and some volumes have no mask data. The `generateSplits()` function in the `utils.py` file handles this issue by matching patients with a given mask configuration. It then creates training and validation split from only that subset.

The ".mat" files should also be oriented such that the the first dimension represents the direction normal to the "coronal" anatomical plane, the second dimension represents the direction normal to the "sagittal" plane, and the third dimension represents the direction normal to the "axial" plane. 
___
## TRAINING THE MODEL
---
**Assuming** you have your data in the form, format, and location that is described above...

### 1. Ensure you have `"os.environ['DATA']"` a.k.a. a user or system-level environment variable "DATA" set to the location of your data.

### 2. Create training data for the UNet.   
   1. (Optional) Run `generateSplits()` in utils.py to generate a stratified training/validation split of vol_idxs. Or, do it by hand.
      
      a. Choose what classes you want to include in the dataset by passing the "mask_critera" argument. The `mask_names` must match the names of masks in the `"patient#day#.mat"` files.
   
   2. Use the training split (a list of vol_idxs representing the training volumes) as an arugment to `"generateNpySlices()"` to generate the training data. The function `"generateNpySlices()"` has an argument for the anatomical plane. If you don't know what you want, then there is a 99% chance you should leave this alone, where it defaults to the `"axial"` plane. But if you want to train from the `"sagittal"` or `"coronal"` planes, those options are available. 
      
      a. You need to pass class names as a  `mask_names` list argument to `"generateNpySlices()"`.
   
   3. Ensure the .npy slices resulting from `"generateNpySlices()"` (2D image arrays) are where they are supposed to be.
      
      a. Default folder is `"os.environ['DATA']/train_data"`.
   
### 3. Set-up the UNet.

   1. Scroll to the bottom of `train.py`. 
   
   2. Change the `"subfolder"` string to change the name of the subfolder where
      your run log data will be stored. Also add a run description for the log,
      if you want.

      a. The default parent folder is `"/unet2D/.runs/"`. So your run logs will be stored at `"/unet2D/.runs/<subfolder>/"`

   3. Add `val_idxs` and `trn_idxs`. These can be from `generateSplits()`, or whatever you want them to be. But:
      
      a. Be careful that the validation indexes do not represent any of the same patients that appear in the training indexes. This would bias the validation. 
         
      b. The ".mat" files that the `val_idxs` represent must have all the masks for 
         the classes you chose. Otherwise it won't work at all.

   4. Change the `"mask_names"` to be the same list of masks used for the function `generateNpySlices()`.

      a. The lists have to be the same order. If they aren't the same order, the network mix up which class is associated with which mask. It will be terrible. 
   
   5. Modify anything else you want about the UNet training scheme found in the 
      call to `train_net()` and in `train_net()` itself. Epochs. Batch size. 
      Learning rate. Learning rate scheduling. Loss functions. Optimizers. 
      Optimizer hyperparameters. If you changed the anatomical plane to something
      other than "axial", you will want to override the "plane" argument to the taining
      to your desired plane. 

### 4. Run `train.py`.

   1. It trains.

### 5. Can run tensorboard to analyze model during training! Or after! Whenever!
---
## MAKING PREDICTIONS
---
Check out the notebook file `"/mveUNet/unet2D/tasks.ipynb"`. There is a "Generate prediction volume from model." task. Modify as needed! Essentially it does this: given a model, the path to the model weights, and a vol_idx, it makes a volume prediction for each object class in the model. Note that you can set the "`threshold`" and "`p_threshold"` arguments in the `"predict_vol_from_vol_idx()"` function to allow raw prediction volumes (voxels in the range [0, 1]) or thresholded masks. I have it saving to a ".mat" format, since I do analysis in MATLAB. 