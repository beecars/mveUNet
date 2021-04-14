# Multi-View Ensemble Multiclass UNet

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

## Instructions for use...
---
**Assuming** you have your data in the form, format, and location that is described above...

### 1. Ensure you have `"os.environ['DATA']"` a.k.a. a user or system-level environment variable "DATA" set to the location of your data.

### 2. Create training data for the UNet.   
   1. Run `generateSplits()` in utils.py to generate a stratified training/validation split. Or, you know, do it by hand.
      
      a. Choose what classes you want to include in the dataset by passing the "mask_critera" argument. The mask_criteria must match the names of masks in the `"patient#day#.mat"` files.
   
   2. Use the training splits as an arugment to `"generateNpySlices()"` to generate the training data.
      
      a. Again you need to pass class data, this time as a "mask_names" 
         argument.
   
   3. Ensure the .npy slices (2D image arrays) are where they are supposed to be.
      
      a. Default folder is `"os.environ['DATA']/train_data"`.
   
### 3. Set-up the UNet.

   1. Scroll to the bottom of `train.py`. 
   
   2. Change the `"subfolder"` string to change the name of the subfolder where
      your run log data will be stored. Also add a run description for the log,
      if you want.

      a. The default parent folder is `"reveal/unet-2D/.runs/"`. So your run logs will be stored at `"reveal/unet-2D/.runs/<subfolder>/"`

   3. Add `val_idxs` and `trn_idxs`. These can be from `generateSplits()`, or whatever you want them to be. But:
      
      a. They shouldn't be patient volumes associated with any of the scans in 
         the training data made in (1). 
         
      b. The ".mat" files that the `val_idxs` represent must have all the masks for 
         the classes you chose. Otherwise it won't work at all.

   4. Change the `"mask_names"` to be the same list of masks used for the function `generateNpySlices()`.

      a. The lists have to be the same order. If they aren't the same order, the network mix up which class is associated with which mask. It will be terrible. 
   
   5. Modify anything else you want about the UNet training scheme found in the 
      call to `train_net()` and in `train_net()` itself. Epochs. Batch size. 
      Learning rate. Learning rate scheduling. Loss functions. Optimizers. 
      Optimizer hyperparameters. Etc.

### 4. Run `train.py`.

   1. It trains.

### 5. Can run tensorboard to analyze model during training! Or after! Whenever!