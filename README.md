# Multi-View, Multiclass 2D UNet for 3D Image Volumes
The code in this repository provides the ability to train and test 2D U-Nets from the three standard projections of an image volume (in medical imaging, these are the axial, sagittal, and coronal anatomical planes). It is designed to work with medical image volumes in the form of MATLAB variable files - but the code is easily generalized to work on 3D volumes, in general, of different filetypes. 

## U-Net Model
The U-Net model used in this repository (in the `"/mveUNet/unet2D/unet/"` folder) is based off of this work: [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet). It is the same architecture as the original [U-Net](https://arxiv.org/abs/1505.04597) but with added batch normalization layers. 

## Dataset
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

For testing: The ".mat" files should contain at least `'ct'`variable representing CT volumes. 
For training: The ".mat" files will also contain mask volume(s) of the same dimension as the CT volume.

Some functions used in this repository use a ``mask_names``(str) list. The training function ``train_net()`` is such an example. Also ``generateNumpySlices()``. List elements must be selected from the same strings as the names of the mask variables in the "patient.mat" files. When training a model, the ``mask_names`` list must be in the same order if they are called for different tasks. 

The ".mat" files should also be oriented such that the the first dimension represents the direction normal to the "coronal" anatomical plane, the second dimension represents the direction normal to the "sagittal" plane, and the third dimension represents the direction normal to the "axial" plane. The reason for this is different data augmentations are performed for each anatomical plane, and some functtions use ``plane`` as an argument, so consistency in the orientation of the volume data is very important here. 

## TRAINING THE MODEL

**Assuming** you have your data in the form, format, and location that is described above...

### 1. Ensure you have `"os.environ['DATA']"` a.k.a. a user or system-level environment variable "DATA" set to the location of your data.
If you would rather not set an environment variable, you will simply have to change the default ``folder`` arguments in many of the functions in the ``utils.py`` file. Recommend just searching the code for ``"folder ="`` if going this route. You would also need to change the ``folder =`` default argument in ``dataset.py``.

### 2. Create training data for the UNet.   
   1. (Optional) If needed, run `generateSplits()` in utils.py to generate a stratified training/validation split of vol_idxs to separate by patient. This is only really needed if you have multiple scans per patient. 
      
      a. Choose what classes you want to include in the dataset by passing the ``mask_names`` (string list) argument. The `mask_names` must match names of masks in the `"patient#day#.mat"` files.
   
   2. Use the training split (a list of vol_idxs representing the training volumes) as an arugment to `"generateNpySlices()"` to generate the training data. The function `"generateNpySlices()"` has an argument for the anatomical plane. If you don't know what you want, then there is a 99% chance you should leave this alone, where it defaults to the `"axial"` plane. But if you want to train from the `"sagittal"` or `"coronal"` planes, those options are available. 
      
      a. You need to pass class names as a  `mask_names` string list argument to `"generateNpySlices()"`. This function builds the training pairs.
   
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
         the classes you chose to train on. Otherwise it won't work at all.

   4. Change the `"mask_names"` to be the same list of masks used for the function `generateNpySlices()`.

      a. The lists have to be the same order. If they aren't the same order, the network mix up which class is associated with which mask. It will be terrible. 
   
   5. Modify anything else you want about the UNet training scheme found in the 
      call to `train_net()` and in `def train_net()` itself. Epochs. Batch size. 
      Learning rate. Learning rate scheduling. Loss functions. Optimizers. 
      Optimizer hyperparameters. If, when creating training data, you changed 
      the anatomical plane to something other than "axial", you will want to 
      override the ``plane`` argument to the ``train_net()`` call.

### 4. Run `train.py`.

   1. It trains.

### 5. Can run tensorboard to analyze model during training! Or after! Whenever!

## MAKING PREDICTIONS

Check out the notebook file `"/mveUNet/unet2D/tasks.ipynb"`. There is a "Generate prediction volume from model." task. Modify as needed! Essentially it does this: given a model, the path to the model weights, and a vol_idx, it makes a volume prediction for each object class in the model. Note that you can set the "`threshold`" and "`p_threshold"` arguments in the `"predict_vol_from_vol_idx()"` function to allow raw prediction volumes (voxels in the range [0, 1]) or thresholded masks. I have it saving to a ".mat" format, since I do analysis in MATLAB. 