REVEAL imaging project code

================================================================================
======= IMPORTANT NOTES  ==== IMPORTANT NOTES  ==== IMPORTANT NOTES  ===========
================================================================================
      
   1. [General] This project assumes toplevel (where this README.txt lives) 
      is the python working directory! 
      
   2. [General] This project will not work on other volume datasets other than 
      the REVEAL dataset described below (at least, not without considerable
      modifications). I am working on a better design that will more easily 
      generalize to other volume datasets. I think it is not far off... 

   2. [REVEAL] The only data required to run the UNet are .mat files containing 
      all the volume data for the patients on particular imaging days:
         
         'patient1_day1.mat', 'patient1_day2.mat', ... , patient22_day3.mat
   
         |^| For the scripts in this package to run properly, these .mat 
         |^| volume files need to be stored in a folder: 
         |^|
         |^|    (os.environ['REVEAL_DATA'] + '/ct_pt_volumes/')
         |^|  
         |^|        where os.environ['REVEAL_DATA'] is a system or user-level 
         |^|        environment variable on the local machine.
         
         These volume data files are accessed throughout this project by 
         arrays in the form [patient_idx(int), day_idx(int)]. For example,
         passing [2, 1] to the "eval_volume()" function as a "vol_idx" 
         argument will evaluate from the 'patient2_day1.mat' data file: 
                              
                                    maps to
               vol_idx = [2, 1]      ---->       patient2_day1.mat

         The .mat's should contain at least 'ct' and 'pt' variables 
         representing CT and PET volumes, and may contain mask volumes  
         in the form of 'spine_mask', 'pelvis_mask', sternum_mask'. 
         
         The masks of the .mat's will be accessed in this project by using a 
         'mask_names' list. List elements must be the same strings as the 
         name of the mask variable in the "patient.mat" file associated with 
         the vol_idx. This list must be ordered by the class number as 
         it appears in the target, where background is automatically assumed 
         zero:
            
         -------------------------------------------> implies target bg = 0
         mask_names = ['spine_mask', 'stern_mask']  > *    spine pixels = 1
         -------------------------------------------> *  sternum pixels = 2 

         Note that not all [patient_idx, day_idx] .mat volumes will have  
         the same mask data, and some volumes have no mask data. The 
         "generateSplits()" function in the "utils.py" file handles this 
         issue by matching patients with a given mask configuration. It
         then creates training and validation split from only that subset.

         The .mat's should also be oriented such that the first two 
         dimensions indexing the volume represent the axial plane in an 
         "back-down" position - the "rows" of the resulting image being  
         the first dimension, and the "columns" being the second
         dimension. The third dimension is then "along the spine". This 
         initial orientation is important because some utility functions 
         in this project assume this orientation for indexing or for 
         performing actions to reproject or re-slice the volume along 
         different image planes. 
================================================================================
================================================================================
CONTENTS:              
---------
unet-2D/
   [PYTORCH] UNet implementation.
   
   Adapted from: https://github.com/milesial/Pytorch-UNet (GNU license)
   Most functions/scripts heavily modified, replaced, or completely changed. 
   Only /unet/ contents (model and components) remains unchanged.
   
   -- train.py: function to train the network on CT data.
   -- losses.py: loss functions for training.
   -- eval.py: validation metrics.
   -- predict.py: function to predict/segment volume from a volume index.
   -- tasks.ipynb: jupyter notebook with some useful tasks set up.
   /unet/
      -- contains the pytorch unet model and component parts.
   /utils/
      -- dataset.py: contains CTMaskDataset for training and VolumeDataset 
                     (which is not yet used)
      -- utils.py: contains various important utilily functions.
      -- augment.py: albumentations augmentation implementation. 
================================================================================
================================================================================
INSTRUCTIONS FOR USE ON REVEAL DATA:
------------------------------------
0. Ensure you have patient volume data stored in the structure describes in the
   "IMPORTANT NOTES" section at the top of this file.
00. Ensure you have "os.environ['REVEAL_DATA']" a.k.a. a user or system-level 
    environment variable "REVEAL_DATA" set to the location of your data.

1. Create training data for the UNet.
   
   A. Recommmend a .ipynb for this. Maybe start at tasks.ipynb.    
   
   B. Run "generateSplits()" in utils.py to generate a stratified training/vali-
      dation split. Or, you know, do it by hand.
      
      a. Choose what classes you want to include in the dataset by passing
         the "mask_critera" argument. The mask_criteria must match the names of
         masks in the patient#day#.mat files.
   
   C. Use the training splits in "generateNpySlices()".
      
      a. Again you need to pass class data, this time as a "mask_names" 
         argument.
   
   D. Ensure the .npy slices (2D image arrays) are where they are supposed to be.
      
      a. Default folder is "os.environ['REVEAL_DATA']/train_data".
   
2. Set-up the UNet.

   A. Scroll to the bottom of train.py. 

   B. The   if __name__ == '__main__':    code will run only when train.py is 
      called to run standalone. 
   
   C. Change the "subfolder" string to change the name of the subfolder where
      your run log data will be stored. Also add a run description for the log,
      if you want.

      a. The default parent folder is "reveal/unet-2D/.runs/". So your run logs
         will be stored at "reveal/unet-2D/.runs/<subfolder>/"

   D. Add val_idxs and trn_idxs (see IMPORTANT NOTES). These can be from 
      "generateSplits()", or whatever you want them to be. But:
      
      a. They shouldn't be patient volumes associated with any of the scans in 
         the training data made in (1). 
         
      b. The .mat files that the val_idxs represent must have all the masks for 
         the classes you chose. Otherwise it won't work at all.

   E. Change the "mask_names" to be the same list of masks used for the function 
      "generateNpySlices()".

      a. The lists have to be the same order. If they aren't the same order, the 
         network mix up which class is associated with which mask. It will be 
         terrible. 
   
   F. Modify anything else you want about the UNet training scheme found in the 
      call to train_net() and in train_net() itself. Epochs. Batch size. 
      Learning rate. Learning rate scheduling. Loss functions. Optimizers. 
      Optimizer hyperparameters. Etc.

3. Run train.py.

   a. It trains.

4. Can run tensorboard to analyze model during training! Or after! Whenever!