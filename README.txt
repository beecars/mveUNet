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
         |^|    (os.environ['REVEAL_DATA'] + '/ct_mask_volumes/')
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
Contents:              
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
   /unet/
      -- contains the pytorch unet model and component parts.
   /utils/
      -- dataset.py: contains CTMaskDataset for training and VolumeDataset 
                     (which is not yet used)
      -- utils.py: contains various important utilily functions.
--------------------------------------------------------------------------------