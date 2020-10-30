REVEAL imaging project code

================================================================================
======= IMPORTANT NOTES  ==== IMPORTANT NOTES  ==== IMPORTANT NOTES  ===========
================================================================================
      
   1. [General] This project assumes toplevel (where this README.txt lives) 
      is the python working directory!

   2. [REVEAL] This project has been entirely restructured to use .mat files
      representing CT and mask volumes. The main idea/goal with the 
      re-structuring of the data and code is to work closer to "volumes" than 
      "scans".
      
      PREVIOUSLY we had a spaghetti filesystem heirarchy of images, naming 
      conventions, and image file types... requiring ad-hoc parsing and utility 
      functions to accomplish what should be simple tasks. This often required
      turning the images back into volumes anyway.
      
      NOW, in contrast, 2D scan data is generated from 3D volumes ON DEMAND
      (like, before training the convnet). This allows for greater 
      flexibility and agility in many areas (for example, exploring 
      reprojections/permutations of the volume data for 2.5D or 3D UNet), 
      and has greatly reduced the total lines of code needed for the project
      while simultaneously making it more extensible. 

      The only data required to run the UNet are .mat files containing all the 
      volume data for the patients on particular imaging days:
         
            'patient1_day1.mat', 'patient1_day2.mat', ... , patient22_day3.mat
      
            |^| For the scripts in this package to run properly, these .mat 
            |^| volume files need to be stored in a folder: 
            |^|
            |^|    (os.environ['REVEAL_DATA'] + '/ct_mask_volumes/')
            |^|  
            |^|        where os.environ['REVEAL_DATA'] is a system-level 
            |^|        environment variable on the local machine. 
               
            The .mat's should contain 'ct' and 'pt' variables representing CT 
            and PET volumes, and may contain mask volumes in the form of 
            'spine_mask', 'pelvis_mask', sternum_mask'. Note that not all 
            [patient_idx, day_idx] .mat volumes will have the same mask data, 
            and some volumes have no mask data. The "generateSplits()" 
            function in the "utils.py" file handles this issue by matching
            patients with a given mask configuration. It then creates and
            training and validation split from that subset.

            The .mat's should also be oriented such that the first two 
            dimensions indexing the volume represent the axial plane in an 
            "back-down" position - the "rows" of the resulting image being the 
            first dimension, and the "columns" being the seconds dimension. The
            third dimension is then "along the spine". This initial orientation
            is important because functions in this project assume this 
            orientation for indexing or for performing actions to reproject or 
            re-slice the volume along different image planes. 
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