REVEAL imaging project code

================================================================================
======= IMPORTANT NOTES  ==== IMPORTANT NOTES  ==== IMPORTANT NOTES  ===========
================================================================================
      
      1. [General] This project assumes toplevel (where this README.txt lives) 
         is the python working directory!

      2. [REVEAL] This project has been entirely restructured...
      
         The main idea/goal with the re-structuring of the data and code is to 
         work closer to "volumes" than "scans".
         
         Previously we had a spaghetti filesystem heirarchy and tons of parsing 
         utility functions, and often we would have to turn these images back 
         into volumes anyway. 
         
         So now I've turned the problem upside down. The only "data" required 
         to run the UNet are .mat files containing all the volume data for a 
         particular patient on a particular day:
            
            'patient1_day1.mat'
            'patient1_day2.mat'
                  ....
         
         The .mat volume files need to be stored in a folder:
            
            os.environ['REVEAL_DATA'] + '/ct_mask_volumes/'

         From these volumes, 2D scan data is generated ON DEMAND as it is needed 
         (like, before training the convnet). This allows for greater 
         flexibility and agility in many areas (for example, exploring 
         reprojections/permutations of the volume data for 2.5D or 3D UNet), 
         and has greatly reduced the total lines of code needed for the project.
        
         All .mat above should contain 'ct' and 'pt' variables representing 
         CT and PET volumes, and many contain mask volumes in the form of 
         'spine_mask', 'pelvis_mask', 'sternum_mask'.
--------------------------------------------------------------------------------


Contents:              
--------------------------------------------------------------------------------
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