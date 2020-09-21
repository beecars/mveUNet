REVEAL imaging project code

================================================================================
======= IMPORTANT NOTES  ==== IMPORTANT NOTES  ==== IMPORTANT NOTES  ===========
================================================================================
      
      1. [For Reveal CT] CTMaskDataset requires setting an environment variable 
         "REVEAL_DATA" to the folder path that contains "CT-PT-Images" and 
         "UCharImages-Multiclass" sub-folders.

      2. [General] This project assumes toplevel (where this README.txt lives) 
         is the python working directory!
--------------------------------------------------------------------------------


Contents:              
--------------------------------------------------------------------------------
unet-3D/     **** NOT WORKING WITH REVEAL CT DATA YET ****
   [TENSORFLOW] 3D UNet implementation in testing...
   https://github.com/ellisdg/3DUnetCNN (MIT license)
   
unet-milesial/
   [PYTORCH] UNet implementation that makes testing and optimizing easier!
   https://github.com/milesial/Pytorch-UNet (GNU license)
   -- Run train.py standalone to train the network (no notebooks).
   -- Pass args with train.py to adjust hyperparameters.
   -- Record losses, learning rates, weights, etc to tensorboard for easy
      analysis and comparison between experiments.
   -- MODIFICATIONS:
      -- Compatibility with REVEAL CT data via CTMaskDataset class.
      -- MixedLoss/FocalLoss from /unet-chuong/.
      -- Changed log & checkpoint save dirs to /unet-milesial/.runs/ & 
         /unet-milesial/.checkpoints/, repsectively.
      -- Added more verbose logging and made hyperparameters save to INFO
         file in /unet-milesial/.runs/.

unet-multiclass/        **** NOT WORKING WELL ****
   [PYTORCH] UNet implemention adapted from Chuong's files to allow for 
             multiclass segmentations.
--------------------------------------------------------------------------------