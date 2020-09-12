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
unet-milesial/
   UNet implementation that makes testing and optimizing easy! 
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

unet-chuong/
   UNet implemention adapted from Chuong's files (with significant 
   modifications)
   -- unet_ct_test.ipynb   A jupyter-style script for easy testing and
                           simple modification of the UNet training. 
   -- dataset.py     Pytorch Dataset subclass for ct training data.
   -- eval.py        Fucntion(s) used for computing validation scores.
   -- losses.py      Loss functions for optimization.
   -- model.py       The UNet model and its sub-parts.
   -- predict.py     Function(s) for running inference tests on completed
                     models. 
   -- train.py       Function optimizing the pytorch model on a single batch. 
                     A component of the training process (not standalone).
   -- utils.py       A variety of image reading, data coallating, plotting,
                     and other tools used throughtout the project. 

unet-multiclass/        **** NOT WORKING WELL ****
   Largely the same contents as /unet-chuong/ but functions are adapted to 
   enable multiclass segmentations.
-------------------------------------------------------------------------------