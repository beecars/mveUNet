REVEAL imaging project code

================================================================================
======= IMPORTANT NOTES  ==== IMPORTANT NOTES  ==== IMPORTANT NOTES  ===========
================================================================================
      
      1. [For Reveal CT] matchFilesFromPatient() requires setting an env. var.
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
   
unet-2D/
   [PYTORCH] UNet implementation.
   https://github.com/milesial/Pytorch-UNet (GNU license)
   -- train.py: function to train the network on CT data.
      -- Can run standalone. 
      -- Pass args with train.py to adjust hyperparameters.
      -- Tensorboard for loss, lr, weights, etc analysis and comparison.
   -- losses.py: loss functions for training.
   -- eval.py: validation metrics function, called between epochs.
   -- predict.py: function to predict/segment volume from CT sequence.
      -- Can run standalone.
      -- Pass args with predict.py to choose patient/day and load model state.
   -- test.py: function to predict volume and compute performance metrics.
      -- Can run standalone.
      -- Pass args with predict.py to choose patient/day and load model state.
   /utils/
      -- dataset.py: contains Dataset classes for REVEAL CT data.
      -- utils.py: contains various utilily functions.
         -- matchFilesFromPatient(): important function that generates lists of
            matching CT/PT/mask data. Used throughout project.

unet-multiclass/        **** NOT WORKING WELL ****
   [PYTORCH] UNet implemention adapted from Chuong's files to allow for 
             multiclass segmentations.
--------------------------------------------------------------------------------