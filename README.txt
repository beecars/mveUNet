REVEAL imaging project code

================================================================================
======= IMPORTANT NOTES  ==== IMPORTANT NOTES  ==== IMPORTANT NOTES  ===========
================================================================================
      
      0. This repository is written for CT/Mask training data living in a 
         very particular folder structure with very particular names. Most
         of the code will not work out of the box without this structure. 
         However, rewriting matchFilesFromPatient() to generate a similar 2D
         file list from a different sturcture will give full functionality.
         There are many functions that can be used or easily generalized without
         interacting with the project-specific matchFilesFromPatient() function.

      1. [For Reveal CT] matchFilesFromPatient() requires setting an env. var.
         "REVEAL_DATA" to the folder path that contains "CT-PT-Images" and 
         "UCharImages-Multiclass" sub-folders:
          
          os.environ['REVEAL_DATA']
            |--- CT-PT-Images
            |     |--- P01
            |     |--- P02
            |     ...   |--- Day_1
            |           |--- Day_2
            |           |      |--- CT
            |           |      |     |--- P_02_001.bin
            |           |      |     |--- P_02_002.bin
            |           |      |     ...
            |           |      |
            |           |      |--- PT-Float
            |           |            |--- P_02_001.float
            |           |--- ...     |--- ...
            | 
            |--- UCharImages-MultiClass
                  |--- P1_1_048_Pelvis.uchar
                  |--- P1_1_049_Pelvis.uchar
                  |--- ...
                  |--- P2_3_107_Spine.uchar
                  |--- P2_3_107_Sternum.unchar
                  |--- ...
                  |--- P21_3_121_Spine.uchar
                  |--- ...
         * under UCharImages, P# is the patient identifier, the following 
           integer is the day identifier, the next integer is the scan 
           identifier, and just before the .uchar suffix is the class identifier

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
   
   Adapted from: https://github.com/milesial/Pytorch-UNet (GNU license)
   Most functions/scripts heavily modified, replaced, or completely changed. 
   Only /unet/ contents (model and components) remains unchanged.
   
   -- train.py: function to train the network on CT data.
      -- Run standalone and save logs and model state to /.runs/.
         -- Script currently set up for 7-fold cross-validation on the 
            REVEAL CT dataset. Cross-validation folds are partitioned by volume 
            and by patient. That is to say, each volume will only appear in one 
            fold, and moreover - each patient will only appear in one fold. This 
            is to prevent overfitting and to achieve validation scores that are 
            more representative of the performace of the model on new data if we 
            were to train with all current data. 
         -- Can easily run with "leave some out" type validation by commenting
            the cross validation lines and uncommenting the relevant lines. 
      -- Pass args with train.py to adjust hyperparameters.
      -- Tensorboard for loss, lr, weights, etc analysis and comparison.
      -- train_net() can be called for non-cross-validated training by passing 
         folds = 1 and redefining the training and validation data in the 
         'main' script.
   -- losses.py: loss functions for training.
      -- I don't think I'm even using any of these right now, train.py is
         working well with the pytorch BCE built-in loss function.
   -- eval.py: validation metrics.
      -- The validation metrics used during training in train.py are full-
         volume calculations (from eval.py file). For the REVEAL CT data we are 
         interested in measuring the performance of the 3D segmentation, thus it 
         makes sense to calculate metrics (dice, IoU) on entire volumes and not 
         individual slices. If we measure IoU on slices and then take the mean
         and report that as our performance, it is not representative of the 
         performance on a volume. This is because a slice with only 1 object
         pixel predicted perfectly gets a perfect IoU of 1, while a slice with 
         1000 object pixels where 900 are predicted correctly (with no FP) will 
         get a .9 IoU score. If we report the average, this results in a .95 IoU
         which is obviously not a reasonable result. When calculating the IoU 
         over the volume in the above situation the value will be very close to 
         0.9, which is a much more meaningful result since we are interested in 
         the volume segmentation.  
   -- predict.py: functions to predict/segment volume from CT sequence derived 
                  from matchFilesFromPatient() or from a CT volume in the format
                  of: .mat, .npy, or 3D numpy array.
      -- Can run standalone and save prediction as .npy file to /.predictions/.
      -- Pass args with predict.py to choose patient/day and load model state.
   /unet/
      -- contains the pytorch unet model and component parts.
   /utils/
      -- dataset.py: contains Dataset classes for REVEAL CT data.
      -- utils.py: contains various utilily functions.
         -- matchFilesFromPatient(): important function that generates lists of
            matching CT/PT/mask data. Used throughout project.
--------------------------------------------------------------------------------