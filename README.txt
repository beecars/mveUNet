REVEAL imaging project code

NOTE: REQUIRES SETTING ENVIRONMENT VARIABLE "REVEAL_DATA" to the folder
      that contains "CT-PT-Images" and UCharImages-Multiclass" sub-folders.
-------------------------------------------------------------------------------
CONTENTS:
unet/
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
-------------------------------------------------------------------------------
    
Files modified from Choung's UNet.
   Notable modifications:
      -- Wrapped filepath strings as pathlib.Path objects
         throughout code.
      -- Aggregated the family of "get_*" defs from "utils.py"
         into a single function with mode argument.
         -- Changed filepath parse for greater robustness w.r.t. 
            different filepath structures.
         -- Replaced "get_*" functions throughout the code.
      -- Cleaned "train.py" so that it's defs can be imported 
         without running other code.
      -- Removed a lot of unused code from various files
         to clean up the test environment.
         -- Old versions in ".backup" folder for posterity. 