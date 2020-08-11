Can use ipynb file to test the UNet training.

Will require changing data paths in:
    "utils.py" -> "match_files_from_patient" 
    (match your own environment) 
    
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