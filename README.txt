can use ipynb file to test the UNet training.

will require changing data paths in:
    "utils.py" -> "match_files_from_patient" 
    (match your own environment) 
    
files modified from Choung's UNet.
    notable modifications:
        -- wrapped filepath strings as pathlib.Path objects
        -- aggregated the functionality of previous "utils.py" -> "get_*" 
           defs into a single function with mode argument
          -- changed filepath parse for greater robustness w.r.t. 
             different filepath structures
          -- replaced "get_*" functions throughout the code
        -- cleaned/adjusted "train.py" so that it's def's can be
           imported without running the code.
