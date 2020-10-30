from os import environ, makedirs
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import whosmat, loadmat
from random import shuffle, choice
from sklearn.model_selection import GroupKFold
from tqdm import tqdm


def loadMatData(vol_idx,
                folder = environ['REVEAL_DATA'] + '\\ct_mask_volumes\\',
                data = 'ct'):
    """
    Loads .mat (MATLAB-type) files into system memory.
    Files must have particular naming convention: 'patient1_day1.mat'
    
    @params:
    vol_idx: the volume identifier in the form [p, d] to be loaded.
    folder: the folder containing the 'patient#_day#.mat' files.
    data: the variable to load from the .mat file (saves time?).

    @returns:
    mat: nparray data object containg the volume data.
    """
    mat = loadmat(folder + f'patient{vol_idx[0]}_day{vol_idx[1]}.mat',
                  variable_names = data)[data]
    return mat


def getScanCount(vol_idxs,
                 folder = environ['REVEAL_DATA'] + '\\ct_mask_volumes\\'):
    """
    From a list of vol_idxs determine the number of scans present.
    Used for counting training/validation scans.
    Files must have particular naming convention: 'patient1_day1.mat'

    @params:
    vol_idxs: a list of vol_idx to be inspected
    folder: the folder containing the 'patient#_day#.mat' files.

    @returns:
    total_scans: the number of scans in the volumes (int)
    """
    total_scans = 0
    for vol_idx in vol_idxs:
        vol_file = (folder + f'patient{vol_idx[0]}_day{vol_idx[1]}.mat')
        scans = whosmat(vol_file)[0][1][-1]
        total_scans = total_scans + scans
    return total_scans


def generateSplits(vol_idxs,
                   vol_folder = environ['REVEAL_DATA'] + '\\ct_mask_volumes\\',
                   mask_criteria = ['ct', 'spine_mask'],
                   val_ratio = 0.15):
    """ Search through the given folder, find .mat files that contain all the 
    mask data passed by 'mask_criteria'. Split up those into a training and a 
    validation set. Any given patient will only appear in EITHER the validation 
    or training set, but not both. This is by design.

    Files must have particular naming convention: 'patient1_day1.mat'
    
    @params:
    vol_idxs: a list of vol_idx to be inspected
    vol_folder: the folder containing the 'patient#_day#.mat' files.
    mask_criteria: list of strings identifying the masks that must be present
                   in the .mat files for the volume to be considered in the 
                   training/validation splits.
    val_ratio: ratio of the size of the validation set to the training set. 

    @returns: 
    val_idxs, trn_idxs: a list of validation and training vol_idxs.
    """
    # initialize array
    compatible_vol_idxs = []
    # look for data that have the required masks
    for vol_idx in vol_idxs:
        vol_file = (vol_folder + f'patient{vol_idx[0]}_day{vol_idx[1]}.mat')
        vol = whosmat(vol_file)
        # this crazy list comp. checks if the vol doesn't contain the masks
        if not all([criteria in [item[0] for item in vol] for criteria in mask_criteria]):
            continue
        # otherwise add it to the list
        compatible_vol_idxs.append(vol_idx)
    shuffle(compatible_vol_idxs)

    # begin the random training/validation split
    val_size = round(val_ratio * len(compatible_vol_idxs))
    val_idxs = []
    bucket = 0  
    while bucket < val_size:
        # fill this bucket to val_size
        temp_idxs = []
        diff = val_size - bucket    # vols left to fill
        chosen_idx = choice(compatible_vol_idxs)    # rand choice vol idx
        chosen_patient = chosen_idx[0]              # patient id of choice
        for idx in compatible_vol_idxs:     # look for other vols with idx
            if idx[0] == chosen_patient:    
                temp_idxs.append(idx)       # add them to the temp array
        if len(temp_idxs) <= diff:          # if they can fit in the bucket
            val_idxs.extend(temp_idxs)      # add temp array to val_idxs
            bucket += len(temp_idxs)        # update the bucket size
            # this filter removes the selected idxs from the larger set
            compatible_vol_idxs = list(filter(lambda idx: idx[0] != chosen_patient, compatible_vol_idxs))
    # the training volumes are what's left
    trn_idxs = compatible_vol_idxs
    return [val_idxs], [trn_idxs]


def generateCrossValidationSplits(vol_idxs,
                   vol_folder = environ['REVEAL_DATA'] + '\\ct_mask_volumes\\',
                   mask_criteria = ['ct', 'spine_mask'],
                   n_folds = 7):
    """ Search through the given folder, find .mat files that contain all the 
    mask data passed by 'mask_criteria'. Split up those into K number of 
    training and validation sets. For any given split, any given patient will 
    only appear in EITHER the validation or training set, but not both. This 
    also means neccessarily that each patient will appear in one and only one
    validation set. This is by design.

    Files must have particular naming convention: 'patient1_day1.mat'
    
    @params:
    vol_idxs: a list of vol_idx to be inspected
    vol_folder: the folder containing the 'patient#_day#.mat' files.
    mask_criteria: list of strings identifying the masks that must be present
                   in the .mat files for the volume to be considered in the 
                   training/validation splits.
    n_folds: the number of folds/splits for the k-fold training scheme. 

    @returns: 
    val_idxs, trn_idxs: 'n_folds' lists of validation and training vol_idxs,
                        to be used for cross-validation training. 
    """
    # initialize array
    compatible_vol_idxs = []
    # look for data that have the required masks
    for vol_idx in vol_idxs:
        vol_file = (vol_folder + f'patient{vol_idx[0]}_day{vol_idx[1]}.mat')
        vol = whosmat(vol_file)
        # this crazy list comp. checks if the vol doesn't contain the masks
        if not all([criteria in [item[0] for item in vol] for criteria in mask_criteria]):
            continue
        # otherwise add it to the list
        compatible_vol_idxs.append(vol_idx)
    vol_idxs = compatible_vol_idxs
    # create groups array from the patient identifiers
    groups = [vol_idx[0] for vol_idx in vol_idxs]
    # create the group k fold object
    GKF = GroupKFold(n_splits=n_folds)
    # run the split method to generate the splits
    val_idx_splits = []
    trn_idx_splits = []
    for trn_idxs, val_idxs in GKF.split(vol_idxs, groups = groups):
        trn_idx_splits.append([vol_idxs[i] for i in trn_idxs])
        val_idx_splits.append([vol_idxs[i] for i in val_idxs])

    return val_idx_splits, trn_idx_splits


def generateNpySlices(vol_idxs,
                      vol_folder = environ['REVEAL_DATA'] + '\\ct_mask_volumes\\',
                      output_folder = environ['REVEAL_DATA'] + '\\train_data\\', 
                      mask_criteria = ['ct', 'spine_mask'],
                      plane = 'axial'):
    """ From a list of vol_idxs, generate 2D .npy files with which to train a
    convnet.

    Places the .npy files in the following file hierarchy:
       
        data_dir
          |--- ct
          |     |--- 0.npy     (useful for a glob-style Dataset...
          |     |--- 1.npy         ... like "CTMaskDataset" found in dataset.py) 
          |     |...
          |--- spine
          |...  |--- 0.npy
                |...
        
    @params:
    vol_idxs: a list of vol_idx to be inspected.
    vol_folder: the folder containing the 'patient#_day#.mat' files.
    output_folder: folder where the .npy files will be written to.
    mask_criteria: list of strings identifying the masks that must be present
                   in the .mat files for the volume to be considered in the 
                   training/validation splits.
    plane: the intended image plane, one of 'axial', 'saggital', 'coronal'.
    """
    # generate file list
    vol_file_list = []
    for vol_idx in vol_idxs:
        vol_file_list.append(vol_folder + f'patient{vol_idx[0]}_day{vol_idx[1]}.mat')
    
    # make output directories as required
    if 'ct' in mask_criteria:
        makedirs(output_folder + 'ct/', exist_ok = True)
    if 'spine_mask' in mask_criteria:
        makedirs(output_folder + 'spine/', exist_ok = True)
    if 'sternum_mask' in mask_criteria:
        makedirs(output_folder + 'sternum/', exist_ok = True)
    if 'pelvis_mask' in mask_criteria:
        makedirs(output_folder + 'pelvis/', exist_ok = True)
    
    with tqdm(total = len(vol_file_list),    # progress bar
                    desc = f'Generating Training Scans', 
                    unit = 'volume',
                    ascii = True,
                    leave = False,
                    bar_format = '{l_bar}{bar:60}{r_bar}{bar:-10b}') as pbar:

        file_num = 0    # will be incremented to name files
        for file in vol_file_list:
            volume = loadmat(file)
            if plane == 'axial':
                 # sample the volume on the "default" axial axis and save
                for idx in range(0, volume['ct'].shape[2]):
                    if 'ct' in mask_criteria:
                        np.save(output_folder + 'ct/' f'{file_num}_ct.npy', 
                                volume['ct'][:, :, idx])
                    if 'spine_mask' in mask_criteria:
                        np.save(output_folder + 'spine/' f'{file_num}_spine.npy', 
                                volume['spine_mask'][:, :, idx])
                    if 'sternum_mask' in mask_criteria:
                        np.save(output_folder + 'sternum/' f'{file_num}_sternum.npy', 
                                volume['stern_mask'][:, :, idx])
                    if 'pelvis_mask' in mask_criteria:
                        np.save(output_folder + 'pelvis/' f'{file_num}_pelvis.npy', 
                                volume['pelvi_mask'][:, :, idx])
                    file_num += 1
                pbar.update(1)  # progress bar
            if plane == 'saggital':
                for idx in range(0, volume['ct'].shape[1]):
                    if (idx % 10 == 0) or (np.count_nonzero(volume['spine_mask'][:, idx, :]) > 0):
                        if 'spine_mask' in mask_criteria:
                            np.save(output_folder + 'spine/' f'{file_num}_spine.npy', 
                                    volume['spine_mask'][:, idx, :])
                        if 'ct' in mask_criteria:
                            np.save(output_folder + 'ct/' f'{file_num}_ct.npy', 
                                    volume['ct'][:, idx, :])
                        if 'sternum_mask' in mask_criteria:
                            np.save(output_folder + 'sternum/' f'{file_num}_sternum.npy', 
                                    volume['stern_mask'][:, idx, :])
                        if 'pelvis_mask' in mask_criteria:
                            np.save(output_folder + 'pelvis/' f'{file_num}_pelvis.npy', 
                                    volume['pelvi_mask'][:, idx, :])
                        file_num += 1
                pbar.update(1)  # progress bar
            if plane == 'coronal':
                for idx in range(0, volume['ct'].shape[0]):
                    if (idx % 10 == 0 or np.count_nonzero(volume['spine_mask'][:, idx, :]) > 0):
                        if 'ct' in mask_criteria:
                            np.save(output_folder + 'ct/' f'{file_num}_ct.npy', 
                                    volume['ct'][idx, :, :])
                        if 'spine_mask' in mask_criteria:
                            np.save(output_folder + 'spine/' f'{file_num}_spine.npy', 
                                    volume['spine_mask'][idx, :, :])
                        if 'sternum_mask' in mask_criteria:
                            np.save(output_folder + 'sternum/' f'{file_num}_sternum.npy', 
                                    volume['stern_mask'][idx, :, :])
                        if 'pelvis_mask' in mask_criteria:
                            np.save(output_folder + 'pelvis/' f'{file_num}_pelvis.npy', 
                                    volume['pelvi_mask'][idx, :, :])
                        file_num += 1
                pbar.update(1)  # progress bar


def plotSomeImages(figures, nrows = 1, ncols=1, interp='none'):
    ''' Plot a dictionary of figures.
    @params:
        figures = <title, figure> dictionary
        ncols = number of columns of subplots wanted in the display
        nrows = number of rows of subplots wanted in the figure
    https://stackoverflow.com/users/975979/gcalmettes
    '''
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10,10))
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap='cividis', interpolation=interp)
        axeslist.ravel()[ind].set_title(title, fontsize=15)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional

