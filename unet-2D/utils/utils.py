import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import whosmat, loadmat


def findMatchingVolumes(volume_folder = os.environ['REVEAL_DATA'] + '\\ct_mask_volumes\\', 
                        mask_criteria = ['ct', 'spine_mask']):
    """
    Search through the given .mat volume file list, find volumes matching all of 
    the given mask criteria.
    @return an iterable zip containing zip(filepath, [patient_idx, day_idx])
    """
    volume_idxs = [[a , b] for b in range(1,4) for a in range(1,23)]    # all REVEAL volumes
    return_file_list = []  # list to hold filepaths for volumes matching mask criteria
    return_idxs = []       # list to hold indexes for volumes matching mask criteria
    for vol_idx in volume_idxs:
        test_vol_file = (volume_folder + f'patient{vol_idx[0]}_day{vol_idx[1]}.mat')
        test_vol = whosmat(test_vol_file)
        if not all([criteria in [item[0] for item in test_vol] for criteria in mask_criteria]):
            continue
        return_file_list.append(test_vol_file)
        return_idxs.append(vol_idx)

    return zip(return_file_list, return_idxs)


def generateNpySlices(volume_file_zip, 
                      output_folder = os.environ['REVEAL_DATA'], 
                      mask_criteria = ['ct', 'spine_mask']):
    """
    Search through the given .mat volume file list, find volumes matching all of the given 
    mask list, from those matched volumes save 2D scans as .npy files.
    @params:
        file_index_zip = iterable containing filepath and patient index, from findMatchingVolumes().
        output_folder = filepath where the .npy files will be saved to. 
        volume_idxs = the [patient, day] indexes defining which volumes to operate on.
        masks = a list of any of ['spine_mask', 'stern_mask', 'pelvi_mask'] determining 
                the data required in the .mat file for the operation to be completed.
    @returns volumes_with_required_masks = a list of index pairs of the matched volumes.
    """
    file_num = 0
    if 'ct' in mask_criteria:
        os.makedirs(output_folder + 'ct/', exist_ok=True)
    if 'spine_mask' in mask_criteria:
        os.makedirs(output_folder + 'spine/', exist_ok=True)
    if 'sternum_mask' in mask_criteria:
        os.makedirs(output_folder + 'sternum/', exist_ok=True)
    if 'pelvis_mask' in mask_criteria:
        os.makedirs(output_folder + 'pelvis/', exist_ok=True)

    for file, idx in volume_file_zip:
        volume = loadmat(file)
        volume_ct = volume['ct']
        for idx in range(0, volume_ct.shape[-1]):
            if 'ct' in mask_criteria:
                np.save(output_folder + 'ct/' f'{file_num}_ct.npy', volume_ct[:, :, idx])
            if 'spine_mask' in mask_criteria:
                volume_spine = volume['spine_mask']
                np.save(output_folder + 'spine/' f'{file_num}_spine.npy', volume_spine[:, :, idx])
            if 'sternum_mask' in mask_criteria:
                volume_stern = volume['stern_mask']
                np.save(output_folder + 'sternum/' f'{file_num}_sternum.npy', volume_stern[:, :, idx])
            if 'pelvis_mask' in mask_criteria:
                volume_pelvi = volume['pelvi_mask']
                np.save(output_folder + 'pelvis/' f'{file_num}_pelvis.npy', volume_pelvi[:, :, idx])
            file_num += 1

def plotSomeImages(figures, nrows = 1, ncols=1):
    '''
    Plot a dictionary of figures.
    @params:
        figures = <title, figure> dictionary
        ncols = number of columns of subplots wanted in the display
        nrows = number of rows of subplots wanted in the figure
    https://stackoverflow.com/users/975979/gcalmettes
    '''
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10,10))
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap='cividis')
        axeslist.ravel()[ind].set_title(title, fontsize=15)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)