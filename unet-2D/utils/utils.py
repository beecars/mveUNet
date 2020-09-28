import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

def readUCharImage(fname, im_size=(512,512), as_tensor=False):
    '''
    Reads in a unsigned character image file to a numpy array or a tensor.
    @params: 
        fname = the string filepath of the Uchar image
        im_size = the expected size of the Uchar image
        as_tensor = boolean, if False, numpy array will be returned. If true, 
                    torch.Tensor will be returned. 
    @returns:
        a 2D numpy array or 2D torch.Tensor containing image data
    '''
    with open(fname, 'rb') as f:
        rawData = f.read()
        img = Image.frombytes('L', im_size, rawData)
        if as_tensor:
            return torch.from_numpy(np.array(img)).float()
        else:
            return np.array(img)
        
def readBinImage(fname, im_size=(512,512), as_tensor=False):
    '''
    Reads in a binary image file to a numpy array or a tensor.
    @params: 
        fname = the string filepath of the binary image
        im_size = the expected size of the binary image
        as_tensor = boolean, if False, numpy array will be returned. If true, 
                    torch.Tensor will be returned. 
    @returns:
        a 2D numpy array or 2D torch.Tensor containing image data
    '''
    with open(fname, 'rb') as f:
        rawData = f.read()
        img = Image.frombytes('I', im_size, rawData)
        if as_tensor:
            return torch.from_numpy(np.array(img).T).float()
        else:
            return np.array(img).T

def readFloatImage(fname, im_size=(512,512)):
    '''
    Reads in a float image file to a numpy array or a tensor.
    @params: 
        fname = the string filepath of the float image
        im_size = the expected size of the float image
        as_tensor = boolean, if False, numpy array will be returned. If true, 
                    torch.Tensor will be returned. 
    @returns:
        a 2D numpy array or 2D torch.Tensor containing image data
    '''
    with open(fname, 'rb') as f:
        rawData = f.read()
        img = Image.frombytes('F', im_size, rawData)
        return np.array(img)

def vol_from_seq(ct_data, mode='bin'):
    '''
    Creates 3D volume array from ordered list of CT scans. 
    @params:
        ct_data = a specific data list generated for the REVEAL CT data
                  by using the matchFilesFromPatients() function. 
        mode = 'bin' or 'uchar' to change file type
    @returns:
        a 3D volume nparray
    '''
    n_cts = len(ct_data)
    if mode == 'bin':
        img_size = readBinImage(ct_data[0]).shape
        volume = np.empty([img_size[0], img_size[1], n_cts])
        for i, file in enumerate(ct_data):
            scan = readBinImage(file)
            volume[:, :, i] = scan
    else:
        img_size = readUCharImage(ct_data[0]).shape
        volume = np.empty([img_size[0], img_size[1], n_cts])
        for i, file in enumerate(ct_data):
            scan = readUCharImage(file)
            volume[:, :, i] = scan
    return volume

def masks2classes(masks):
    ''' 
    From multiple masks representing multiple classes, creates a single-mask 
    representation where "pixel" value is an integer class label. Input is an
    array of binary image masks, output is a single multi-class mask.
    @params:
        masks = an n-length list of mask image data
    @returns:
        a single 2D numpy array with pixel values 1-n reprensenting n classes
        from n masks
    '''
    object_class = 1
    mask_size = np.shape(masks)[1:3]
    target = np.zeros(mask_size)
    for mask in masks:
        target = target + mask * object_class
        object_class = object_class + 1
    return target

def matchFilesFromPatient(patient_idx, 
                          day_selection, 
                          data_folder = os.environ['REVEAL_DATA'],
                          mode='ALL_DATA',
                          no_empties=False):
    '''
    THIS FUNCTION COMPLETELY RELIES ON AN INCREDIBLY SPECIFIC FILE STRUCTURE AND
    NAMING CONVENTIONS, SPECIFIC TO THE REVEAL PROJECT.

    For a given patient/day, constructs a 2d list containing lines of 
    matching image filepaths. From this list of matching filepaths, a selection 
    of data "columns" are returned based on the 'mode' parameter passed to 
    this function. Manipulates columns as ndarray but returns as list.
    @params:
        patient_idx = the patient number used to parse and return CT data
        day_idx = the day index used in conjuction with patient index to parse
                  and return CT data
        data_folder = the location where the INCREDIBLY SPECIFIC data structure
                      lives
        mode = see modes below
        no_empties = for the CT_SPINE mode, returns only CT images with non-
                     empty mask files (edge case for testing).
    @returns:
        a 2D list of structure depending on the modes listed below

    Modes:
    'CT_ONLY': gets only the CT files.
        RETURNS: [[str(ct)]]
    'CT_SPINE': matches ct files and spine segmentation labels.
        RETURNS: [[str(ct)], 
                  [str(spine mask)]]
    'CT_PT_SPINE': matches ct and pt files with spine segmentation labels.
        RETURNS: [[str(ct)], 
                  [str(pt)],
                  [str(spine mask)]]
    'CT_SPINE_STERNUM_PELVIS': matches ct files and multiclass segmentations.
        RETURNS: [[str(ct)], 
                  [str(spine mask)], 
                  [str(sternum mask)], 
                  [str(pelvis mask)]]
    'CT_SPINE_STERNUM': matches ct files and only spine, sternum segmentations.
        RETURNS: [[str(ct)], 
                  [str(spine mask)], 
                  [str(sternum mask)]]
    'CT_PT': matches ct files and pt files.
        RETURNS: [[str(ct)],
                  [str(pt)]]
    'SPINE_MASK_ONLY': gets only the spine ground truth masks.
        RETURNS: [[str(spine mask)]]
    'ALL_DATA': matches across all the data sources.
        RETURNS: [[str(ct)], 
                  [str(pt)],
                  [str(mask)],
                  [str(spine mask)], 
                  [str(sternum mask)], 
                  [str(pelvis mask)]]
    '''
    # Define organizational folder structure for ct, pt, mask files.
    ct_pt_folder = Path(data_folder, 'CT-PT-Images')
    ct_path = Path('{}/P{:02d}/Day_{}/CT'
                   .format(ct_pt_folder, patient_idx, day_selection))
    ct_fnames = [file.__str__() for file in list(ct_path.glob('*'))]
    
    pt_path = Path('{}/P{:02d}/Day_{}/PT-Float'
                   .format(ct_pt_folder, patient_idx, day_selection))
    
    mask_folder = Path(data_folder, 'UCharImages-Multiclass')
    mask_prefix = Path(mask_folder,'P{}_{}'.format(patient_idx, day_selection))
    mask_pattern = 'P{}_{}_'.format(patient_idx, day_selection)
    mask_fnames = [file.__str__() for file in 
                   list(mask_folder.glob(mask_pattern + '*.uchar'))]

    # Initialize 'data' list to be filled by for loop.
    data = []
    for ct_fname in ct_fnames:
        # Get ct_fname file index (the last 3 numbers in the file name...)
        # Used to identify matching data.
        slice_idx = ct_fname[-7:].split('.')[0]

        # Get matching pt_fname from index.
        pt_fname = Path('{}/P{:02d}_{}_{}_Pelvis.float'
                        .format(pt_path, patient_idx, day_selection, slice_idx)
                       ).__str__()
        
        # Look for potential SPINE mask file with ct index.
        maybe_spine_fname = ('{}_{}_Spine.uchar'
                             .format(mask_prefix, slice_idx))
        if maybe_spine_fname in mask_fnames:
            spine_fname = maybe_spine_fname
        elif no_empties:    
            # Special case to return only entries where spine mask is avail.
            continue       
        else:
            spine_fname = Path('{}/empty.uchar'
                               .format(mask_folder)).__str__()
        
        # Look for potential STERNUM mask file with ct index.
        maybe_sternum_fname = ('{}_{}_Sternum.uchar'
                               .format(mask_prefix, slice_idx))
        if maybe_sternum_fname in mask_fnames:
            sternum_fname = maybe_sternum_fname
        else:
            sternum_fname = Path('{}/empty.uchar'
                                 .format(mask_folder)).__str__()
        
        # Look for potential PELVIS mask file with ct index. 
        maybe_pelvis_fname = ('{}_{}_Pelvis.uchar'
                              .format(mask_prefix, slice_idx))
        if maybe_pelvis_fname in mask_fnames:
            pelvis_fname = maybe_pelvis_fname
        else:
            pelvis_fname = Path('{}/empty.uchar'
                                .format(mask_folder)).__str__()
        
        data.append((ct_fname,          # data[:,0]
                     pt_fname,          # data[:,1]
                     spine_fname,       # data[:,2]
                     sternum_fname,     # data[:,3]
                     pelvis_fname))     # data[:,4]
    
    # Convert data list to nparray for easier column manipulations.
    data = np.array(data)
    if len(data) == 0:
        return
        
    # Use 'mode' argument to select the desired output from 'data'.
    if mode == 'CT_ONLY':
        output = np.array(data[:,0])
    elif mode == 'CT_SPINE':
        output = np.array([data[:,0], data[:,2]]).T
    elif mode == 'SPINE_MASK_ONLY':
        output = np.array(data[:,2])
    elif mode == 'CT_PT_SPINE':
        output = np.array([data[:,0], data[:,1], data[:,2]]).T                    
    elif mode == 'CT_SPINE_STERNUM_PELVIS':
        output = np.array([data[:,0], data[:,2], data[:,3], data[:,4]]).T
    elif mode == 'CT_SPINE_STERNUM':
        output = np.array([data[:,0], data[:,2], data[:,3]]).T
    elif mode == 'CT_PT':
        output = np.array([data[:,0], data[:,1]]).T
    elif mode == 'ALL_DATA':
        output = np.array([data[:,0], data[:,1], data[:,2], 
                           data[:,3], data[:,4]]).T
    else:
        print('ERROR: Enter valid mode parameter')
        return

    return output.tolist()

def matchFilesFromPatients(patient_idxs, day_idxs, mode = 'CT_SPINE'):
    ''' 
    Wraps matchFilesFromPatient() to iterate over any number of patients 
    and days. Used to fill a training or validation Dataset like CTMaskDataset 
    or CTSequenceDataset.
    @
    '''
    
    files = []
    volume_idxs = []
    for patient_idx in patient_idxs:
        for day_idx in day_idxs:
            matched_data = matchFilesFromPatient(patient_idx, 
                                                 day_idx, 
                                                 mode = mode, 
                                                 no_empties = True)
            try:
                files.extend(matched_data)
            except:
                print(f'No data found for patient {patient_idx} on day {day_idx}')
            else: 
                # if files is extended, record the patient/day
                volume_idxs.append([patient_idx, day_idx])

    return files, volume_idxs

def generateCrossvalidationSets():
    '''
    Generates cross-validation datasets from the available reveal CT data.
    This could be done more elegantly, like with random sampling to 
    populate the folds, but it would take a few hours to generalize and it's
    not worth it at 2:11am. The key is that a patient should never appear in 
    more than one fold, and each fold should have the same number of volumes. 
    '''
    # k folds
    fold1 = [1, 2]            # 5 total volumes w/ spine seg.
    fold2 = [3, 6]            # 5 total volumes w/ spine seg.
    fold3 = [4, 7, 8]         # 5 total volumes w/ spine seg.
    fold4 = [5, 10, 11]       # 5 total volumes w/ spine seg.
    fold5 = [9, 12, 13]       # 5 total volumes w/ spine seg.
    fold6 = [14, 16, 17]      # 5 total volumes w/ spine seg.
    fold7 = [18, 19, 20]      # 5 total volumes w/ spine seg.

    # remember the val_idxs are used almost exclusively in the predict_volumes
    # function to parse the ct_data by patient and day index...
    val_data1, val1_idxs = matchFilesFromPatients(fold1, range(1,4))
    val_data2, val2_idxs = matchFilesFromPatients(fold2, range(1,4))
    val_data3, val3_idxs = matchFilesFromPatients(fold3, range(1,4))
    val_data4, val4_idxs = matchFilesFromPatients(fold4, range(1,4))
    val_data5, val5_idxs = matchFilesFromPatients(fold5, range(1,4))
    val_data6, val6_idxs = matchFilesFromPatients(fold6, range(1,4))
    val_data7, val7_idxs = matchFilesFromPatients(fold7, range(1,4))
    
    # one fold absent from each split
    trn_split1 = fold2 + fold3 + fold4 + fold5 + fold6 + fold7
    trn_split2 = fold1 + fold3 + fold4 + fold5 + fold6 + fold7
    trn_split3 = fold1 + fold2 + fold4 + fold5 + fold6 + fold7
    trn_split4 = fold1 + fold2 + fold3 + fold5 + fold6 + fold7
    trn_split5 = fold1 + fold2 + fold3 + fold4 + fold6 + fold7
    trn_split6 = fold1 + fold2 + fold3 + fold4 + fold5 + fold7
    trn_split7 = fold1 + fold2 + fold3 + fold4 + fold5 + fold6

    train_data1, _ = matchFilesFromPatients(trn_split1, range(1,4))
    train_data2, _ = matchFilesFromPatients(trn_split2, range(1,4))
    train_data3, _ = matchFilesFromPatients(trn_split3, range(1,4))
    train_data4, _ = matchFilesFromPatients(trn_split4, range(1,4))
    train_data5, _ = matchFilesFromPatients(trn_split5, range(1,4))
    train_data6, _ = matchFilesFromPatients(trn_split6, range(1,4))
    train_data7, _ = matchFilesFromPatients(trn_split7, range(1,4))

    val_datas = [val_data1, val_data2, val_data3, val_data4, 
                 val_data5, val_data6, val_data7]
    
    val_idxs = [val1_idxs, val2_idxs, val3_idxs, val4_idxs, 
                val5_idxs, val6_idxs, val7_idxs]

    train_datas = [train_data1, train_data2, train_data3, train_data4, 
                   train_data5, train_data6, train_data7]

    return val_datas, val_idxs, train_datas

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
