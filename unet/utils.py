from PIL import Image
import numpy as np
import scipy.io as scio
import os
import model
import torch
from pathlib import Path

def readUCharImage(fname, im_size=(512,512)):
    with open(fname, 'rb') as f:
        rawData = f.read()
        img = Image.frombytes('L', im_size, rawData)
        return np.array(img)
        
def readBinImage(fname, im_size=(512,512)):
    with open(fname, 'rb') as f:
        rawData = f.read()
        img = Image.frombytes('I', im_size, rawData)
        return np.array(img).T

def readFloatImage(fname, im_size=(512,512)):
    with open(fname, 'rb') as f:
        rawData = f.read()
        img = Image.frombytes('F', im_size, rawData)
        return np.array(img)

def match_files_from_patient(
        patient_idx, 
        day_selection, 
        ct_pt_folder='C:/.py_workspace/reveal/.reveal_data/CT-PT-Images', 
        mask_folder='C:/.py_workspace/reveal/.reveal_data/UCharImages-MultiClass',
        mode='ALL_DATA'):
    '''
    For a given patient/day, constructs a 2d list containing lines of 
    matching image filepaths. From this list of matching filepaths, a selection 
    of data "columns" are returned based on the 'mode' parameter passed to 
    this function. Manipulates columns as ndarray but returns as list.

    Modes:
    'CT_SPINE': matches ct files and spine segmentation labels.
        RETURNS: [[str(ct)], 
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
    'ALL_DATA': matches across all the data sources.
        RETURNS: [[str(ct)], 
                  [str(pt)],
                  [str(mask)],
                  [str(spine mask)], 
                  [str(sternum mask)], 
                  [str(pelvis mask)]]
    '''
    # Define CT organizational folder path.
    ct_path = Path('{}/P{:02d}/Day_{}/CT'
                   .format(ct_pt_folder, patient_idx, day_selection))
    # Get list of CT filenames from the Patient/Day arguments.
    ct_fnames = [file.__str__() for file in list(ct_path.glob('*'))]
    # Define PT organizational folder path.
    pt_path = Path('{}/P{:02d}/Day_{}/PT-Float'
                   .format(ct_pt_folder, patient_idx, day_selection))
    # Wrap 'mask_folder' as a Path object.
    mask_folder = Path(mask_folder)
    # Mask prefix used later in loop.
    mask_prefix = Path(mask_folder,'P{}_{}'.format(patient_idx, day_selection))
    # Define patient/day naming pattern for the mask files. 
    mask_pattern = 'P{}_{}_'.format(patient_idx, day_selection)
    # Get the list of ALL mask files (paths) in mask folder that match the 
    # mask pattern. List includes binary, spine, sternum, pelvis filepaths.
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
        
        # Append the various the filepaths to the 2d data list.
        data.append((ct_fname,          # data[:,0]
                     pt_fname,          # data[:,1]
                     spine_fname,       # data[:,2]
                     sternum_fname,     # data[:,3]
                     pelvis_fname))     # data[:,4]
    
    # Convert data list to nparray for easier column manipulations.
    data = np.array(data)
    # Use 'mode' argument to select the desired output from 'data'.
    if mode == 'CT_SPINE':
        output = np.array([data[:,0], data[:,2]]).T
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_spine_mask_v2(mat_fname, model, output_path):
    mat_data = scio.loadmat(mat_fname)
    CT = mat_data['ct_hounsfield']
    pixel_spacing = mat_data['ct_info'][0,0]['PixelSpacing']
    slice_spacing = mat_data['ct_info'][0,0]['SliceThickness']

    L = CT.shape[-1]
    spine_masks = np.zeros((L, 320, 320))
    ct_images = np.zeros((L, 320, 320))

    model.eval()
    with torch.no_grad():
        for idx in range(0, L):
            ct_data = CT[:, :, idx].astype(np.float32)
            ct_data = ct_data[96:-96, 96:-96]
            ct_images[idx, :, :] = ct_data

            cts = np.expand_dims(ct_data, axis=0)
            cts = np.expand_dims(cts, axis=0)
            cts = torch.from_numpy(cts).cuda()

            outputs = model(cts)
            masks_probs = torch.squeeze(F.sigmoid(outputs))
            mask = masks_probs.cpu().numpy()
            spine_masks[idx, :, :] = mask

    base_fname = os.path.basename(mat_fname)
    scio.savemat(os.path.join(output_path, 'spine-{}'.format(base_fname)), 
                 {'mask':np.transpose(spine_masks, [1, 2, 0]),
                  'ct': np.transpose(ct_images, [1, 2, 0]),
                  'pixel_spacing': pixel_spacing,
                  'slice_spacing': slice_spacing})