import numpy as np
import scipy as sp

def cdfFromHist(hist):
    """
    From a generic "histogram" array, returns the corresponding cumulative
    distribution array.
    """
    pdf = hist/sum(hist)
    cdf = np.zeros(len(hist))
    cumulative_sum = 0
    for idx, freq in enumerate(pdf):
        cdf[idx] = freq + cumulative_sum
        cumulative_sum = cumulative_sum + freq
    return cdf

def findClosestValueIdx(array, value):
    """
    Returns the index of the value closest to the given number.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def findCentroidSliceIdx(vol_mask):
    """
    Finds the approximate center slice by constructing a histogram of the number 
    of mask pixels in successive slices and finding the mean.
    
    @params:
    vol_data
    """
    area_hist = []
    n_slices = np.shape(vol_mask)[1]
    for slice_idx in range(n_slices):
        slice = vol_mask[:, slice_idx, :]
        slice_area = np.count_nonzero(slice)
        area_hist = np.append(area_hist, slice_area)
        cdf = cdfFromHist(area_hist)
    return findClosestValueIdx(cdf, 0.5)
    
def getLocalStack(vol_data, centerIdx, span, dim):
    """
    Retruns a 3D stack of slices
    """
    if dim == 0:
        stack = vol_data[(centerIdx-span):(centerIdx+span+1),: , :]
    elif dim == 1:
        stack = vol_data[:, (centerIdx-span):(centerIdx+span+1), :]
    else:
        stack = vol_data[:, :, (centerIdx-span):(centerIdx+span+1)]
    return stack

def addStack(stack, dim):
    """
    Adds image data along an dimension of a 3D image stack.
    """
    if dim == 0:
        aggregate = np.zeros((stack.shape[1], stack.shape[2]))
        for slice_idx in range(stack.shape[0]):
            aggregate = aggregate + stack[slice_idx, :, :]
    elif dim == 1:
        aggregate = np.zeros((stack.shape[0], stack.shape[2]))
        for slice_idx in range(stack.shape[1]):
            aggregate = aggregate + stack[:, slice_idx, :]
    else:
        aggregate = np.zeros((stack.shape[0], stack.shape[1]))
        for slice_idx in range(stack.shape[2]):
            aggregate = aggregate + stack[:, :, slice_idx]
    return aggregate