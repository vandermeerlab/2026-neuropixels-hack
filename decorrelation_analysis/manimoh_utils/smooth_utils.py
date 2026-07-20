import numpy as np
from scipy.signal import windows

def smooth_peth(peth_values, bin_size, std=0.04):
    """
    Smooth PETH values using a gaussian kernel
    
    Parameters
    ----------
    peth_values : numpy.ndarray
        Array of PETH values to smooth
    bin_size : float
        Bin size in seconds
    std : float
        Standard deviation of gaussian kernel in seconds (default 1 ms)
    
    Returns
    -------
    numpy.ndarray
        Smoothed PETH values
    """
    # Convert std to number of bins
    std_bins = std / bin_size
    
    # Calculate window size (6 standard deviations, which captures 99.7% of Gaussian)
    # This is much more reasonable than 100 * std
    M = int(6 * std_bins)
    if M % 2 == 0:
        M += 1
    
    # Ensure minimum window size
    M = max(M, 3)
    
    # Don't let window be larger than signal (causes issues)
    M = min(M, len(peth_values))
    
    # Create and normalize gaussian window
    window = windows.gaussian(M=M, std=std_bins)
    window = window / window.sum()
    
    # Apply convolution
    smoothed = np.convolve(peth_values, window, mode='same')
    
    return smoothed