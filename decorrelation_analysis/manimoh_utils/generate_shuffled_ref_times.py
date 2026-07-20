import numpy as np

def generate_shuffled_ref_times(original_times, epoch_start, epoch_end, tmin, tmax, n_shuffles=1000):
    """
    Generate shuffled reference times preserving inter-trial interval distribution.
    
    Parameters:
    -----------
    original_times : array-like
        Original trial start times
    epoch_start : float
        Start time of the epoch
    epoch_end : float
        End time of the epoch
    tmin : float
        Time window start (negative for baseline)
    tmax : float
        Time window end for response
    n_shuffles : int
        Number of shuffles to generate
    
    Returns:
    --------
    np.ndarray
        Array of shape (n_shuffles, n_trials) containing shuffled reference times
    """
    # Convert to numpy array and ensure 1D
    original_times = np.asarray(original_times).flatten()
    n_trials = len(original_times)
    
    # Calculate original inter-trial intervals
    original_itis = np.diff(original_times)
    
    # Define valid time range for first trial
    valid_start = epoch_start - tmin
    valid_end = epoch_end - tmax
    
    # Initialize output array with correct shape
    shuffled_times = np.zeros((n_shuffles, n_trials))
    
    for shuffle_idx in range(n_shuffles):
        # Randomly pick first trial time
        first_time = np.random.uniform(valid_start, valid_end - np.sum(original_itis))
        
        # Initialize this shuffle's times with the first time
        these_times = np.zeros(n_trials)
        these_times[0] = first_time[0]
        
        # Generate subsequent times using original ITIs
        for i, iti in enumerate(original_itis):
            these_times[i + 1] = these_times[i] + iti
        
        # Assign to output array
        shuffled_times[shuffle_idx, :] = these_times
        
    return shuffled_times