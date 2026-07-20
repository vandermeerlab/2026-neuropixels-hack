import pynapple as nap
import numpy as np
from .smooth_utils import smooth_peth

def trialify_and_bin_spikes(spike_data, trial_ref_times, time_win, bin_size, to_smooth=False, smooth_std=0.04):
    """
    Function to bin spikes in a time window around reference times
    
    Args:
        spike_data (nap.TimeSeries): TimeSeries Group object containing spike times
        trial_ref_times (ndarray): numpy array containing 1 x n timepoints, 1 for each trial
        time_win (tuple): Tuple containing the time window around trial_ref_times
        bin_size (float): Size of each bin (Make sure this is in the same units as the time window)
        to_smooth (bool): Whether to smooth the PETH
        smooth_std (float): Standard deviation of gaussian kernel in seconds (default 40ms)
    
    Returns:
        np.ndarray: Array of binned spike counts
    """
    # idx = np.where(spike_data['global_id'].contains(unit_id))[0][0]
    
    out_peth = nap.compute_perievent(timestamps=spike_data, tref=nap.Ts(t=trial_ref_times, time_units="s",
        time_support=spike_data.time_support), minmax=(time_win[0]-bin_size/2, time_win[1]+bin_size/2), 
                                     time_unit="s", )
    out_fr = out_peth.count(bin_size)/bin_size
    if to_smooth:
        out_fr_mean = np.mean(out_fr, axis=1)
        out_fr = smooth_peth(out_fr_mean.values, bin_size, std=smooth_std)
    return out_fr.values
    