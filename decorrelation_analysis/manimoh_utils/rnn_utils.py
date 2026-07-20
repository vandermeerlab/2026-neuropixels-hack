import numpy as np
import pynapple as nap

def prepare_data_for_rnn(spikes, onsets, bin_size=0.05, time_window=(-0.5, 0.5)):
    """
    Prepare neural data for RNN by creating sequences of binned spikes for each trial
    
    Parameters:
    -----------
    spikes : pynapple.TsGroup
        Spike times for all neurons
    onsets : dict
        Dictionary with stimulus onset times for each stimulus type
    bin_size : float
        Size of time bins in seconds
    time_window : tuple
        Time window around stimulus onset to include (in seconds)
    
    Returns:
    --------
    sequences : np.array
        Array of shape (n_trials, n_time_bins, n_neurons) containing binned spikes
    labels : np.array
        Vector of trial labels (stimulus identities)
    """
    
    tmin, tmax = time_window
    all_sequences = []
    all_labels = []
    
    # For each stimulus
    for stimulus, onset_times in onsets.items():
        tcenter = onset_times.start + np.mean(time_window)
        trial_peth = nap.compute_perievent(timestamps = spikes, tref=nap.Ts(tcenter, time_units="s", time_support=spikes.time_support), \
            minmax=0.5*(tmax-tmin), time_unit="s")
        all_responses = []
        # For each neuron
        for unit_idx in spikes.keys():
            response = (trial_peth[unit_idx].count(bin_size))/bin_size 
            all_responses.append(response.values)
        
        all_units_responses = np.stack(all_responses, axis=0) # shape (neurons x time_bins x trials)
        trialwise_units_responses = np.transpose(all_units_responses, (2, 1, 0)) # shape (trials x time_bins x neurons)
        all_sequences.append(trialwise_units_responses)
        all_labels.extend([stimulus] * len(onset_times))
    
    # Stack sequences
    sequences = np.vstack(all_sequences)
    labels = np.array(all_labels)
    
    return sequences, labels
