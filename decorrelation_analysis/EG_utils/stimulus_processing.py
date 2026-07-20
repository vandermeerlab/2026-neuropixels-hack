import pynapple as nap
import numpy as np
def get_odor_types_from_data(data):
    """
    Extract odor types from session metadata
    
    Parameters
    ----------
    data : pynapple object
        Loaded NWB data
    
    Returns
    -------
    odor_types : list
        List of odor types found in metadata (e.g., ['Odor A', 'Odor B', 'Odor C', 'Odor D', 'Odor E', 'Odor F'])
    """
    odor_types = []
    odor_codes = ['A', 'B', 'C', 'D', 'E', 'F']
    
    for key in data.keys():
        if 'odor' in key.lower():
            odor_name = key.replace(' ON', '')
            if odor_name not in odor_types:
                odor_types.append(odor_name)
    
    # If no odor types found, return default
    if not odor_types:
        print("Warning: No odor types found in metadata. Using default types: ['Odor A', 'Odor B', 'Odor C', 'Odor D', 'Odor E', 'Odor F']")
        odor_types = [f'Odor {code}' for code in odor_codes]
    
    return odor_types


def get_odor_onsets(data, meta_data, odor_types):
    """
    Extract odor onset times from NWB file for each odor type
    
    Parameters
    ----------
    data : pynapple object
        Loaded NWB data
    meta_data : dict
        Session metadata dictionary
    odor_types : list
        List of odor types to extract (e.g., ['Odor A', 'Odor B', 'Odor C', 'Odor D', 'Odor E', 'Odor F'])
    
    Returns
    -------
    odor_onsets : dict
        Dictionary mapping odor types to their onset IntervalSets
    """
    odor_onsets = {}
    
    for odor in odor_types:
        key = f'{odor} ON'
        
        if key in data:
            try:
                odor_onsets[odor] = nap.IntervalSet(data[key])
                print(f"Found {len(odor_onsets[odor])} onset(s) for '{odor}'")
            except Exception as e:
                print(f"Warning: Error extracting onsets for '{odor}': {e}")
                odor_onsets[odor] = None
        else:
            print(f"Warning: No onsets found for '{odor}'")
            odor_onsets[odor] = None
    
    return odor_onsets

def get_stimulus_onsets(data):
    """
    Extract stimulus onset times for rewards and airpuffs from NWB file
    
    Parameters
    ----------
    data : pynapple object
        Loaded NWB data
    
    Returns
    -------
    stimulus_onsets : dict
        Dictionary containing 1-dimensional time series for 'rewards' and 'airpuffs'
    """
    stimulus_onsets = {}
    
    # Get reward onsets
    if 'reward' in data:
        try:
            stimulus_onsets['reward'] = data['reward'].times()
            print(f"Found {len(stimulus_onsets['reward'])} reward onset(s)")
        except Exception as e:
            print(f"Warning: Error extracting reward onsets: {e}")
            stimulus_onsets['reward'] = None
    else:
        print("Warning: No reward onsets found")
        stimulus_onsets['reward'] = None
    
    # Get airpuff onsets
    if 'airpuff' in data:
        try:
            stimulus_onsets['airpuff'] = data['airpuff'].times()
            print(f"Found {len(stimulus_onsets['airpuff'])} airpuff onset(s)")
        except Exception as e:
            print(f"Warning: Error extracting airpuff onsets: {e}")
            stimulus_onsets['airpuff'] = None
    else:
        print("Warning: No airpuff onsets found")
        stimulus_onsets['airpuff'] = None
    
    return stimulus_onsets