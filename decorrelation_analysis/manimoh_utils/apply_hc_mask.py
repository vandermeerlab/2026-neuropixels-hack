def apply_hc_mask(electrode_table, thresh, spikes, meta_data):
    """
    Create a mask for hippocampal units based on depth offset from best SWR channels.
    
    Parameters:
    -----------
    electrode_table : pd.DataFrame
        Table containing electrode information including labels and depths
    thresh : float
        Distance threshold in microns
    spikes : dict
        Dictionary containing spike data with 'global_id' and 'depth' information
    meta_data : dict
        Dictionary containing best SWR channel information for each probe
        
    Returns:
    --------
    list
        Boolean mask indicating whether each unit is within threshold distance of best SWR channel
    """
    hc_mask = []
    
    for key in spikes:
        device, shank, uid = spikes['global_id'][key].strip().split('.')
        
        if (device == 'imec0') and not (meta_data['imec0_best_SWR_channel'] is None):
            offset = spikes['depth'][key] - \
                electrode_table[electrode_table['label'] == device+'.ap#'+meta_data['imec0_best_SWR_channel']]['depth'].to_numpy()[0]
        elif (device == 'imec1') and not (meta_data['imec1_best_SWR_channel'] is None):
            offset = spikes['depth'][key] - \
                electrode_table[electrode_table['label'] == device+'.ap#'+meta_data['imec1_best_SWR_channel']]['depth'].to_numpy()[0]
        else:
            offset = thresh + 1
            
        hc_mask.append(abs(offset) <= thresh)
    
    return hc_mask
