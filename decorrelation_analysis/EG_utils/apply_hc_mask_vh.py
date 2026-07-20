def apply_hc_mask_vh(electrode_table, threshDH, threshVH, spikes, meta_data):
    """
    Create a mask for hippocampal units based on depth offset from best SWR channels.
    
    Parameters:
    -----------
    electrode_table : pd.DataFrame
        Table containing electrode information including labels and depths
    threshDH: float
        Distance threshold in microns - isolate dorsal hippocampus neurons
    threshVH: float
        Distance threshold in microns - isolate ventral hippocampus neurons
    spikes : dict
        Dictionary containing spike data with 'global_id' and 'depth' information
    meta_data : dict
        Dictionary containing best SWR channel information for each probe
        Keys expected: imec0_dorsal_SWR_channel, imec0_ventral_SWR_channel,
                      imec1_dorsal_SWR_channel, imec1_ventral_SWR_channel
        
    Returns:
    --------
    list
        Boolean mask indicating whether each unit is within threshold distance of 
        any best SWR channel (dorsal or ventral)
    """
    hc_mask = []
    
    for key in spikes:
        device, shank, uid = spikes['global_id'][key].strip().split('.')
        # unit_depth = spikes['depth'][key]-175  # Adjust for probe tip offset (correct LFP depth, incorrect unit depths)
        unit_depth = spikes['depth'][key] # No adjustment for probe tip offset, since unit depths are already corrected in NWB file
        is_hc_unit = False
        
        # Check dorsal SWR channel
        dorsal_key = f'{device}_dorsal_SWR_channel'
        if (dorsal_key in meta_data) and (meta_data[dorsal_key] is not None):
            try:
                dorsal_depth = electrode_table[
                    electrode_table['label'] == f'{device}.ap#{meta_data[dorsal_key]}'
                ]['depth'].to_numpy()[0]
                dorsal_offset = abs(unit_depth - dorsal_depth)
                if dorsal_offset <= threshDH:
                    is_hc_unit = True
            except IndexError:
                pass
        
        # Check ventral SWR channel
        ventral_key = f'{device}_ventral_SWR_channel'
        if (ventral_key in meta_data) and (meta_data[ventral_key] is not None):
            try:
                ventral_depth = electrode_table[
                    electrode_table['label'] == f'{device}.ap#{meta_data[ventral_key]}'
                ]['depth'].to_numpy()[0]
                ventral_offset = abs(unit_depth - ventral_depth)
                if ventral_offset <= threshVH:
                    is_hc_unit = True
            except IndexError:
                pass
        
        hc_mask.append(is_hc_unit)
    
    return hc_mask