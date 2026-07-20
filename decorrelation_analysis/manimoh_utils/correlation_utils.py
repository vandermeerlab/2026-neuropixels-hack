import numpy as np

def analyze_correlation_structure(corr_matrix, start_idx, stop_idx, wiggle_room=0):
    """
    Streamlined analysis of correlation structure within a symmetric matrix.
    For each column, calculates key metrics comparing left and right sections.
    
    Args:
        corr_matrix (numpy.ndarray): A symmetric correlation matrix
        start_idx (int): Starting index (inclusive)
        stop_idx (int): Stopping index (inclusive)
        wiggle_room (int, optional): Number of elements to avoid near the diagonal
                                     and matrix boundaries for more stable analysis. Default is 0.
        
    Returns:
        dict: Dictionary containing the following for each column index:
              - 'col_diff': Left column mean - Right column mean
              - 'left_mean': Mean of all elements in left half
              - 'right_mean': Mean of all elements in right half
              - 'row_diff_mean': Mean of (row right mean - row left mean) across all rows
    """
    n = corr_matrix.shape[0]
    
    # Verify the range is valid
    if start_idx < 0 or stop_idx >= n or start_idx >= stop_idx:
        raise ValueError(f"Invalid range: [{start_idx}, {stop_idx}]")
    
    # Verify matrix is symmetric about main diagonal
    if not np.allclose(corr_matrix, corr_matrix.T):
        raise ValueError("Matrix is not symmetric about main diagonal")
    
    # Initialize result dictionary
    result = {
        'col_diff': [],        # Left column mean - Right column mean
        'left_mean': [],       # Mean of all elements in left half
        'right_mean': [],      # Mean of all elements in right half
        'row_diff_mean': []    # Mean of (row right mean - row left mean) across all rows
    }
    
    # Process each potential boundary column
    for idx in range(start_idx, stop_idx):
        # ---- Calculate column-wise difference ----
        # For left column (idx), get upper triangular elements
        left_col_vals = corr_matrix[:idx, idx]
        
        # For right column (idx+1), get upper triangular elements
        right_col_vals = corr_matrix[:idx+1, idx+1]
        
        # Calculate column means
        left_col_mean = np.mean(left_col_vals) if len(left_col_vals) > 0 else 0
        right_col_mean = np.mean(right_col_vals) if len(right_col_vals) > 0 else 0
        
        # Calculate column difference (left - right)
        col_diff = left_col_mean - right_col_mean
        result['col_diff'].append((idx, col_diff))
        
        # ---- Calculate half means ----
        # Left half elements (respecting wiggle_room)
        left_half_vals = []
        for i in range(start_idx, idx+1):
            for j in range(i+1, idx+1):
                # Only include elements far enough from diagonal and boundary
                if j - i > wiggle_room and j <= idx - wiggle_room:
                    left_half_vals.append(corr_matrix[i, j])
        
        # Right half elements (respecting wiggle_room)
        right_half_vals = []
        for i in range(start_idx, idx+1):
            for j in range(idx+1, stop_idx+1):
                # Only include elements far enough from diagonal and boundary
                if j - i > wiggle_room and j <= stop_idx - wiggle_room:
                    right_half_vals.append(corr_matrix[i, j])
        
        # Calculate half means
        left_half_mean = np.mean(left_half_vals) if left_half_vals else np.nan
        right_half_mean = np.mean(right_half_vals) if right_half_vals else np.nan
        
        # Store the half means
        result['left_mean'].append((idx, left_half_mean))
        result['right_mean'].append((idx, right_half_mean))
        
        # ---- Calculate row-wise differences ----
        row_diffs = []
        
        for row_idx in range(start_idx, stop_idx):
            # Skip rows that are too close to boundaries
            if row_idx < start_idx + wiggle_room:
                continue
                
            # Left half elements for this row
            row_left_vals = []
            for j in range(row_idx+1+wiggle_room, idx+1-wiggle_room):
                if j > row_idx and j - row_idx > wiggle_room:
                    row_left_vals.append(corr_matrix[row_idx, j])
            
            # Right half elements for this row
            row_right_vals = []
            for j in range(idx+1, stop_idx+1-wiggle_room):
                if j > row_idx and j - row_idx > wiggle_room:
                    row_right_vals.append(corr_matrix[row_idx, j])
            
            # Skip if not enough elements in either half
            if not row_left_vals or not row_right_vals:
                continue
            
            # Calculate row means
            row_left_mean = np.mean(row_left_vals)
            row_right_mean = np.mean(row_right_vals)
            
            # Calculate row difference (right - left)
            row_diff = row_right_mean - row_left_mean
            row_diffs.append(row_diff)
        
        # Calculate mean of row differences
        row_diff_mean = np.mean(row_diffs) if row_diffs else np.nan
        
        # Add row difference mean
        result['row_diff_mean'].append((idx, row_diff_mean))
    
    return result

def find_decorrelation_boundary(results_dict, diff_thresh, drop_thresh=None, left_thresh=None, right_thresh=None):
    """
    Finds the column index that represents a decorrelation boundary based on multiple criteria.
    
    Args:
        results_dict (dict): Dictionary of results from analyze_correlation_structure
                            containing 'col_diff', 'row_diff_mean', 'left_mean', and 'right_mean'
        diff_thresh (float): The threshold value that row_diff_mean must not exceed
        drop_thresh (float, optional): If provided, only column indices with col_difference
                                       less than this threshold will be considered
        left_thresh (float, optional): If provided, only consider indices where left_mean
                                       is greater than this threshold
        right_thresh (float, optional): If provided, only consider indices where right_mean
                                        is less than this threshold
        
    Returns:
        tuple: (idx, col_diff_value, row_diff_mean, left_mean, right_mean) if a suitable index is found,
               None if no index satisfies all conditions
    """
    
    # Verify the required keys exist in the results dictionary
    required_keys = ['col_diff', 'row_diff_mean', 'left_mean', 'right_mean']
    for key in required_keys:
        if key not in results_dict:
            raise ValueError(f"Required key '{key}' not found in results dictionary. "
                           f"Available keys: {list(results_dict.keys())}")
    
    # Extract the col_diff values
    col_diffs = results_dict['col_diff']
    
    # Filter out columns with col_diff >= drop_thresh if drop_thresh is provided
    if drop_thresh is not None:
        col_diffs = [(idx, val) for idx, val in col_diffs if val < drop_thresh]
        
        # If no columns remain after filtering, return None
        if not col_diffs:
            return None
    
    # Sort the (possibly filtered) col_diffs by the difference value
    sorted_col_diffs = sorted(col_diffs, key=lambda x: x[1])
    
    # Create lookup dicts for the metrics
    row_diff_lookup = {idx: val for idx, val in results_dict['row_diff_mean']}
    left_mean_lookup = {idx: val for idx, val in results_dict['left_mean']}
    right_mean_lookup = {idx: val for idx, val in results_dict['right_mean']}
    
    # Filter col_diffs based on left_mean and right_mean thresholds
    filtered_col_diffs = []
    for idx, col_diff_val in sorted_col_diffs:
        left_mean = left_mean_lookup.get(idx, np.nan)
        right_mean = right_mean_lookup.get(idx, np.nan)
        
        # Skip if any value is NaN
        if np.isnan(left_mean) or np.isnan(right_mean):
            continue
        
        # Apply left_thresh and right_thresh as filtering conditions
        left_mean_cond = left_mean > left_thresh if left_thresh is not None else True
        right_mean_cond = right_mean < right_thresh if right_thresh is not None else True
        
        if left_mean_cond and right_mean_cond:
            filtered_col_diffs.append((idx, col_diff_val))
    
    # If no indices remain after filtering, return None
    if not filtered_col_diffs:
        return None
    
    # Iterate through the filtered col_diffs
    for idx, col_diff_value in filtered_col_diffs:
        # Get the corresponding metric values
        row_diff_mean = row_diff_lookup.get(idx, np.nan)
        left_mean = left_mean_lookup.get(idx, np.nan)
        right_mean = right_mean_lookup.get(idx, np.nan)
        
        # Skip if row_diff_mean is NaN
        if np.isnan(row_diff_mean):
            continue
        
        # Check row_diff_mean threshold condition
        if row_diff_mean <= diff_thresh:
            return (idx, col_diff_value, row_diff_mean, left_mean, right_mean)
    
    # If no index satisfies all conditions
    return None

def apply_decorr_mask(spikes, meta_data, block='block1'):
    """
    Create three masks for neurons based on their decoupling times for a specific block.
    
    Parameters:
    -----------
    spikes : dict
        Dictionary containing spike data with 'global_id' information
    meta_data : dict
        Dictionary containing decoupling times information for each probe/shank
    block : str, optional
        Block identifier, either 'block1' or 'block2' (default: 'block1')
        
    Returns:
    --------
    tuple of three lists
        noDecorr_mask: Boolean mask where True indicates decoupling time of -1
        noQ_mask: Boolean mask where True indicates decoupling time of -2
        decorr_mask: Boolean mask where True indicates decoupling time is not -1 or -2
    """
    if block not in ['block1', 'block2']:
        raise ValueError("block parameter must be either 'block1' or 'block2'")
    
    # Determine which index to use based on the block
    index = 1 if block == 'block1' else 3
    
    noDecorr_mask = []
    noQ_mask = []
    decorr_mask = []
    
    for key in spikes:
        device, shank, uid = spikes['global_id'][key].strip().split('.')
        
        # Construct the key for meta_data
        meta_key = f"{device}_{shank}_decorr_times"
        
        # Check if the key exists and if we have valid data
        if meta_key in meta_data and isinstance(meta_data[meta_key], list) and len(meta_data[meta_key]) > index:
            decorr_time = meta_data[meta_key][index]
            
            # Create the three masks based on the conditions
            noDecorr_mask.append(decorr_time == -1)  # True if time is -1
            noQ_mask.append(decorr_time == -2)       # True if time is -2
            # True if time is not -1 and not -2 (i.e., actual decoupling time)
            decorr_mask.append(decorr_time != -1 and decorr_time != -2)
        else:
            # If we can't find data, default all masks to False for this unit
            noDecorr_mask.append(False)
            noQ_mask.append(False)
            decorr_mask.append(False)
    
    return noDecorr_mask, noQ_mask, decorr_mask

def get_new_block_boundaries(meta_data, block='block1', boundary_thresh=0.1):
    """
    Calculate new block boundaries based on decoupling times from meta_data.
    
    Parameters:
    -----------
    meta_data : dict
        Dictionary containing experiment metadata including block times and decoupling times
    block : str, optional
        Block identifier, either 'block1' or 'block2' (default: 'block1')
    boundary_thresh : float, optional
        Threshold as fraction of block duration to determine boundary adjustment (default: 0.1)
        
    Returns:
    --------
    tuple
        (new_start, new_end): Updated block boundaries
    """
    if block not in ['block1', 'block2']:
        raise ValueError("block parameter must be either 'block1' or 'block2'")
    
    # Determine which index to use based on the block
    index = 1 if block == 'block1' else 3
    
    # Get original block start and end times
    start_key = f"{block}start"
    end_key = f"{block}end"
    
    if start_key not in meta_data or end_key not in meta_data:
        raise ValueError(f"Block times for {block} not found in meta_data")
    
    og_start = meta_data[start_key]
    og_end = meta_data[end_key]
    block_duration = og_end - og_start
    
    # Calculate threshold in absolute time
    time_thresh = boundary_thresh * block_duration
    
    # Collect all valid decoupling times (not -1 or -2)
    valid_decorr_times = []
    
    for key, value in meta_data.items():
        if key.endswith('_decorr_times') and isinstance(value, list) and len(value) > index:
            decorr_time = value[index]
            if decorr_time != -1 and decorr_time != -2:
                valid_decorr_times.append(decorr_time)
    
    # If no valid decoupling times, return original boundaries
    if not valid_decorr_times:
        return (og_start, og_end)
    
    # Check if any valid times lie outside of the threshold boundaries
    start_thresh = og_start + time_thresh
    end_thresh = og_end - time_thresh
    
    for time in valid_decorr_times:
        if time >= start_thresh or time <= end_thresh:
            # If any time is outside thresholds, return original boundaries
            return (og_start, og_end)
    
    # Find valid times that are within threshold of start
    valid_start_times = [time for time in valid_decorr_times if abs(time - og_start) <= time_thresh]
    
    # Find valid times that are within threshold of end
    valid_end_times = [time for time in valid_decorr_times if abs(time - og_end) <= time_thresh]
    
    # Calculate new boundaries
    new_start = max([og_start] + valid_start_times) if valid_start_times else og_start
    new_end = min([og_end] + valid_end_times) if valid_end_times else og_end
    
    return (new_start, new_end)