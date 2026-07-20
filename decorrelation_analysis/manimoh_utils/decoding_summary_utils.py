import numpy as np
import pandas as pd
import os
import pickle
from pathlib import Path

import pynapple as nap
from manimoh_utils import *

def summarize_standard_decoding(mice_list, dir_path, subset='all'):
    """
    Summarize decoding results from pickle files for specified mice.
    
    Parameters:
    -----------
    mice_list : list
        List of mouse IDs to process
    dir_path : str
        Base directory path where mouse data is stored
    subset : str
        Which subset of data to use: 'standard', 'stable', or 'unstable'
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing summarized decoding results
    """
    # Validate subset parameter
    if subset not in ['all', 'stable', 'unstable']:
        raise ValueError("subset must be one of 'all', 'stable', or 'unstable'")
    
    # Initialize the results list
    results = []
    
    # Get file suffixes based on subset parameter
    primary_suffix = ""
    if subset == 'stable':
        primary_suffix = "_stable_only"
    elif subset == 'unstable':
        primary_suffix = "_unstable_only"
    
    # Create a dictionary mapping mice to their session folders
    mouse_sess_dict = {}
    
    # Iterate through each mouse to build the dictionary
    for mouse in mice_list:
        mouse_dir = os.path.join(dir_path, mouse)
        if not os.path.exists(mouse_dir):
            continue
            
        sessions = []
        for session_folder in os.listdir(mouse_dir):
            session_path = os.path.join(mouse_dir, session_folder)
            if os.path.isdir(session_path):
                # Determine which files to check based on subset
                if subset == 'all':
                    # For 'all' subset, just check for standard files
                    block1_path = os.path.join(session_path, f'block1_standard_decoding.pkl')
                    block2_path = os.path.join(session_path, f'block2_standard_decoding.pkl')
                    
                    if os.path.exists(block1_path) or os.path.exists(block2_path):
                        sessions.append(session_path)
                else:
                    # For stable or unstable subsets, check for the specific files
                    block1_primary_path = os.path.join(session_path, f'block1_standard_decoding{primary_suffix}.pkl')
                    block2_primary_path = os.path.join(session_path, f'block2_standard_decoding{primary_suffix}.pkl')
                    
                    if os.path.exists(block1_primary_path) or os.path.exists(block2_primary_path):
                        sessions.append(session_path)
        
        if sessions:
            mouse_sess_dict[mouse] = sessions
    
    # Process each mouse and session
    for mouse, sessions in mouse_sess_dict.items():
        for session_path in sessions:
            session_name = os.path.basename(session_path)
            
            # Process files based on subset
            if subset == 'all':
                # For 'all' subset, just process standard files
                block1_path = os.path.join(session_path, 'block1_standard_decoding.pkl')
                if os.path.exists(block1_path):
                    process_standard_block_file(mouse, session_name, block1_path, True, results)
                
                block2_path = os.path.join(session_path, 'block2_standard_decoding.pkl')
                if os.path.exists(block2_path):
                    process_standard_block_file(mouse, session_name, block2_path, False, results)
            else:
                # For stable or unstable subsets
                # Process block1 files
                block1_primary_path = os.path.join(session_path, f'block1_standard_decoding{primary_suffix}.pkl')
                block1_standard_path = os.path.join(session_path, 'block1_standard_decoding.pkl')
                
                if os.path.exists(block1_primary_path):
                    with open(block1_primary_path, 'rb') as f:
                        block1_primary_data = pickle.load(f)
                    
                    # Check if primary results are empty
                    if not block1_primary_data.get('b1_decoding_results', []):
                        if os.path.exists(block1_standard_path):
                            print(f"Using standard decoding for mouse {mouse}, session {session_name}, block 1")
                            process_standard_block_file(mouse, session_name, block1_standard_path, True, results)
                    else:
                        process_standard_block_file(mouse, session_name, block1_primary_path, True, results)
                
                # Process block2 files
                block2_primary_path = os.path.join(session_path, f'block2_standard_decoding{primary_suffix}.pkl')
                block2_standard_path = os.path.join(session_path, 'block2_standard_decoding.pkl')
                
                if os.path.exists(block2_primary_path):
                    with open(block2_primary_path, 'rb') as f:
                        block2_primary_data = pickle.load(f)
                    
                    # Check if primary results are empty
                    if not block2_primary_data.get('b2_decoding_results', []):
                        if os.path.exists(block2_standard_path):
                            print(f"Using standard decoding for mouse {mouse}, session {session_name}, block 2")
                            process_standard_block_file(mouse, session_name, block2_standard_path, False, results)
                    else:
                        process_standard_block_file(mouse, session_name, block2_primary_path, False, results)
    
    # Create DataFrame from results
    if not results:
        return pd.DataFrame(columns=[
            'mouse', 'session', 'isBlock1', 'isBlockUE', 'session_type', 
            'neuron_count', 'decoding_accuracy_mean', 'decoding_accuracy_std',
            'decoding_f1_mean', 'decoding_f1_std'
        ])
    
    return pd.DataFrame(results)

def process_standard_block_file(mouse, session_name, file_path, is_block1, results):
    """
    Process a single block pickle file (standard_decoding) and extract the required information.
    
    Parameters:
    -----------
    mouse : str
        Mouse ID
    session_name : str
        Session folder name
    file_path : str
        Path to the pickle file
    is_block1 : bool
        True if processing block1, False if processing block2
    results : list
        List to append results to
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Set the prefix based on block number
        prefix = 'b1' if is_block1 else 'b2'
        
        # Extract required fields
        decoding_results = data.get(f'{prefix}_decoding_results', [])
        decoding_tbase = data.get(f'{prefix}_decoding_tbase', [])
        neuron_count = data.get(f'{prefix}_neuron_count', np.nan)
        sess_category = data.get('sess_category', '')
        block_category = data.get(f'{prefix}_category', '')
        
        # Check if results are empty
        if not decoding_results:
            row = {
                'mouse': mouse,
                'session': session_name,
                'isBlock1': is_block1,
                'isBlockUE': block_category == 'UE',
                'session_type': sess_category,
                'neuron_count': neuron_count,
                'decoding_accuracy_mean': np.nan,
                'decoding_accuracy_std': np.nan,
                'decoding_f1_mean': np.nan,
                'decoding_f1_std': np.nan
            }
            results.append(row)
        else:
            # Extract metrics using list comprehensions - one row per pkl file
            row = {
                'mouse': mouse,
                'session': session_name,
                'isBlock1': is_block1,
                'isBlockUE': block_category == 'UE',
                'session_type': sess_category,
                'neuron_count': neuron_count,
                'decoding_accuracy_mean': [decoding_results[key].get('mean_accuracy', np.nan) for key in decoding_tbase],
                'decoding_accuracy_std': [decoding_results[key].get('std_accuracy', np.nan) for key in decoding_tbase],
                'decoding_f1_mean': [decoding_results[key].get('mean_f1_score', np.nan) for key in decoding_tbase],
                'decoding_f1_std': [decoding_results[key].get('std_f1_score', np.nan) for key in decoding_tbase]
            }
            results.append(row)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        # Add a row with error information
        row = {
            'mouse': mouse,
            'session': session_name,
            'isBlock1': is_block1,
            'isBlockUE': np.nan,
            'session_type': 'ERROR',
            'neuron_count': np.nan,
            'decoding_accuracy_mean': np.nan,
            'decoding_accuracy_std': np.nan,
            'decoding_f1_mean': np.nan,
            'decoding_f1_std': np.nan
        }
        results.append(row)

def summarize_localizer_decoding(mice_list, dir_path, subset='stable'):
    """
    Summarize localizer (Block 3) decoding results from pickle files for specified mice.
    
    Parameters:
    -----------
    mice_list : list
        List of mouse IDs to process
    dir_path : str
        Base directory path where mouse data is stored
    subset : str
        Which subset of data to use: 'stable' or 'unstable' (default: 'stable')
        Note: 'all' is not applicable for localizer decoding
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing summarized localizer decoding results
    """
    # Validate subset parameter
    if subset not in ['stable', 'unstable']:
        raise ValueError("subset must be either 'stable' or 'unstable'")
    
    # Initialize the results list
    results = []
    
    # Get file suffix based on subset parameter
    primary_suffix = "_stable_only" if subset == 'stable' else "_unstable_only"
    
    # Create a dictionary mapping mice to their session folders
    mouse_sess_dict = {}
    
    # Iterate through each mouse to build the dictionary
    for mouse in mice_list:
        mouse_dir = os.path.join(dir_path, mouse)
        if not os.path.exists(mouse_dir):
            continue
            
        sessions = []
        for session_folder in os.listdir(mouse_dir):
            session_path = os.path.join(mouse_dir, session_folder)
            if os.path.isdir(session_path):
                # Check for Block 3 localizer files
                block3_path = os.path.join(session_path, f'block3_localizer_decoding{primary_suffix}.pkl')
                
                if os.path.exists(block3_path):
                    sessions.append(session_path)
        
        if sessions:
            mouse_sess_dict[mouse] = sessions
    
    # Process each mouse and session
    for mouse, sessions in mouse_sess_dict.items():
        for session_path in sessions:
            session_name = os.path.basename(session_path)
            
            # Process Block 3 file
            block3_path = os.path.join(session_path, f'block3_localizer_decoding{primary_suffix}.pkl')
            if os.path.exists(block3_path):
                process_localizer_block_file(mouse, session_name, block3_path, results)
    
    # Create DataFrame from results
    if not results:
        return pd.DataFrame(columns=[
            'mouse', 'session', 'session_type', 
            'neuron_count', 'decoding_accuracy_mean', 'decoding_accuracy_std',
            'decoding_f1_mean', 'decoding_f1_std'
        ])
    
    return pd.DataFrame(results)

def process_localizer_block_file(mouse, session_name, file_path, results):
    """
    Process a single Block 3 localizer pickle file and extract the required information.
    
    Parameters:
    -----------
    mouse : str
        Mouse ID
    session_name : str
        Session folder name
    file_path : str
        Path to the pickle file
    results : list
        List to append results to
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract required fields (Block 3 uses 'b3_' prefix)
        decoding_results = data.get('b3_decoding_results', [])
        decoding_tbase = data.get('b3_decoding_tbase', [])
        neuron_count = data.get('b3_neuron_count', np.nan)
        sess_category = data.get('sess_category', 'localizer')
        
        # Check if results are empty
        if not decoding_results:
            row = {
                'mouse': mouse,
                'session': session_name,
                'session_type': sess_category,
                'neuron_count': neuron_count,
                'decoding_accuracy_mean': np.nan,
                'decoding_accuracy_std': np.nan,
                'decoding_f1_mean': np.nan,
                'decoding_f1_std': np.nan
            }
            results.append(row)
        else:
            # Extract metrics using list comprehensions - one row per pkl file
            row = {
                'mouse': mouse,
                'session': session_name,
                'session_type': sess_category,
                'neuron_count': neuron_count,
                'decoding_accuracy_mean': [decoding_results[key].get('mean_accuracy', np.nan) for key in decoding_tbase],
                'decoding_accuracy_std': [decoding_results[key].get('std_accuracy', np.nan) for key in decoding_tbase],
                'decoding_f1_mean': [decoding_results[key].get('mean_f1_score', np.nan) for key in decoding_tbase],
                'decoding_f1_std': [decoding_results[key].get('std_f1_score', np.nan) for key in decoding_tbase]
            }
            results.append(row)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        # Add a row with error information
        row = {
            'mouse': mouse,
            'session': session_name,
            'session_type': 'ERROR',
            'neuron_count': np.nan,
            'decoding_accuracy_mean': np.nan,
            'decoding_accuracy_std': np.nan,
            'decoding_f1_mean': np.nan,
            'decoding_f1_std': np.nan
        }
        results.append(row)
        
def summarize_target_decoding(mice_list, dir_path, subset='all', target='G'):
    """
    Summarize target decoding results from pickle files for specified mice.
    
    Parameters:
    -----------
    mice_list : list
        List of mouse IDs to process
    dir_path : str
        Base directory path where mouse data is stored
    subset : str
        Which subset of data to use: 'all', 'stable', or 'unstable'
    target : str
        Target odor to extract (default 'G'). Valid values are letters A-G.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing summarized decoding results
    """
    # Validate subset parameter
    if subset not in ['all', 'stable', 'unstable']:
        raise ValueError("subset must be one of 'all', 'stable', or 'unstable'")
        
    # Initialize the results list
    results = []
    
    # Get file suffixes based on subset parameter
    primary_suffix = ""
    if subset == 'stable':
        primary_suffix = "_stable_only"
    elif subset == 'unstable':
        primary_suffix = "_unstable_only"
    
    # Create a dictionary mapping mice to their session folders
    mouse_sess_dict = {}
    
    # Iterate through each mouse to build the dictionary
    for mouse in mice_list:
        mouse_dir = os.path.join(dir_path, mouse)
        if not os.path.exists(mouse_dir):
            continue
            
        sessions = []
        for session_folder in os.listdir(mouse_dir):
            session_path = os.path.join(mouse_dir, session_folder)
            if os.path.isdir(session_path):
                # Determine which files to check based on subset
                if subset == 'all':
                    # For 'all' subset, just check for target files
                    block1_path = os.path.join(session_path, f'block1_target_decoding.pkl')
                    block2_path = os.path.join(session_path, f'block2_target_decoding.pkl')
                    
                    if os.path.exists(block1_path) or os.path.exists(block2_path):
                        sessions.append(session_path)
                else:
                    # For stable or unstable subsets, check for the specific files
                    block1_primary_path = os.path.join(session_path, f'block1_target_decoding{primary_suffix}.pkl')
                    block2_primary_path = os.path.join(session_path, f'block2_target_decoding{primary_suffix}.pkl')
                    
                    if os.path.exists(block1_primary_path) or os.path.exists(block2_primary_path):
                        sessions.append(session_path)
        
        if sessions:
            mouse_sess_dict[mouse] = sessions
    
    # Process each mouse and session
    for mouse, sessions in mouse_sess_dict.items():
        for session_path in sessions:
            session_name = os.path.basename(session_path)
            
            # Process files based on subset
            if subset == 'all':
                # For 'all' subset, process target files
                block1_path = os.path.join(session_path, 'block1_target_decoding.pkl')
                if os.path.exists(block1_path):
                    process_target_block_file(mouse, session_name, block1_path, True, results, target)
                
                block2_path = os.path.join(session_path, 'block2_target_decoding.pkl')
                if os.path.exists(block2_path):
                    process_target_block_file(mouse, session_name, block2_path, False, results, target)
            else:
                # For stable or unstable subsets
                # Process block1 files
                block1_primary_path = os.path.join(session_path, f'block1_target_decoding{primary_suffix}.pkl')
                block1_target_path = os.path.join(session_path, 'block1_target_decoding.pkl')
                
                if os.path.exists(block1_primary_path):
                    with open(block1_primary_path, 'rb') as f:
                        block1_primary_data = pickle.load(f)
                    
                    # Check if primary results are empty
                    if not block1_primary_data.get('b1_decoding_results', []):
                        if os.path.exists(block1_target_path):
                            print(f"Using target decoding for mouse {mouse}, session {session_name}, block 1")
                            process_target_block_file(mouse, session_name, block1_target_path, True, results, target)
                    else:
                        process_target_block_file(mouse, session_name, block1_primary_path, True, results, target)
                
                # Process block2 files
                block2_primary_path = os.path.join(session_path, f'block2_target_decoding{primary_suffix}.pkl')
                block2_target_path = os.path.join(session_path, 'block2_target_decoding.pkl')
                
                if os.path.exists(block2_primary_path):
                    with open(block2_primary_path, 'rb') as f:
                        block2_primary_data = pickle.load(f)
                    
                    # Check if primary results are empty
                    if not block2_primary_data.get('b2_decoding_results', []):
                        if os.path.exists(block2_target_path):
                            print(f"Using target decoding for mouse {mouse}, session {session_name}, block 2")
                            process_target_block_file(mouse, session_name, block2_target_path, False, results, target)
                    else:
                        process_target_block_file(mouse, session_name, block2_primary_path, False, results, target)
    
    # Create DataFrame from results
    if not results:
        return pd.DataFrame(columns=[
            'mouse', 'session', 'isBlock1', 'isBlockUE', 'session_type', 
            'neuron_count', 'decoding_accuracy_mean', 'decoding_accuracy_std',
            'decoding_f1_mean', 'decoding_f1_std', 'odor_type'
        ])
    
    return pd.DataFrame(results)

def process_target_block_file(mouse, session_name, file_path, is_block1, results, target):
    """
    Process a single block pickle file and extract the required information.
    
    Parameters:
    -----------
    mouse : str
        Mouse ID
    session_name : str
        Session folder name
    file_path : str
        Path to the pickle file
    is_block1 : bool
        True if processing block1, False if processing block2
    results : list
        List to append results to
    target : str
        Target odor to extract (a letter from A to G)
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Set the prefix based on block number
        prefix = 'b1' if is_block1 else 'b2'
        
        # Extract required fields
        decoding_results = data.get(f'{prefix}_decoding_results', [])
        decoding_tbase = data.get(f'{prefix}_decoding_tbase', [])
        neuron_count = data.get(f'{prefix}_neuron_count', np.nan)
        sess_category = data.get('sess_category', '')
        block_category = data.get(f'{prefix}_category', '')
        
        # Check if results are empty
        if not decoding_results:
            # Add a row for target
            row_target = {
                'mouse': mouse,
                'session': session_name,
                'isBlock1': is_block1,
                'isBlockUE': block_category == 'UE',
                'session_type': sess_category,
                'neuron_count': neuron_count,
                'decoding_accuracy_mean': np.nan,
                'decoding_accuracy_std': np.nan,
                'decoding_f1_mean': np.nan,
                'decoding_f1_std': np.nan,
                'odor_type': target
            }
            results.append(row_target)
            
            # Add a row for others
            row_others = row_target.copy()
            row_others['odor_type'] = 'others'
            results.append(row_others)
        else:
            # Get all available odors (keys) in the results
            available_odors = set()
            for tbase in decoding_tbase:
                if tbase in decoding_results:
                    available_odors.update(decoding_results[tbase].keys())
            
            # Ensure the target is available
            if target in available_odors:
                # Extract target metrics
                target_accuracy_mean = []
                target_accuracy_std = []
                target_f1_mean = []
                target_f1_std = []
                
                for tbase in decoding_tbase:
                    if tbase in decoding_results and target in decoding_results[tbase]:
                        target_accuracy_mean.append(decoding_results[tbase][target].get('mean_balanced_accuracy', np.nan))
                        target_accuracy_std.append(decoding_results[tbase][target].get('std_balanced_accuracy', np.nan))
                        target_f1_mean.append(decoding_results[tbase][target].get('mean_f1_score', np.nan))
                        target_f1_std.append(decoding_results[tbase][target].get('std_f1_score', np.nan))
                
                # Add target row
                row_target = {
                    'mouse': mouse,
                    'session': session_name,
                    'isBlock1': is_block1,
                    'isBlockUE': block_category == 'UE',
                    'session_type': sess_category,
                    'neuron_count': neuron_count,
                    'decoding_accuracy_mean': target_accuracy_mean,
                    'decoding_accuracy_std': target_accuracy_std,
                    'decoding_f1_mean': target_f1_mean,
                    'decoding_f1_std': target_f1_std,
                    'odor_type': target
                }
                results.append(row_target)
                
                # Extract metrics for other odors
                other_odors = [odor for odor in available_odors if odor != target]
                if other_odors:
                    # Initialize arrays for other odors' metrics
                    others_accuracy_means = []
                    others_accuracy_stds = []
                    others_f1_means = []
                    others_f1_stds = []
                    
                    # For each timebase
                    for tbase in decoding_tbase:
                        if tbase in decoding_results:
                            # For each timebase, collect metrics for all other odors
                            tbase_acc_means = []
                            tbase_acc_stds = []
                            tbase_f1_means = []
                            tbase_f1_stds = []
                            
                            for odor in other_odors:
                                if odor in decoding_results[tbase]:
                                    tbase_acc_means.append(decoding_results[tbase][odor].get('mean_balanced_accuracy', np.nan))
                                    tbase_acc_stds.append(decoding_results[tbase][odor].get('std_balanced_accuracy', np.nan))
                                    tbase_f1_means.append(decoding_results[tbase][odor].get('mean_f1_score', np.nan))
                                    tbase_f1_stds.append(decoding_results[tbase][odor].get('std_f1_score', np.nan))
                            
                            # Average the metrics for this timebase
                            others_accuracy_means.append(np.nanmean(tbase_acc_means) if tbase_acc_means else np.nan)
                            others_accuracy_stds.append(np.nanmean(tbase_acc_stds) if tbase_acc_stds else np.nan)
                            others_f1_means.append(np.nanmean(tbase_f1_means) if tbase_f1_means else np.nan)
                            others_f1_stds.append(np.nanmean(tbase_f1_stds) if tbase_f1_stds else np.nan)
                    
                    # Add others row
                    row_others = {
                        'mouse': mouse,
                        'session': session_name,
                        'isBlock1': is_block1,
                        'isBlockUE': block_category == 'UE',
                        'session_type': sess_category,
                        'neuron_count': neuron_count,
                        'decoding_accuracy_mean': others_accuracy_means,
                        'decoding_accuracy_std': others_accuracy_stds,
                        'decoding_f1_mean': others_f1_means,
                        'decoding_f1_std': others_f1_stds,
                        'odor_type': 'others'
                    }
                    results.append(row_others)
            else:
                # Target not available - add empty rows
                row_target = {
                    'mouse': mouse,
                    'session': session_name,
                    'isBlock1': is_block1,
                    'isBlockUE': block_category == 'UE',
                    'session_type': sess_category,
                    'neuron_count': neuron_count,
                    'decoding_accuracy_mean': np.nan,
                    'decoding_accuracy_std': np.nan,
                    'decoding_f1_mean': np.nan,
                    'decoding_f1_std': np.nan,
                    'odor_type': target
                }
                results.append(row_target)
                
                row_others = row_target.copy()
                row_others['odor_type'] = 'others'
                results.append(row_others)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        # Add error rows for both target and others
        row_target = {
            'mouse': mouse,
            'session': session_name,
            'isBlock1': is_block1,
            'isBlockUE': np.nan,
            'session_type': 'ERROR',
            'neuron_count': np.nan,
            'decoding_accuracy_mean': np.nan,
            'decoding_accuracy_std': np.nan,
            'decoding_f1_mean': np.nan,
            'decoding_f1_std': np.nan,
            'odor_type': target
        }
        results.append(row_target)
        
        row_others = row_target.copy()
        row_others['odor_type'] = 'others'
        results.append(row_others)
        
def summarize_subsampled_decoding(mice_list, dir_path):
    """
    Summarize subsampled decoding results from pickle files for specified mice.
    
    Parameters:
    -----------
    mice_list : list
        List of mouse IDs to process
    dir_path : str
        Base directory path where mouse data is stored
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing summarized decoding results for various subsampling levels
    """
    # Initialize the results list
    results = []
    
    # Create a dictionary mapping mice to their session folders
    mouse_sess_dict = {}
    
    # Iterate through each mouse to build the dictionary
    for mouse in mice_list:
        mouse_dir = os.path.join(dir_path, mouse)
        if not os.path.exists(mouse_dir):
            continue
            
        sessions = []
        for session_folder in os.listdir(mouse_dir):
            session_path = os.path.join(mouse_dir, session_folder)
            if os.path.isdir(session_path):
                block1_path = os.path.join(session_path, 'subsampled_block1_standard_decoding.pkl')
                block2_path = os.path.join(session_path, 'subsampled_block2_standard_decoding.pkl')
                
                if os.path.exists(block1_path) or os.path.exists(block2_path):
                    sessions.append(session_path)
        
        if sessions:
            mouse_sess_dict[mouse] = sessions
    
    # Process each mouse and session
    for mouse, sessions in mouse_sess_dict.items():
        for session_path in sessions:
            session_name = os.path.basename(session_path)
            
            # Process block1 file
            block1_path = os.path.join(session_path, 'subsampled_block1_standard_decoding.pkl')
            if os.path.exists(block1_path):
                process_subsampled_block_file(mouse, session_name, block1_path, True, results)
            
            # Process block2 file
            block2_path = os.path.join(session_path, 'subsampled_block2_standard_decoding.pkl')
            if os.path.exists(block2_path):
                process_subsampled_block_file(mouse, session_name, block2_path, False, results)
    
    # Create DataFrame from results
    if not results:
        # Define columns for empty dataframe
        columns = [
            'mouse', 'session', 'isBlock1', 'isBlockUE', 'session_type', 
            'neuron_count', 'decoding_tbase', 'subsample_counts', 'num_rounds'
        ]
        # Add columns for each metric and subsample level
        subsample_levels = ['sub_25', 'sub_50', 'sub_75', 'sub_100', 'all_neurons']
        metrics = ['balanced_accuracy', 'accuracy', 'f1_score']
        for level in subsample_levels:
            for metric in metrics:
                columns.append(f'{level}_{metric}')
                columns.append(f'{level}_{metric}_std')
        
        return pd.DataFrame(columns=columns)
    
    return pd.DataFrame(results)

def process_subsampled_block_file(mouse, session_name, file_path, is_block1, results):
    """
    Process a single block pickle file (subsampled_standard_decoding) and extract the required information.
    
    Parameters:
    -----------
    mouse : str
        Mouse ID
    session_name : str
        Session folder name
    file_path : str
        Path to the pickle file
    is_block1 : bool
        True if processing block1, False if processing block2
    results : list
        List to append results to
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Set the prefix based on block number
        prefix = 'b1' if is_block1 else 'b2'
        
        # Extract required fields
        decoding_results = data.get(f'{prefix}_decoding_results', {})
        decoding_tbase = data.get(f'{prefix}_decoding_tbase', [])
        neuron_count = data.get(f'{prefix}_neuron_count', np.nan)
        sess_category = data.get('sess_category', '')
        block_category = data.get(f'{prefix}_category', '')
        subsample_counts = data.get('subsample_counts', [25, 50, 75, 100])
        num_rounds = data.get('num_rounds', 50)
        
        # Check if results are empty
        if not decoding_results:
            row = {
                'mouse': mouse,
                'session': session_name,
                'isBlock1': is_block1,
                'isBlockUE': block_category == 'UE',
                'session_type': sess_category,
                'neuron_count': neuron_count,
                'decoding_tbase': decoding_tbase,
                'subsample_counts': subsample_counts,
                'num_rounds': num_rounds
            }
            
            # Add empty arrays for all metrics and subsample levels
            for level in ['sub_25', 'sub_50', 'sub_75', 'sub_100', 'all_neurons']:
                for metric in ['balanced_accuracy', 'accuracy', 'f1_score']:
                    row[f'{level}_{metric}'] = []
                    row[f'{level}_{metric}_std'] = []
            
            results.append(row)
        else:
            # Base row with common info
            row = {
                'mouse': mouse,
                'session': session_name,
                'isBlock1': is_block1,
                'isBlockUE': block_category == 'UE',
                'session_type': sess_category,
                'neuron_count': neuron_count,
                'decoding_tbase': decoding_tbase,
                'subsample_counts': subsample_counts,
                'num_rounds': num_rounds
            }
            
            # Initialize arrays for each metric and subsample level
            for level in [f'sub_{count}' for count in subsample_counts] + ['all_neurons']:
                for metric in ['balanced_accuracy', 'accuracy', 'f1_score']:
                    row[f'{level}_{metric}'] = []
                    row[f'{level}_{metric}_std'] = []
            
            # For each time bin, extract metrics for each subsample level
            for t in decoding_tbase:
                if t in decoding_results:
                    time_bin_results = decoding_results[t]
                    
                    # For each subsample level
                    for level in [f'sub_{count}' for count in subsample_counts] + ['all_neurons']:
                        if level in time_bin_results:
                            level_results = time_bin_results[level]
                            
                            # For each metric
                            for metric in ['balanced_accuracy', 'accuracy', 'f1_score']:
                                if metric in level_results:
                                    row[f'{level}_{metric}'].append(level_results[metric])
                                else:
                                    row[f'{level}_{metric}'].append(np.nan)
                                
                                # Standard deviation (if available)
                                std_key = f'{metric}_std'
                                if std_key in level_results:
                                    row[f'{level}_{metric}_std'].append(level_results[std_key])
                                else:
                                    row[f'{level}_{metric}_std'].append(np.nan)
                        else:
                            # Level not found for this time bin
                            for metric in ['balanced_accuracy', 'accuracy', 'f1_score']:
                                row[f'{level}_{metric}'].append(np.nan)
                                row[f'{level}_{metric}_std'].append(np.nan)
            
            results.append(row)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        # Add a row with error information
        row = {
            'mouse': mouse,
            'session': session_name,
            'isBlock1': is_block1,
            'isBlockUE': np.nan,
            'session_type': 'ERROR',
            'neuron_count': np.nan,
            'decoding_tbase': [],
            'subsample_counts': [],
            'num_rounds': np.nan
        }
        
        # Add empty arrays for all metrics and subsample levels
        for level in ['sub_25', 'sub_50', 'sub_75', 'sub_100', 'all_neurons']:
            for metric in ['balanced_accuracy', 'accuracy', 'f1_score']:
                row[f'{level}_{metric}'] = []
                row[f'{level}_{metric}_std'] = []
        
        results.append(row)

def summarize_segmentwise_decoding(mice_list, dir_path, num_segments=2, subset='stable'):
    """
    Summarize segmentwise decoding results from pickle files for specified mice.
    
    Parameters:
    -----------
    mice_list : list
        List of mouse IDs to process
    dir_path : str
        Base directory path where mouse data is stored
    num_segments : int
        Number of segments used in the decoding (default=2)
    subset : str
        Which subset of data to use: 'stable' or 'unstable'
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing summarized segmentwise decoding results
    """
    # Validate subset parameter
    if subset not in ['stable', 'unstable']:
        raise ValueError("subset must be either 'stable' or 'unstable'")
    
    # Initialize the results list
    results = []
    
    # Determine file suffix based on subset
    subset_suffix = f"_{subset}_only"
    
    # Create a dictionary mapping mice to their session folders
    mouse_sess_dict = {}
    
    # Iterate through each mouse to build the dictionary
    for mouse in mice_list:
        mouse_dir = os.path.join(dir_path, mouse)
        if not os.path.exists(mouse_dir):
            continue
            
        sessions = []
        for session_folder in os.listdir(mouse_dir):
            session_path = os.path.join(mouse_dir, session_folder)
            if os.path.isdir(session_path):
                # Check for the specific segmentwise files
                block1_path = os.path.join(session_path, f'block1_decoding{subset_suffix}_{num_segments}_segments.pkl')
                block2_path = os.path.join(session_path, f'block2_decoding{subset_suffix}_{num_segments}_segments.pkl')
                
                if os.path.exists(block1_path) or os.path.exists(block2_path):
                    sessions.append(session_path)
        
        if sessions:
            mouse_sess_dict[mouse] = sessions
    
    # Process each mouse and session
    for mouse, sessions in mouse_sess_dict.items():
        for session_path in sessions:
            session_name = os.path.basename(session_path)
            
            # Process block1 file
            block1_path = os.path.join(session_path, f'block1_decoding{subset_suffix}_{num_segments}_segments.pkl')
            if os.path.exists(block1_path):
                process_segmentwise_block_file(mouse, session_name, block1_path, True, results)
            
            # Process block2 file
            block2_path = os.path.join(session_path, f'block2_decoding{subset_suffix}_{num_segments}_segments.pkl')
            if os.path.exists(block2_path):
                process_segmentwise_block_file(mouse, session_name, block2_path, False, results)
    
    # Create DataFrame from results
    if not results:
        return pd.DataFrame(columns=[
            'mouse', 'session', 'isBlock1', 'isBlockUE', 'session_type', 
            'segment_index', 'num_segments', 'neuron_count', 
            'decoding_accuracy_mean', 'decoding_accuracy_std', 
            'decoding_f1_mean', 'decoding_f1_std', 'normalized_conf_mats', 'tbase'
        ])
    
    return pd.DataFrame(results)

def process_segmentwise_block_file(mouse, session_name, file_path, is_block1, results):
    """
    Process a single block pickle file from segmentwise decoding and extract the required information.
    
    Parameters:
    -----------
    mouse : str
        Mouse ID
    session_name : str
        Session folder name
    file_path : str
        Path to the pickle file
    is_block1 : bool
        True if processing block1, False if processing block2
    results : list
        List to append results to
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Set the prefix based on block number
        prefix = 'b1' if is_block1 else 'b2'
        
        # Extract common data
        sess_category = data.get('sess_category', '')
        block_category = data.get(f'{prefix}_category', '')
        neuron_count = data.get(f'{prefix}_neuron_count', np.nan)
        num_segments = data.get('num_segments', 0)
        
        # Get the segmentwise results
        segment_results = data.get(f'{prefix}_decoding_results', [])
        
        # If no segments or empty results, add a single row with NaN values
        if not segment_results:
            row = {
                'mouse': mouse,
                'session': session_name,
                'isBlock1': is_block1,
                'isBlockUE': block_category == 'UE',
                'session_type': sess_category,
                'segment_index': np.nan,
                'num_segments': num_segments,
                'neuron_count': neuron_count,
                'decoding_accuracy_mean': np.nan,
                'decoding_accuracy_std': np.nan,
                'decoding_f1_mean': np.nan,
                'decoding_f1_std': np.nan,
                'normalized_conf_mats': None,
                'tbase': None
            }
            results.append(row)
        else:
            # Process each segment
            for segment_data in segment_results:
                segment_index = segment_data.get('segment_index', np.nan)
                decoding_results = segment_data.get('decoding_results', {})
                decoding_tbase = segment_data.get('decoding_tbase', [])
                
                # Check if decoding results are empty
                if not decoding_results or all(v is None for v in decoding_results.values()):
                    # Add a row with NaN values for this segment
                    row = {
                        'mouse': mouse,
                        'session': session_name,
                        'isBlock1': is_block1,
                        'isBlockUE': block_category == 'UE',
                        'session_type': sess_category,
                        'segment_index': segment_index,
                        'num_segments': num_segments,
                        'neuron_count': neuron_count,
                        'decoding_accuracy_mean': np.nan,
                        'decoding_accuracy_std': np.nan,
                        'decoding_f1_mean': np.nan,
                        'decoding_f1_std': np.nan,
                        'normalized_conf_mats': None,
                        'tbase': None
                    }
                    results.append(row)
                else:
                    # Extract metrics for this segment
                    accuracy_means = []
                    accuracy_stds = []
                    f1_means = []
                    f1_stds = []
                    norm_conf_mats_by_time = {}
                    
                    # Collect metrics for each time point
                    for tpoint in decoding_tbase:
                        if tpoint in decoding_results and decoding_results[tpoint] is not None:
                            accuracy_means.append(decoding_results[tpoint].get('mean_accuracy', np.nan))
                            accuracy_stds.append(decoding_results[tpoint].get('std_accuracy', np.nan))
                            f1_means.append(decoding_results[tpoint].get('mean_f1_score', np.nan))
                            f1_stds.append(decoding_results[tpoint].get('std_f1_score', np.nan))
                            
                            # Extract confusion matrices if available
                            if 'confusion_matrices' in decoding_results[tpoint]:
                                conf_mats = decoding_results[tpoint].get('confusion_matrices', [])
                                if conf_mats:
                                    avg_conf_mat = np.mean(conf_mats, axis=0)
                                    # Avoid division by zero when normalizing
                                    row_sums = np.sum(avg_conf_mat, axis=1)
                                    # Only normalize rows with non-zero sums
                                    normalized_conf_mat = np.zeros_like(avg_conf_mat, dtype=float)
                                    for i, row_sum in enumerate(row_sums):
                                        if row_sum > 0:
                                            normalized_conf_mat[i, :] = avg_conf_mat[i, :] / row_sum
                                    
                                    norm_conf_mats_by_time[tpoint] = normalized_conf_mat
                    
                    # Add a row for this segment
                    row = {
                        'mouse': mouse,
                        'session': session_name,
                        'isBlock1': is_block1,
                        'isBlockUE': block_category == 'UE',
                        'session_type': sess_category,
                        'segment_index': segment_index,
                        'num_segments': num_segments,
                        'neuron_count': neuron_count,
                        'decoding_accuracy_mean': accuracy_means if accuracy_means else np.nan,
                        'decoding_accuracy_std': accuracy_stds if accuracy_stds else np.nan,
                        'decoding_f1_mean': f1_means if f1_means else np.nan,
                        'decoding_f1_std': f1_stds if f1_stds else np.nan,
                        'normalized_conf_mats': norm_conf_mats_by_time,
                        'tbase': decoding_tbase
                    }
                    results.append(row)
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        # Add a row with error information
        row = {
            'mouse': mouse,
            'session': session_name,
            'isBlock1': is_block1,
            'isBlockUE': np.nan,
            'session_type': 'ERROR',
            'segment_index': np.nan,
            'num_segments': np.nan,
            'neuron_count': np.nan,
            'decoding_accuracy_mean': np.nan,
            'decoding_accuracy_std': np.nan,
            'decoding_f1_mean': np.nan,
            'decoding_f1_std': np.nan,
            'normalized_conf_mats': None,
            'tbase': None
        }
        results.append(row)
        
def summarize_cross_segment_decoding(mice_list, dir_path, num_segments=2, subset='stable'):
    """
    Summarize cross-segment decoding results from pickle files for specified mice.
    
    Parameters:
    -----------
    mice_list : list
        List of mouse IDs to process
    dir_path : str
        Base directory path where mouse data is stored
    num_segments : int
        Number of segments used in the decoding (default=2)
    subset : str
        Which subset of data to use: 'stable' or 'unstable'
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing summarized cross-segment decoding results
    """
    # Validate subset parameter
    if subset not in ['stable', 'unstable']:
        raise ValueError("subset must be either 'stable' or 'unstable'")
    
    # Initialize the results list
    results = []
    
    # Determine file suffix based on subset
    subset_suffix = f"_{subset}_only" if subset == 'stable' else '_unstable_only'
    
    # Create a dictionary mapping mice to their session folders
    mouse_sess_dict = {}
    
    # Iterate through each mouse to build the dictionary
    for mouse in mice_list:
        mouse_dir = os.path.join(dir_path, mouse)
        if not os.path.exists(mouse_dir):
            continue
            
        sessions = []
        for session_folder in os.listdir(mouse_dir):
            session_path = os.path.join(mouse_dir, session_folder)
            if os.path.isdir(session_path):
                # Check for the specific cross-segment files
                block1_path = os.path.join(session_path, f'block1_cross_test_segment_decoder_{num_segments}seg.pkl')
                block2_path = os.path.join(session_path, f'block2_cross_test_segment_decoder_{num_segments}seg.pkl')
                
                if os.path.exists(block1_path) or os.path.exists(block2_path):
                    sessions.append(session_path)
        
        if sessions:
            mouse_sess_dict[mouse] = sessions
    
    # Process each mouse and session
    for mouse, sessions in mouse_sess_dict.items():
        for session_path in sessions:
            session_name = os.path.basename(session_path)
            
            # Process block1 file
            block1_path = os.path.join(session_path, f'block1_cross_test_segment_decoder_{num_segments}seg.pkl')
            if os.path.exists(block1_path):
                process_cross_segment_block_file(mouse, session_name, block1_path, True, results, num_segments)
            
            # Process block2 file
            block2_path = os.path.join(session_path, f'block2_cross_test_segment_decoder_{num_segments}seg.pkl')
            if os.path.exists(block2_path):
                process_cross_segment_block_file(mouse, session_name, block2_path, False, results, num_segments)
    
    # Create DataFrame from results
    if not results:
        return pd.DataFrame(columns=[
            'mouse', 'session', 'isBlock1', 'isBlockUE', 'session_type',
            'train_segment', 'test_segment', 'is_within_segment',
            'num_segments', 'neuron_count', 'tbase',
            'accuracy_tvec', 'accuracy_std_tvec', 'balanced_accuracy_tvec', 
            'balanced_accuracy_std_tvec', 'f1_macro_tvec', 'f1_std_tvec'
        ])
    
    return pd.DataFrame(results)

def process_cross_segment_block_file(mouse, session_name, pickle_path, is_block1, results_list, num_segments):
    """
    Process a single cross-segment decoding pickle file and add results to the list.
    
    Parameters:
    -----------
    mouse : str
        Mouse ID
    session_name : str
        Session name
    pickle_path : str
        Path to the pickle file
    is_block1 : bool
        Whether this is block1 (True) or block2 (False)
    results_list : list
        List to append results to
    num_segments : int
        Number of segments
    """
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading {pickle_path}: {e}")
        return
    
    # Extract metadata
    sess_category = data.get('sess_category', 'unknown')
    block_category = data.get('b1_category' if is_block1 else 'b2_category', 'unknown')
    is_block_ue = block_category == 'UE'
    
    # Extract cross-test results
    cross_test_results = data.get('cross_test_results', {})
    segment_results = cross_test_results.get('segment_results', {})
    neuron_count = cross_test_results.get('neuron_count', 0)
    tbase = cross_test_results.get('tbase', [])
    
    # Process each train-test segment combination
    for train_seg in range(num_segments):
        for test_seg in range(num_segments):
            key = f'train_seg{train_seg}_test_seg{test_seg}'
            
            # Initialize lists to store metrics across time
            accuracy_tvec = []
            accuracy_std_tvec = []
            balanced_accuracy_tvec = []
            balanced_accuracy_std_tvec = []
            f1_macro_tvec = []
            f1_std_tvec = []
            
            # Collect metrics for each time point
            for time_bin in tbase:
                if time_bin in segment_results and key in segment_results[time_bin]:
                    metrics = segment_results[time_bin][key]
                    accuracy_tvec.append(metrics.get('accuracy', np.nan))
                    accuracy_std_tvec.append(metrics.get('accuracy_std', np.nan))
                    balanced_accuracy_tvec.append(metrics.get('balanced_accuracy', np.nan))
                    balanced_accuracy_std_tvec.append(metrics.get('balanced_accuracy_std', np.nan))
                    f1_macro_tvec.append(metrics.get('f1_macro', np.nan))
                    f1_std_tvec.append(metrics.get('f1_std', np.nan))
                else:
                    # If no data for this time point, append NaN
                    accuracy_tvec.append(np.nan)
                    accuracy_std_tvec.append(np.nan)
                    balanced_accuracy_tvec.append(np.nan)
                    balanced_accuracy_std_tvec.append(np.nan)
                    f1_macro_tvec.append(np.nan)
                    f1_std_tvec.append(np.nan)
            
            # Create entry for this train-test combination
            result_entry = {
                'mouse': mouse,
                'session': session_name,
                'isBlock1': is_block1,
                'isBlockUE': is_block_ue,
                'session_type': sess_category,
                'train_segment': train_seg,
                'test_segment': test_seg,
                'is_within_segment': train_seg == test_seg,
                'num_segments': num_segments,
                'neuron_count': neuron_count,
                'tbase': tbase,
                'accuracy_tvec': accuracy_tvec,
                'accuracy_std_tvec': accuracy_std_tvec,
                'balanced_accuracy_tvec': balanced_accuracy_tvec,
                'balanced_accuracy_std_tvec': balanced_accuracy_std_tvec,
                'f1_macro_tvec': f1_macro_tvec,
                'f1_std_tvec': f1_std_tvec
            }
            
            results_list.append(result_entry)