import pandas as pd
import os

def get_manual_responsivity(csv_file, metadata, method='Onset'):
    """
    Process odor response data and categorize by blocks.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
    metadata : dict
        Dictionary containing block assignments from parse_expkeys
    method : str
        One of 'Onset', 'Offset', 'Both', 'Localizer', 'All'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: cell_ID, Block1_responsivity, Block2_resposivity
    """
    def get_block_assignments():
        odor_codes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        block1_odors = []
        block2_odors = []
        
        for code in odor_codes:
            odor = f'Odor {code}'
            if odor in metadata['block1_type']:
                block1_odors.append(code)
            else:
                block2_odors.append(code)
                
        return block1_odors, block2_odors
    
    def check_value(value):
        """Check if value is not sentinel value (-9999) and not NaN"""
        return pd.notna(value) and value != -9999
    
    def load_novel_block_data(csv_path):
        """Load and process novel block ratings if they exist"""
        novel_path = os.path.join(os.path.dirname(csv_path), 'novel_block_ratings.csv')
        if os.path.exists(novel_path):
            novel_df = pd.read_csv(novel_path)
            # Create a mapping dictionary for each cell ID and its G odor responsivity
            novel_map = {}
            for _, row in novel_df.iterrows():
                cell_suffix = row['cell_ID']  # This has format imecX_Y
                # Store both onset and offset information for each block
                novel_map[cell_suffix] = {
                    'block1_onset': row['Block 1 Onset'],
                    'block1_offset': row['Block 1 Offset'],
                    'block2_onset': row['Block 2 Onset'],
                    'block2_offset': row['Block 2 Offset']
                }
            return novel_map
        return None

    def get_responsive_odors_row(row, odor_list, method, novel_map=None):
        responsive = []
        for odor in odor_list:
            is_responsive = False
            
            # Special handling for Odor G if novel_map exists
            if odor == 'G' and novel_map is not None:
                # Extract the imecX_Y suffix from the cell_ID
                cell_id = row['cell_ID']
                cell_suffix = cell_id[cell_id.find('imec'):]
                
                if cell_suffix in novel_map:
                    novel_data = novel_map[cell_suffix]
                    
                    if method in ['Onset', 'Both', 'All']:
                        # Check block1 or block2 based on which odor_list we're processing
                        is_block1 = 'G' in [o for o in odor_list if f'Odor {o}' in metadata['block1_type']]
                        onset_key = 'block1_onset' if is_block1 else 'block2_onset'
                        
                        if method == 'Onset':
                            is_responsive = novel_data[onset_key] == 1
                        elif method == 'Both':
                            offset_key = 'block1_offset' if is_block1 else 'block2_offset'
                            is_responsive = novel_data[onset_key] == 1 and novel_data[offset_key] == 1
                        else:  # All
                            is_responsive = novel_data[onset_key] == 1
                            
                    if not is_responsive and method in ['Offset', 'All']:
                        is_block1 = 'G' in [o for o in odor_list if f'Odor {o}' in metadata['block1_type']]
                        offset_key = 'block1_offset' if is_block1 else 'block2_offset'
                        is_responsive = novel_data[offset_key] == 1
                        
            else:  # Original logic for other odors
                if method in ['Onset', 'Both', 'All']:
                    block_onset = f'Odor {odor} Block Onset'
                    if block_onset in row.index and check_value(row[block_onset]):
                        if method == 'Onset':
                            is_responsive = True
                        elif method == 'Both':
                            block_offset = f'Odor {odor} Block Offset'
                            is_responsive = check_value(row[block_offset])
                        else:  # All
                            is_responsive = True
                
                if not is_responsive and method in ['Offset', 'All']:
                    block_offset = f'Odor {odor} Block Offset'
                    if block_offset in row.index and check_value(row[block_offset]):
                        is_responsive = True
                
                if not is_responsive and method in ['Localizer', 'All']:
                    loc_onset = f'Odor {odor} Localizer Onset'
                    loc_offset = f'Odor {odor} Localizer Offset'
                    if (loc_onset in row.index and loc_offset in row.index and 
                        check_value(row[loc_onset]) and check_value(row[loc_offset])):
                        is_responsive = True
            
            if is_responsive:
                responsive.append(odor)
                
        return ','.join(responsive) if responsive else ''
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Check if we need to load novel block ratings
    has_odor_g = any(check_value(row[f'Odor G Block {col}']) 
                    for col in ['Onset', 'Offset'] 
                    for _, row in df.iterrows())
    
    novel_map = load_novel_block_data(csv_file) if has_odor_g else None
    
    # Get block assignments
    block1_odors, block2_odors = get_block_assignments()
    
    # Initialize output DataFrame
    result = pd.DataFrame()
    result['cell_ID'] = df['cell_ID']
    
    # Process each block
    result['Block1_responsivity'] = df.apply(
        lambda row: get_responsive_odors_row(row, block1_odors, method, novel_map), 
        axis=1
    )
    
    result['Block2_responsivity'] = df.apply(
        lambda row: get_responsive_odors_row(row, block2_odors, method, novel_map), 
        axis=1
    )
    
    return result