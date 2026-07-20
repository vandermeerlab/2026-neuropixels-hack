import pandas as pd

def get_pystat_responsivity(csv_file, stat_type, threshold, metadata):
    """
    Process odor responsivity data and categorize by blocks.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
    stat_type : str
        Statistical test to use ('WSR' or 'MWU')
    threshold : float
        Significance threshold
    metadata : dict
        Dictionary containing block assignments from parse_expkeys
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: global_unit_id, Block1_responsivity, Block2_responsivity
    """
    
    def get_block_assignments(metadata):
        odor_codes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        block1_odors = []
        block2_odors = []
        
        for code in odor_codes:
            odor = f'Odor {code}'
            if odor in metadata['block1_type']:
                block1_odors.append(code)
            if odor in metadata['block2_type']:
                block2_odors.append(code)
                
        return block1_odors, block2_odors
    
    # Get block assignments from metadata
    block1_odors, block2_odors = get_block_assignments(metadata)
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Initialize output DataFrame
    result = pd.DataFrame()
    result['global_unit_id'] = df['global_unit_id']
    
    # Function to get responsive odors for a given row
    def get_responsive_odors_row(row, odor_list, stat_type, block=None):
        responsive = []
        for odor in odor_list:
            # Try both possible column name formats
            possible_col_names = [
                f'Odor {odor} {stat_type}',  # Original format
                f'Odor {odor} {stat_type} Block {block}' if block else None  # New format
            ]
            
            # Remove None from possible_col_names
            possible_col_names = [col for col in possible_col_names if col is not None]
            
            # Check if any of the possible column names exist and meet the threshold
            for col_name in possible_col_names:
                if col_name in row.index and row[col_name] < threshold:
                    responsive.append(odor)
                    break  # Break once we find a responsive column
                    
        return ','.join(responsive) if responsive else ''
    
    # Process each block with the appropriate block number
    result['Block1_responsivity'] = df.apply(
        lambda row: get_responsive_odors_row(row, block1_odors, stat_type, block=1), 
        axis=1
    )
    
    result['Block2_responsivity'] = df.apply(
        lambda row: get_responsive_odors_row(row, block2_odors, stat_type, block=2), 
        axis=1
    )
    
    return result