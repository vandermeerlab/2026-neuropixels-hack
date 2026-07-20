import os
import warnings
from pathlib import Path
import pandas as pd

def compare_responsivity_methods(df1, df2, method_types, save_path='.', prefix=None):
    """
    Compare responsivity results from two methods and save to CSV files.
    
    Parameters:
    -----------
    df1 : pd.DataFrame
        First method's DataFrame
    df2 : pd.DataFrame
        Second method's DataFrame
    method_types : tuple
        ('manual' or 'auto', 'manual' or 'auto') indicating types of df1 and df2
    save_path : str
        Path where comparison folder will be created
    prefix : str, optional
        Prefix for output folder name. If None, defaults to method_types combination
    
    Returns:
    --------
    tuple
        (results_dict, folder_path_str)
    """
    def parse_unit_id(id_str, method_type):
        """Extract probe number and unit number based on method type"""
        if method_type == 'manual':
            parts = id_str.split('_')
            probe_num = parts[-2][-1]
            unit_num = parts[-1]
        else:
            parts = id_str.split('.')
            probe_num = parts[0][-1]
            unit_num = parts[2]
        return probe_num, unit_num
    
    def is_subset(str1, str2):
        """Check if any odor in str1 appears in str2"""
        if not str1 or not str2:
            return False
        set1 = set(str1.split(','))
        set2 = set(str2.split(','))
        return bool(set1 & set2)
    
    # Generate folder name and handle existing folders
    if prefix is None:
        prefix = f"{method_types[0]}_{method_types[1]}"
    
    folder_name = f"{prefix}_comparison"
    folder_path = Path(save_path) / folder_name
    
    # Handle existing folder
    if folder_path.exists():
        suffix = 1
        while (folder_path.parent / f"{folder_name}_{suffix}").exists():
            suffix += 1
        new_folder_name = f"{folder_name}_{suffix}"
        folder_path = folder_path.parent / new_folder_name
        warnings.warn(f"Folder {folder_name} already exists. Creating {new_folder_name} instead.")
    
    # Create the folder
    folder_path.mkdir(parents=True)
    
    # Initialize result containers
    results = {
        'perfect_match': [],
        'loose_match': [],
        'only_method1': [],
        'only_method2': []
    }
    
    # Get appropriate column names
    resp_cols = {
        'manual': ('Block1_responsivity', 'Block2_responsivity'),
        'auto': ('Block1_responsivity', 'Block2_responsivity')
    }
    
    # Create dictionary for second DataFrame lookup
    df2_dict = {}
    for _, row in df2.iterrows():
        probe_num, unit_num = parse_unit_id(
            row['cell_ID' if method_types[1] == 'manual' else 'global_unit_id'], 
            method_types[1]
        )
        df2_dict[(probe_num, unit_num)] = row.to_dict()
    
    # Compare entries
    for _, row1 in df1.iterrows():
        probe_num, unit_num = parse_unit_id(
            row1['cell_ID' if method_types[0] == 'manual' else 'global_unit_id'], 
            method_types[0]
        )
        key = (probe_num, unit_num)
        
        if key not in df2_dict:
            continue
            
        row2 = df2_dict[key]
        
        # Get responsivity values
        df1_block1 = row1[resp_cols[method_types[0]][0]]
        df1_block2 = row1[resp_cols[method_types[0]][1]]
        df2_block1 = row2[resp_cols[method_types[1]][0]]
        df2_block2 = row2[resp_cols[method_types[1]][1]]
        
        # Get global_unit_id from the auto method
        if method_types[0] == 'auto':
            global_id = row1['global_unit_id']
        else:  # method_types[1] must be 'auto'
            global_id = row2['global_unit_id']
        
        result_tuple = (
            global_id,
            df1_block1,
            df1_block2,
            df2_block1,
            df2_block2
        )
        
        # Check selectivity
        method1_selective = bool(df1_block1 or df1_block2)
        method2_selective = bool(df2_block1 or df2_block2)
        
        # Perfect match check
        if (df1_block1 == df2_block1 and 
            df1_block2 == df2_block2 and 
            (method1_selective or method2_selective)):
            results['perfect_match'].append(result_tuple)
            results['loose_match'].append(result_tuple)
            continue
            
        # Loose match check
        if ((is_subset(df1_block1, df2_block1) or is_subset(df2_block1, df1_block1)) or
            (is_subset(df1_block2, df2_block2) or is_subset(df2_block2, df1_block2))):
            results['loose_match'].append(result_tuple)
            continue
            
        # Only method1 check
        if method1_selective and not method2_selective:
            # For 'only_method1', use cell_ID if method1 is manual
            if method_types[0] == 'manual':
                method1_tuple = (
                    row1['cell_ID'],
                    df1_block1,
                    df1_block2,
                    df2_block1,
                    df2_block2
                )
                results['only_method1'].append(method1_tuple)
            else:
                results['only_method1'].append(result_tuple)
            
        # Only method2 check
        elif method2_selective and not method1_selective:
            # For 'only_method2', use cell_ID if method2 is manual
            if method_types[1] == 'manual':
                method2_tuple = (
                    row2['cell_ID'],
                    df1_block1,
                    df1_block2,
                    df2_block1,
                    df2_block2
                )
                results['only_method2'].append(method2_tuple)
            else:
                results['only_method2'].append(result_tuple)
    
    # Save results to CSVs in the created folder
    for category, tuples in results.items():
        if tuples:
            df = pd.DataFrame(tuples, columns=[
                'cell_ID' if (category == 'only_method1' and method_types[0] == 'manual') or 
                            (category == 'only_method2' and method_types[1] == 'manual')
                else 'global_unit_id',
                'Method1_Block1', 
                'Method1_Block2',
                'Method2_Block1', 
                'Method2_Block2'
            ])
            filename = folder_path / f"{category}.csv"
            df.to_csv(filename, index=False)
    
    return results, str(folder_path)
