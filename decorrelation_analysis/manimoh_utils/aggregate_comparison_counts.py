import os
import pandas as pd
import glob

def aggregate_comparison_counts(parent_folders, method):
    """
    Aggregate row counts from comparison CSVs across multiple folders.
    
    Parameters:
    -----------
    parent_folders : list
        List of parent directory paths
    method : str
        Method keyword to search for in comparison folder names
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with counts for each comparison type
    """
    results = []
    
    for parent_folder in parent_folders:
        # Find the comparison folder
        comparison_folder = glob.glob(os.path.join(parent_folder, f'{method}_manual_comparison'))
        
        if not comparison_folder:
            continue
            
        comparison_folder = comparison_folder[0]
        
        # Initialize counts
        counts = {
            'parent_folder': parent_folder,
            'perfect_match': 0,
            'loose_match': 0,
            'only_method1': 0,
            'only_method2': 0
        }
        
        # Count rows in each CSV
        for csv_name in ['perfect_match.csv', 'loose_match.csv', 'only_method1.csv', 'only_method2.csv']:
            csv_path = os.path.join(comparison_folder, csv_name)
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                counts[csv_name.replace('.csv', '')] = len(df)
        
        results.append(counts)
    
    # Create DataFrame from results
    return pd.DataFrame(results)

