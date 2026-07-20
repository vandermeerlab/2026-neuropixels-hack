import os
import pandas as pd
import shutil
import glob
import warnings

def visualize_comparison_fingerprints(parent_folder):
    """
    Organize visualization files based on comparison CSVs.
    
    Parameters:
    -----------
    parent_folder : str
        Path to the parent directory containing comparison folders and PNG files
    """
    # Find all comparison folders
    comparison_folders = glob.glob(os.path.join(parent_folder, '*_comparison'))
    
    if not comparison_folders:
        warnings.warn(f"No comparison folders found in {parent_folder}")
        return
        
    for comp_folder in comparison_folders:
        # List of possible CSV files
        csv_files = [
            "loose_match.csv",
            "only_method1.csv",
            "only_method2.csv",
            "perfect_match.csv"
        ]
        
        # Process each CSV if it exists
        for csv_file in csv_files:
            csv_path = os.path.join(comp_folder, csv_file)
            if not os.path.exists(csv_path):
                continue
                
            # Create subfolder (removing .csv extension)
            subfolder_name = os.path.splitext(csv_file)[0]
            subfolder_path = os.path.join(comp_folder, subfolder_name)
            os.makedirs(subfolder_path, exist_ok=True)
            
            # Copy the CSV file to the subfolder
            try:
                shutil.copy2(csv_path, os.path.join(subfolder_path, csv_file))
            except Exception as e:
                warnings.warn(f"Error copying CSV file {csv_file}: {str(e)}")
            
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Determine which column name is used
            id_column = 'global_unit_id' if 'global_unit_id' in df.columns else 'cell_ID'
            
            # Process each unit
            for _, row in df.iterrows():
                unit_id = row[id_column]
                
                try:
                    # Extract imec number and unit number regardless of format
                    if id_column == 'global_unit_id':
                        # Format: imec0.shank0.197
                        parts = unit_id.split('.')
                        if len(parts) != 3:
                            warnings.warn(f"Unexpected unit_id format: {unit_id}")
                            continue
                        imec_num = parts[0]  # imec0
                        unit_num = parts[2]   # 197
                    else:
                        # Format: any format containing imec0 and 197
                        parts = unit_id.split('_')
                        imec_parts = [p for p in parts if p.startswith('imec')]
                        if not imec_parts:
                            warnings.warn(f"No imec number found in cell_ID: {unit_id}")
                            continue
                        imec_num = imec_parts[0]
                        # Find the number part
                        numbers = [p for p in parts if p.isdigit()]
                        if not numbers:
                            warnings.warn(f"No unit number found in cell_ID: {unit_id}")
                            continue
                        unit_num = numbers[-1]
                    
                    # Use the same search pattern for both formats
                    search_pattern = f"*_{imec_num}_{unit_num}_*.png"
                    
                    # Search for matching PNG in parent directory
                    matching_files = glob.glob(os.path.join(parent_folder, search_pattern))
                    
                    if not matching_files:
                        warnings.warn(f"No matching PNG found for unit {unit_id}")
                        continue
                        
                    if len(matching_files) > 1:
                        warnings.warn(f"Multiple matching PNGs found for unit {unit_id}: {matching_files}")
                        
                    # Copy the PNG file(s) to the appropriate subfolder
                    for png_file in matching_files:
                        dest_path = os.path.join(subfolder_path, os.path.basename(png_file))
                        try:
                            shutil.copy2(png_file, dest_path)
                        except Exception as e:
                            warnings.warn(f"Error copying {png_file}: {str(e)}")
                            
                except Exception as e:
                    warnings.warn(f"Error processing unit {unit_id}: {str(e)}")
                    continue
                    