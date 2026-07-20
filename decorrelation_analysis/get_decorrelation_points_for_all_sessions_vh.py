import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
import pandas as pd
import os
import logging
import shutil

# Assuming manimoh_utils folder and _init_.py is in the same directory
from manimoh_utils import *
from EG_utils import *

# List of input directories to process
input_dir_list = [
    'D:\\vCA1\\M593\\preprocessed\\M593_2025_12_27',
    'D:\\vCA1\\M653\\preprocessed\\M653_2026_02_04',
    'D:\\vCA1\\M656\\preprocessed\\M656_2026_03_25',
    'D:\\vCA1\\M656\\preprocessed\\M656_2026_03_26_conservativeSpikeSorting',
    'D:\\vCA1\\M649\\preprocessed\\M649_2026_03_30_conservativeSpikeSorting',
    'D:\\vCA1\\M649\\preprocessed\\M649_2026_04_01',
    'D:\\vCA1\\M560\\preprocessed\\M560_2025_12_19'
    ]

# Convert string paths to Path objects
input_dir_paths = [Path(input_dir) for input_dir in input_dir_list]

# Process each input directory
for input_dir in input_dir_paths:
    try:
        print(f"Started processing directory {input_dir}:")
        
        # Create output directory - delete if exists first
        output_dir = input_dir / "decorr_analysis"
        if output_dir.exists():
            print(f"Removing existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up logging to file in the decorr_analysis folder
        log_file = output_dir / "decorr_analysis.log"
        logging.basicConfig(
            filename=str(log_file),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True
        )
        
        # Find nwb file in input_dir
        try:
            nwb_filepath = next(input_dir.rglob('*.nwb'))
            logging.info(f"Found NWB file: {nwb_filepath}")
        except StopIteration:
            print(f"No NWB file found in {input_dir}")
            logging.error(f"No NWB file found in {input_dir}")
            continue
        
        session_metadata = parse_expkeys(input_dir)
        logging.info(f"Parsed session metadata: Subject={session_metadata['subject']}, Date={session_metadata['date']}")
        
        # Load nwb file
        data = nap.load_file(str(nwb_filepath))
        spikes = data['units']
        logging.info(f"Loaded NWB file with {len(spikes.keys())} units")
        
        # Apply hc mask to spiking data - keep dorsal and ventral hippocampus units
        electrode_table = data.nwb.electrodes.to_dataframe()
        threshDH = 300  # microns
        threshVH = 300  # microns
        hc_mask = apply_hc_mask_vh(electrode_table, threshDH, threshVH, spikes, session_metadata)
        spikes = spikes[hc_mask]
        logging.info(f"Applied hippocampus mask, {np.sum(hc_mask)} units remaining")
        
        # Get epoch times - dynamically determine number of blocks
        buffer = 15  # seconds
        epoch_names = []
        epoch_times = []

        # Find all blocks in data
        block_epochs = []
        block_num = 1
        while f'Block {block_num}' in data.keys():
            block_epochs.append(nap.IntervalSet(data[f'Block {block_num}']))
            block_num += 1

        if len(block_epochs) == 0:
            print(f"No blocks found in {input_dir}")
            logging.error(f"No blocks found in {input_dir}")
            continue

        # Handle single block case (no buffer, only one epoch)
        if len(block_epochs) == 1:
            epoch_names = ['block1']
            epoch_times = np.array([[block_epochs[0].start[0], block_epochs[0].end[0]]])
            logging.info(f"Single block detected. Defined 1 epoch: {epoch_names}")
            logging.info(f"Defined epoch times: {epoch_times}")

        # Handle multiple blocks case (with buffer)
        else:
            epoch_names = ['pre-rest']
            epoch_times = []
    
            # First epoch: pre-rest (before first block)
            epoch_times.append([0, block_epochs[0].start[0] - buffer])
    
            # Add block and rest epochs
            for block_idx, block_epoch in enumerate(block_epochs):
                # Add block epoch
                epoch_names.append(f'block{block_idx + 1}')
                epoch_times.append([block_epoch.start[0], block_epoch.end[0]])
        
                # Add rest epoch after block (except after the last block)
                if block_idx < len(block_epochs) - 1:
                    epoch_names.append(f'mid-rest{block_idx + 1}')
                    next_block_start = block_epochs[block_idx + 1].start[0]
                    epoch_times.append([block_epoch.end[0] + buffer, next_block_start - buffer])
    
            # Last epoch: post-rest (after last block)
            epoch_names.append('post-rest')
            last_block_end = block_epochs[-1].end[0]
            if 'block3start' in session_metadata.keys():
                epoch_times.append([last_block_end + buffer, session_metadata['block3start'] - buffer])
            else:
                epoch_times.append([last_block_end + buffer, data['LFP'].times()[-1] - buffer])
        
            epoch_times = np.array(epoch_times)
            logging.info(f"Multiple blocks detected. Defined {len(epoch_names)} epochs: {epoch_names}")
            logging.info(f"Defined epoch times: {epoch_times}")

        # Initialize tracking dictionary for decorrelation times upfront with "-1" for all shanks
        decorr_times = {}
        for pid in range(2):
            for sid in range(4):
                shank_key = f"imec{pid}.shank{sid}"
                decorr_times[shank_key] = ["-1"] * len(epoch_names)
        
        # Create the shank correlation plot
        plt.figure(figsize=(24, 12))
        big_font = 12
        small_font = 10
        # parameters for correlation plot / identifying decorrelation points - EG & MM selected different ones
        tbin = 10  # in seconds
        wr = 5  # wiggle room in bins
        row_corr_diff = -0.5  # right - left 
        col_corr_drop = -0.25  # drop threshold for col_diff
        left_mean = 0.25  # Mean correlation value for left half
        right_mean = -0.05  # Mean correlation value for right half
        
        # Initialize dictionary to store analysis results
        shank_dict = {}
        
        # Loop through all probes and shanks
        for pid in range(2):
            for sid in range(4):
                shank_key = f"imec{pid}.shank{sid}"
                shank_dict[shank_key] = {}
                
                # Create a mask for this shank
                shank_mask = [shank_key in spikes['global_id'].values[x] for x in range(len(spikes.keys()))]
                
                # Skip shanks with fewer than 2 units
                if np.sum(shank_mask) <= 1:
                    print(f"Less than 2 units in {shank_key}, n = {np.sum(shank_mask)}")
                    logging.info(f"Less than 2 units in {shank_key}, n = {np.sum(shank_mask)}")
                    # Skip to the next shank - decorr_times already has -1 values for this shank
                    continue
                    
                # Count spikes in bins
                Q_spikes = spikes[shank_mask].count(tbin)
                Q_spikes_matrix = Q_spikes.values.T
                
                # 0 - 1 norming the spikes matrix
                Q_spikes_matrix = (Q_spikes_matrix - np.min(Q_spikes_matrix, axis=1)[:, np.newaxis]) / \
                    (np.max(Q_spikes_matrix, axis=1) - np.min(Q_spikes_matrix, axis=1))[:, np.newaxis]
                
                # Save Q_spikes_matrix and spikes_times for later use
                shank_dict[shank_key]['Q_spikes_matrix'] = Q_spikes_matrix
                shank_dict[shank_key]['spikes_times'] = Q_spikes.times()
                
                # Calculate correlation matrix
                Q_corr = np.corrcoef(Q_spikes_matrix, rowvar=False)
                
                # Skip if correlation matrix has NaN values
                if np.isnan(Q_corr).any():
                    print(f"Correlation matrix not well formed for {shank_key}, n = {np.sum(shank_mask)}")
                    logging.info(f"Correlation matrix not well formed for {shank_key}, n = {np.sum(shank_mask)}")
                    # Skip to the next shank - decorr_times already has -1 values for this shank
                    continue
                    
                # Plot heatmap of Q_corr
                ax = plt.subplot(2, 4, pid * 4 + (sid + 1))
                sns.heatmap(Q_corr)
                time_axis = Q_spikes.times()
                tick_loc = np.arange(0, len(time_axis), 30)
                tick_labels = np.round(time_axis[tick_loc])
                
                # Set the axis labels
                ax.set_xticks([], [])
                ax.set_yticks([], [])
                ax.set_title(f"shank {shank_key}, n = {Q_spikes_matrix.shape[0]}")
                
                # Now do epoch-wise analysis
                for iEpoch in range(len(epoch_names)):
                    epoch_name = epoch_names[iEpoch]
                    shank_dict[shank_key][iEpoch] = {}
                    
                    # Draw boxes around block timings
                    start_idx = np.argmin(abs(time_axis - epoch_times[iEpoch][0]))
                    stop_idx = np.argmin(abs(time_axis - epoch_times[iEpoch][1]))
                    shank_dict[shank_key][iEpoch]['time_idx'] = (start_idx, stop_idx)
                    width = stop_idx - start_idx
                    rect = patches.Rectangle((start_idx, start_idx), 
                                           width, width, linewidth=1, edgecolor='black', facecolor='none')
                    plt.gca().add_patch(rect)
                    plt.gca().text(start_idx + 10, start_idx + 10, epoch_name, fontsize=big_font, color='black')
                    
                    # Analyze correlation structure within the epoch
                    this_res = analyze_correlation_structure(Q_corr, start_idx, stop_idx, wr)
                    # Save results for later use
                    shank_dict[shank_key][iEpoch]['corr_structure'] = this_res
                    
                    # Find the decorrelation boundary
                    boundary = find_decorrelation_boundary(this_res, diff_thresh=row_corr_diff,
                                                         drop_thresh=col_corr_drop, 
                                                         left_thresh=left_mean, 
                                                         right_thresh=right_mean
                                                         ) # EFBG note -  hard-coded left and right mean thresholds may not be generalizing well across epochs/sessions
                    
                    if boundary is None:
                        print(f"No suitable boundary found for {epoch_name}, {shank_key}")
                        logging.info(f"No suitable boundary found for {epoch_name}, {shank_key}")
                        shank_dict[shank_key][iEpoch]['boundary'] = None
                        # Keep the default "-1" value
                    else:
                        shank_dict[shank_key][iEpoch]['boundary'] = boundary
                        decorr_time = time_axis[boundary[0]]
                        # Replace the value at this index
                        decorr_times[shank_key][iEpoch] = str(decorr_time)
                        logging.info(f"Found boundary for {shank_key}, {epoch_name} at time {decorr_time}")
                        
                        plt.axvline(x=boundary[0], color='green', linestyle='--')
                        ax.text(start_idx - width, stop_idx - 0.75 * width,
                              f"drop:{round(boundary[1], 2)}", fontsize=small_font, color='black')
                        ax.text(start_idx - width, stop_idx - 0.5 * width,
                              f"right-left diff: {round(boundary[2], 2)}", fontsize=small_font, color='black')
                        ax.text(start_idx - width, stop_idx - 0.25 * width,
                              f"Left-mean: {round(boundary[3], 2)}", fontsize=small_font, color='black')
                        ax.text(start_idx - width, stop_idx,
                              f"Right-mean: {round(boundary[4], 2)}", fontsize=small_font, color='black')
                        
                    # Export correlation analysis to CSV for all epochs (whether boundary found or not)
                    try:
                        this_res = shank_dict[shank_key][iEpoch]['corr_structure']
                        this_res_df = pd.DataFrame(this_res)
                        this_res_df['col_diff'] = [x[1] for x in this_res_df['col_diff']]
                        this_res_df['left_mean'] = [x[1] for x in this_res_df['left_mean']]
                        this_res_df['right_mean'] = [x[1] for x in this_res_df['right_mean']]
                        this_res_df['row_diff_mean'] = [x[1] for x in this_res_df['row_diff_mean']]
                        
                        # Export dataframe
                        csv_path = output_dir / f"{shank_key}_{epoch_name}_corr_analysis.csv"
                        this_res_df.to_csv(csv_path)
                        logging.info(f"Exported correlation analysis to {csv_path}")
                    except Exception as e:
                        print(f"Error exporting correlation analysis for {shank_key}, {epoch_name}: {str(e)}")
                        logging.error(f"Error exporting correlation analysis for {shank_key}, {epoch_name}: {str(e)}")
                
                ax.set_box_aspect(1)
        
        # Save the correlation matrix figure
        plt.suptitle(f'Correlation of Q-matrix for {session_metadata["subject"]} {session_metadata["date"]}, separated shank-wise')
        plt.savefig(str(output_dir / "shank_separated_pop_correlation.png"), dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved shank correlation plot to {output_dir / 'shank_separated_pop_correlation.png'}")
        
        # Now create spike train plots for all shank/epoch combinations with detected boundaries
        for pid in range(2):
            for sid in range(4):
                shank_key = f"imec{pid}.shank{sid}"
                
                # Skip if this shank was not analyzed (had too few units or NaN correlations)
                if shank_key not in shank_dict or not shank_dict[shank_key]:
                    continue
                
                # Skip if Q_spikes_matrix is not available for this shank
                if 'Q_spikes_matrix' not in shank_dict[shank_key]:
                    continue
                
                for iEpoch in range(len(epoch_names)):
                    epoch_name = epoch_names[iEpoch]
                    
                    # Skip if this epoch was not analyzed for this shank
                    if iEpoch not in shank_dict[shank_key]:
                        continue
                        
                    # Skip if time_idx is not available for this shank/epoch
                    if 'time_idx' not in shank_dict[shank_key][iEpoch]:
                        continue
                    
                    # Create spike trains plot
                    try:
                        spike_matrix = shank_dict[shank_key]['Q_spikes_matrix']
                        ncount = spike_matrix.shape[0]
                        tvec = shank_dict[shank_key]['spikes_times']
                        
                        plt.figure(figsize=(12, 8))
                        ncols = 4
                        nrows = int(np.ceil(ncount / ncols))
                        
                        for i in range(ncount):
                            plt.subplot(nrows, ncols, i + 1)
                            plt.plot(tvec, spike_matrix[i])
                            
                            # Draw boundary line if one was found
                            boundary = shank_dict[shank_key][iEpoch].get('boundary')
                            if boundary is not None:
                                plt.axvline(x=tvec[boundary[0]], color='green', linestyle='--')
                                
                            this_epoch_times = shank_dict[shank_key][iEpoch]['time_idx']
                            plt.xlim(tvec[this_epoch_times[0]], tvec[this_epoch_times[1]])
                            
                            # Have x ticks only for the last row
                            if i < ncount - ncols:
                                plt.xticks([], [])
                            plt.yticks([], [])
                            
                        plt.suptitle(f"Firing rate for {shank_key} in {epoch_name}")
                        plt.tight_layout()
                        
                        spike_plot_path = output_dir / f"{shank_key}_{epoch_name}_spike_trains.png"
                        plt.savefig(str(spike_plot_path), dpi=300, bbox_inches='tight')
                        plt.close()
                        logging.info(f"Saved spike trains plot to {spike_plot_path}")
                    except Exception as e:
                        print(f"Error creating spike_trains plot for {shank_key}, {epoch_name}: {str(e)}")
                        logging.error(f"Error creating spike_trains plot for {shank_key}, {epoch_name}: {str(e)}")
        
        # Write decorrelation times to ExpKeys_Lines.txt
        try:
            expkeys_path = output_dir / "ExpKeys_Lines.txt"
            with open(str(expkeys_path), 'w') as f:
                # Ensure we write all 8 possible shanks
                for pid in range(2):
                    for sid in range(4):
                        shank_key = f"imec{pid}.shank{sid}"
                        # Get times, or default to all -1s if not present
                        times = decorr_times.get(shank_key, ["-1"] * len(epoch_names))
                        times_str = ", ".join(times)
                        # Replace the dot with underscore in shank_key for ExpKeys
                        expkeys_key = shank_key.replace(".", "_")
                        f.write(f"ExpKeys.{expkeys_key}_decorr_times = {{{times_str}}};\n")
            logging.info(f"Saved decorrelation times to {expkeys_path}")
        except Exception as e:
            print(f"Error writing ExpKeys_Lines.txt: {str(e)}")
            logging.error(f"Error writing ExpKeys_Lines.txt: {str(e)}")
        
        print(f"Finished processing directory {input_dir}")
        logging.info(f"Finished processing directory {input_dir}")
    except Exception as e:
        print(f"Error processing directory {input_dir}: {str(e)}")
        logging.error(f"Error processing directory {input_dir}: {str(e)}", exc_info=True)
        continue