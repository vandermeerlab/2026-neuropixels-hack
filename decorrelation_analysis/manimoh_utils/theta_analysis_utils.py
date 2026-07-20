import pynapple as nap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
import os
from pathlib import Path
from scipy import signal, stats


def resultant_length(angles):
    """Calculate resultant length (PLV) for a set of angles"""
    n = len(angles)
    if n == 0:
        return 0
    # Ensure angles are wrapped to [-π, π]
    angles = np.angle(np.exp(1j * angles))
    return np.abs(np.sum(np.exp(1j * angles))) / n


def circular_mean(angles):
    """Calculate circular mean of angles using scipy"""
    if len(angles) == 0:
        return 0
    # Ensure angles are wrapped to [-π, π]
    angles = np.angle(np.exp(1j * angles))
    return stats.circmean(angles, low=-np.pi, high=np.pi)


def calculate_plv_vectorized(spike_phases, n_subsamples=1000, subsample_size=200):
    """Vectorized PLV calculation with unique elements per subsample"""
    
    # Convert to numpy array if it's a pynapple object
    if hasattr(spike_phases, 'values'):
        spike_phases = spike_phases.values
    elif hasattr(spike_phases, 'd'):
        spike_phases = spike_phases.d
    else:
        spike_phases = np.array(spike_phases)
    
    if len(spike_phases) < subsample_size:
        return np.nan, np.nan
    
    # Generate all random indices - each row is a unique subsample
    all_indices = np.array([np.sort(np.random.choice(len(spike_phases), size=subsample_size, replace=False)) 
                           for _ in range(n_subsamples)])
    
    # Vectorized PLV calculation
    subsampled_phases = spike_phases[all_indices]  # Shape: (n_subsamples, subsample_size)
    
    # Calculate PLV for all subsamples at once
    complex_phases = np.exp(1j * subsampled_phases)
    plv_values = np.abs(np.mean(complex_phases, axis=1))
    mean_phases = np.angle(np.mean(complex_phases, axis=1))
    
    return np.mean(plv_values), np.angle(np.mean(np.exp(1j * mean_phases)))


def generate_shuffled_plvs(spike_ts, theta_phase_ts, n_shuffles=1000, n_subsamples=1000, subsample_size=200):
    """Generate shuffled PLVs by circularly shifting spike times"""
    shuffled_plvs = []
    
    for _ in range(n_shuffles):
        # Circularly shift spike times
        shifted_ts = nap.process.randomize.shift_timestamps(spike_ts, min_shift=1, max_shift=500)
        
        # Get theta phases at shifted spike times
        shifted_phases = shifted_ts.value_from(theta_phase_ts)
        
        # Remove NaN values and wrap phases to [-π, π]
        shifted_phases = shifted_phases[~np.isnan(shifted_phases)]
        shifted_phases = np.angle(np.exp(1j * shifted_phases))  # Wrap to [-π, π]
        
        if len(shifted_phases) >= subsample_size:
            # Calculate subsampled PLV for this shuffle
            shuffled_plv, _ = calculate_plv_vectorized(shifted_phases, n_subsamples, subsample_size)
            if not np.isnan(shuffled_plv):
                shuffled_plvs.append(shuffled_plv)
    
    return np.array(shuffled_plvs)


def compute_resultant_vector(phases, plvs, weighted=False):
    """
    Compute resultant vector from phases and PLVs
    If weighted=True, uses PLV values as weights
    Returns: resultant_length, resultant_phase
    Resultant_phase is between -π and π
    """
    if weighted:
        # Weighted by PLV
        x_components = plvs * np.cos(phases)
        y_components = plvs * np.sin(phases)
        resultant_x = np.sum(x_components) / len(phases)
        resultant_y = np.sum(y_components) / len(phases)
    else:
        # Unweighted - all vectors normalized to unit length
        x_components = np.cos(phases)
        y_components = np.sin(phases)
        resultant_x = np.mean(x_components)
        resultant_y = np.mean(y_components)
        # Scale by mean PLV for visualization
        mean_plv = np.mean(plvs)
        resultant_x *= mean_plv
        resultant_y *= mean_plv
    
    resultant_length = np.sqrt(resultant_x**2 + resultant_y**2)
    resultant_phase = np.arctan2(resultant_y, resultant_x)
    
    return resultant_length, resultant_phase


def calculate_all_plvs_for_block(sel_spikes, selective_odors, theta_phase, onsets, epoch, block_name="Block"):
    """
    Calculate PLVs for all conditions in one go for a block
    
    Parameters:
    -----------
    sel_spikes : pynapple TsGroup
        Selected spikes for the block
    selective_odors : array
        Array of odor selectivity for each cell
    theta_phase : pynapple Tsd
        Theta phase time series
    onsets : dict
        Dictionary of odor onsets
    epoch : pynapple IntervalSet
        Block epoch
    block_name : str
        Name for logging purposes
        
    Returns:
    --------
    results_df : pandas DataFrame
        DataFrame with all PLV results
    """
    
    print(f"\nAnalyzing {block_name}...")
    results = []
    
    for i, cell in enumerate(sel_spikes.index):
        this_ts = sel_spikes[cell]
        this_selectivity = selective_odors[i]
        
        # Get onset times for different conditions
        all_onsets, sel_onsets, other_onsets = [], [], []
        for odor in onsets.keys():
            all_onsets.append(onsets[odor].start)
            if odor == this_selectivity:
                sel_onsets.append(onsets[odor].start)
            else:
                other_onsets.append(onsets[odor].start)
        
        all_onsets = np.hstack(all_onsets)
        other_onsets = np.hstack(other_onsets)
        sel_onsets = np.hstack(sel_onsets)
        all_onsets.sort()
        sel_onsets.sort()
        other_onsets.sort()
        
        # Define different time periods
        all_trial_epoch = nap.IntervalSet(all_onsets, all_onsets + 2.5)
        sel_trial_epoch = nap.IntervalSet(sel_onsets, sel_onsets + 2.5)
        other_trial_epoch = nap.IntervalSet(other_onsets, other_onsets + 2.5)
        off_trial_epoch = epoch.set_diff(all_trial_epoch)
        
        # Restrict spikes to different conditions
        all_spikes_ts = this_ts  # All spikes in the block
        trial_spikes_ts = this_ts.restrict(all_trial_epoch)
        sel_trial_spikes_ts = this_ts.restrict(sel_trial_epoch)
        other_trial_spikes_ts = this_ts.restrict(other_trial_epoch)
        off_trial_spikes_ts = this_ts.restrict(off_trial_epoch)
        
        # Get theta phases for each condition
        conditions = {
            'all_spikes': all_spikes_ts,
            'trial_spikes': trial_spikes_ts,
            'sel_trial_spikes': sel_trial_spikes_ts,
            'other_trial_spikes': other_trial_spikes_ts,
            'off_trial_spikes': off_trial_spikes_ts
        }
        
        cell_result = {
            'global_ID': sel_spikes['global_id'][cell],
            'odor_selectivity': this_selectivity
        }
        
        # Calculate PLV for each condition
        for condition_name, spike_ts in conditions.items():
            theta_phases = spike_ts.value_from(theta_phase)
            
            # Remove NaN values and wrap phases to [-π, π]
            valid_phases = theta_phases[~np.isnan(theta_phases)]
            valid_phases = np.angle(np.exp(1j * valid_phases))
            
            if len(valid_phases) >= 200:
                plv, mean_phase = calculate_plv_vectorized(valid_phases)
                cell_result[f'{condition_name}_PLV'] = plv
                cell_result[f'{condition_name}_mean_phase'] = mean_phase
            else:
                cell_result[f'{condition_name}_PLV'] = np.nan
                cell_result[f'{condition_name}_mean_phase'] = np.nan
                if len(valid_phases) < 50:  # Only warn for very low spike counts
                    print(f"Warning: Cell {sel_spikes['global_id'][cell]} {condition_name} has insufficient spikes ({len(valid_phases)})")
        
        results.append(cell_result)
    
    return pd.DataFrame(results)


def plot_cell_phase_modulation(results_df, sel_spikes, theta_phase, onsets, epoch, color_code, block_name, save_path=None):
    """
    Plot phase modulation for cells in the specified layout
    
    Parameters:
    -----------
    results_df : pandas DataFrame
        Results from calculate_all_plvs_for_block
    sel_spikes : pynapple TsGroup
        Selected spikes
    theta_phase : pynapple Tsd
        Theta phase time series
    onsets : dict
        Dictionary of odor onsets
    epoch : pynapple IntervalSet
        Block epoch
    color_code : dict
        Color mapping for odors
    block_name : str
        Block name for title
    save_path : str, optional
        Path to save the figure
    """
    
    n_cells = len(results_df)
    if n_cells == 0:
        print(f"No cells with sufficient data for plotting in {block_name}")
        return
    
    # 2 cells per row, 3 columns per cell
    ncols_per_cell = 3
    ncells_per_row = 2
    total_cols = ncells_per_row * ncols_per_cell
    nrows = int(np.ceil(n_cells / ncells_per_row))
    
    fig = plt.figure(figsize=(18, 3 * nrows))
    
    def normalize_and_plot(data, color, ax_to_use, linestyle='-', alpha=1.0):
        """Helper function to normalize data to 0-1 range"""
        if data.shape[0] > 0 and not data.isna().all():
            min_val = data.min()
            max_val = data.max()
            if max_val > min_val:
                normalized = (data - min_val) / (max_val - min_val)
            else:
                normalized = data * 0 + 0.5
            line = ax_to_use.plot(data.index, normalized, color=color, 
                                 linestyle=linestyle, alpha=alpha)[0]
            return line, min_val, max_val
        return None, None, None
    
    for i, (idx, row) in enumerate(results_df.iterrows()):
        row_idx = i // ncells_per_row
        cell_in_row = i % ncells_per_row
        
        # Calculate base column for this cell (each cell takes 3 columns)
        base_col = cell_in_row * ncols_per_cell
        
        # Get cell data
        cell_idx = list(sel_spikes['global_id'].values).index(row['global_ID'])
        this_selectivity = row['odor_selectivity']
        
        # Get onset times
        all_onsets, sel_onsets, other_onsets = [], [], []
        for odor in onsets.keys():
            all_onsets.append(onsets[odor].start)
            if odor == this_selectivity:
                sel_onsets.append(onsets[odor].start)
            else:
                other_onsets.append(onsets[odor].start)
        
        all_onsets = np.hstack(all_onsets)
        other_onsets = np.hstack(other_onsets)
        sel_onsets = np.hstack(sel_onsets)
        all_onsets.sort()
        sel_onsets.sort()
        other_onsets.sort()
        
        # Get spike trains for different conditions
        this_ts = sel_spikes[sel_spikes.index[cell_idx]]
        all_trial_epoch = nap.IntervalSet(all_onsets, all_onsets + 2.5)
        
        # Create TsGroups for tuning curve calculation
        all_ts = nap.TsGroup({0: this_ts})
        trial_ts = nap.TsGroup({0: this_ts.restrict(all_trial_epoch)})
        sel_trial_ts = nap.TsGroup({0: this_ts.restrict(nap.IntervalSet(sel_onsets, sel_onsets + 2.5))})
        other_trial_ts = nap.TsGroup({0: this_ts.restrict(nap.IntervalSet(other_onsets, other_onsets + 2.5))})
        off_trial_ts = nap.TsGroup({0: this_ts.restrict(epoch.set_diff(all_trial_epoch))})
        
        # Calculate tuning curves
        all_theta_modulation = nap.compute_1d_tuning_curves(
            group=all_ts, feature=theta_phase, nb_bins=51, minmax=(-np.pi, np.pi))
        trial_theta_modulation = nap.compute_1d_tuning_curves(
            group=trial_ts, feature=theta_phase, nb_bins=51, minmax=(-np.pi, np.pi))
        sel_theta_modulation = nap.compute_1d_tuning_curves(
            group=sel_trial_ts, feature=theta_phase, nb_bins=51, minmax=(-np.pi, np.pi))
        other_theta_modulation = nap.compute_1d_tuning_curves(
            group=other_trial_ts, feature=theta_phase, nb_bins=51, minmax=(-np.pi, np.pi))
        off_theta_modulation = nap.compute_1d_tuning_curves(
            group=off_trial_ts, feature=theta_phase, nb_bins=51, minmax=(-np.pi, np.pi))
        
        # COLUMN 1: All spikes
        ax1 = plt.subplot(nrows, total_cols, row_idx * total_cols + base_col + 1)
        
        if not np.isnan(row['all_spikes_PLV']):
            ax1.plot(all_theta_modulation.iloc[:, 0], color=color_code[this_selectivity])
            ax1.axvline(x=row['all_spikes_mean_phase'], color='black', linestyle='--', alpha=0.7)
        
        # Titles only for top row
        if row_idx == 0:
            ax1.set_title(f"{row['global_ID']}")
        
        # Y-axis labels only for leftmost cells
        if cell_in_row == 0:
            ax1.set_ylabel("Firing rate (Hz)")
        
        # X-axis labels only for bottom row
        if row_idx == nrows - 1:
            ax1.set_xlabel("Phase bin")
        else:
            ax1.set_xticklabels([])
        
        # COLUMN 2: Selective vs Other trials
        ax2 = plt.subplot(nrows, total_cols, row_idx * total_cols + base_col + 2)
        ax2_right = ax2.twinx()
        
        # Plot selective trials on left axis
        if not np.isnan(row['sel_trial_spikes_PLV']) and sel_trial_ts[0].shape[0] > 200:
            line, sel_min, sel_max = normalize_and_plot(
                sel_theta_modulation.iloc[:, 0], 
                color_code[this_selectivity], 
                ax2
            )
            if line is not None:
                ax2.axvline(x=row['sel_trial_spikes_mean_phase'], 
                           color=color_code[this_selectivity], linestyle='--', alpha=0.7)
        
        # Plot other trials on right axis
        if not np.isnan(row['other_trial_spikes_PLV']) and other_trial_ts[0].shape[0] > 200:
            line, other_min, other_max = normalize_and_plot(
                other_theta_modulation.iloc[:, 0], 
                'gray', 
                ax2_right
            )
            if line is not None:
                ax2_right.axvline(x=row['other_trial_spikes_mean_phase'], 
                                 color='gray', linestyle='--', alpha=0.7)
        
        # Titles only for top row
        if row_idx == 0:
            ax2.set_title("Target vs Others")
        
        # X-axis labels only for bottom row
        if row_idx == nrows - 1:
            ax2.set_xlabel("Phase bin")
        else:
            ax2.set_xticklabels([])
        
        # COLUMN 3: In-trial vs Off-trial
        ax3 = plt.subplot(nrows, total_cols, row_idx * total_cols + base_col + 3)
        ax3_right = ax3.twinx()
        
        # Plot trial spikes on left axis
        if not np.isnan(row['trial_spikes_PLV']) and trial_ts[0].shape[0] > 200:
            line, trial_min, trial_max = normalize_and_plot(
                trial_theta_modulation.iloc[:, 0], 
                'black', 
                ax3
            )
            if line is not None:
                ax3.axvline(x=row['trial_spikes_mean_phase'], 
                           color='black', linestyle='--', alpha=0.7)
        
        # Plot off-trial spikes on right axis
        if not np.isnan(row['off_trial_spikes_PLV']) and off_trial_ts[0].shape[0] > 200:
            line, off_min, off_max = normalize_and_plot(
                off_theta_modulation.iloc[:, 0], 
                'teal', 
                ax3_right
            )
            if line is not None:
                ax3_right.axvline(x=row['off_trial_spikes_mean_phase'], 
                                 color='teal', linestyle='--', alpha=0.7)
        
        # Titles only for top row
        if row_idx == 0:
            ax3.set_title("In-trial vs Out")
        
        # X-axis labels only for bottom row
        if row_idx == nrows - 1:
            ax3.set_xlabel("Phase bin")
        else:
            ax3.set_xticklabels([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved cell-wise plot to {save_path}")
    
    plt.close()


def plot_plv_summary(b1_results_df, b2_results_df, sess_metadata, color_code, weighted=True, save_path=None):
    """
    Create 5x2 summary polar plots
    
    Parameters:
    -----------
    b1_results_df : pandas DataFrame
        Block 1 results
    b2_results_df : pandas DataFrame
        Block 2 results
    sess_metadata : dict
        Session metadata containing block types
    color_code : dict
        Color mapping for odors
    weighted : bool
        Whether to use weighted circular mean
    save_path : str, optional
        Path to save the figure
    """
    
    conditions = [
        ('all_spikes', 'All spikes'),
        ('sel_trial_spikes', 'Selective odor trials'),
        ('other_trial_spikes', 'Other trials'),
        ('trial_spikes', 'All odor trials'),
        ('off_trial_spikes', 'Off trials')
    ]
    
    fig = plt.figure(figsize=(12, 20))
    gs = GridSpec(5, 2, hspace=0.3, wspace=0.2)
    
    # Block titles
    block1_title = f"Block 1: {'Random' if sess_metadata.get('block1_type')[0] == 'UE' else 'Predictable'}"
    block2_title = f"Block 2: {'Random' if sess_metadata.get('block2_type')[0] == 'UE' else 'Predictable'}"
    
    # Parameters for plotting
    individual_cell_linewidth = 2
    resultant_vector_linewidth = 4
    arrow_linewidth = 4
    
    for row_idx, (condition, condition_label) in enumerate(conditions):
        for col_idx, (results_df, block_title) in enumerate([(b1_results_df, block1_title), 
                                                            (b2_results_df, block2_title)]):
            
            ax = plt.subplot(gs[row_idx, col_idx], projection='polar')
            
            # Filter out NaN values for this condition
            good_df = results_df.dropna(subset=[f'{condition}_PLV', f'{condition}_mean_phase'])
            
            if len(good_df) > 0:
                # Plot individual cells
                for i, row in good_df.iterrows():
                    phase = row[f'{condition}_mean_phase']
                    plv = row[f'{condition}_PLV']
                    color = color_code[row['odor_selectivity']]
                    ax.plot([phase, phase], [0, plv], color=color, alpha=0.3, 
                           linewidth=individual_cell_linewidth)
                
                # Calculate and plot resultant vectors for each odor
                legend_handles = []
                legend_labels = []
                
                for odor in good_df['odor_selectivity'].unique():
                    odor_data = good_df[good_df['odor_selectivity'] == odor]
                    if len(odor_data) > 0:
                        phases = odor_data[f'{condition}_mean_phase'].values
                        plvs = odor_data[f'{condition}_PLV'].values
                        
                        # Remove any remaining NaN values
                        mask = ~(np.isnan(phases) | np.isnan(plvs))
                        phases = phases[mask]
                        plvs = plvs[mask]
                        
                        if len(phases) > 0:
                            # Compute resultant vector
                            resultant_length, resultant_phase = compute_resultant_vector(
                                phases, plvs, weighted=weighted)
                            
                            # Plot thick line for resultant vector
                            line = ax.plot([resultant_phase, resultant_phase], [0, resultant_length], 
                                         color=color_code[odor], linewidth=resultant_vector_linewidth)[0]
                            
                            # Add arrow at the outer circle
                            ax.annotate('', xy=(resultant_phase, 0.6), xytext=(resultant_phase, 0.55),
                                       arrowprops=dict(arrowstyle='->', color=color_code[odor], 
                                                     lw=arrow_linewidth))
                            
                            # Store for legend (only for top row)
                            if row_idx == 0:
                                legend_handles.append(line)
                                legend_labels.append(odor)
            
            # Set axis properties
            ax.set_ylim(0, 0.6)
            ax.set_rticks([0.1, 0.2, 0.3, 0.4, 0.5])
            ax.grid(True, alpha=0.3)
            
            # Set theta labels
            ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
            ax.set_xticklabels(['0', 'π/2', '±π', '-π/2'])
            
            # Add titles for top row
            if row_idx == 0:
                ax.set_title(block_title, pad=20)
            
            # Add legend for top row
            if row_idx == 0 and legend_handles:
                ax.legend(legend_handles, legend_labels, loc='upper left', 
                         bbox_to_anchor=(1.1, 1))
        
        # Add row labels on the left
        fig.text(0.02, 0.9 - row_idx * 0.18, condition_label, rotation=90, 
                va='center', ha='center', fontsize=12, weight='bold')
    
    # plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary plot to {save_path}")
    
    plt.close()