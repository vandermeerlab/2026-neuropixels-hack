import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from IPython.display import display
import ipywidgets as widgets
from matplotlib.colors import to_hex, to_rgb
from matplotlib.gridspec import GridSpec
from .smooth_utils import smooth_peth

def multi_raster(spikes, onsets, window_size=2.5, events_to_skip=1, figsize=(15, 8)):
    """
    Create a raster plot with colored vertical lines marking stimulus onsets and offsets.
    
    Parameters:
    -----------
    spikes : nap.Ts or nap.TsGroup
        Spike times for one or multiple neurons.
    onsets : dict of nap.IntervalSet
        Dictionary containing stimulus onset times with odor types as keys.
    window_size : float, optional
        Half-width of the time window to display (in seconds). Default is 2.5.
    events_to_skip : int, optional
        Number of events to skip when navigating. Default is 1.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (15, 8).
    """
    # Determine neuron count
    if isinstance(spikes, nap.TsGroup):
        neuron_ids = list(spikes.keys())
    else:  # Single Ts object
        neuron_ids = ["Neuron"]
        spikes = {"Neuron": spikes}
    
    # Create a color map for different odors - we'll allow these to be modified
    unique_odors = list(onsets.keys())
    default_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_odors)))
    odor_colors = {odor: to_hex(default_colors[i]) for i, odor in enumerate(unique_odors)}
    
    # Define a set of available colors for selection
    color_options = {
        "Blue": "#1f77b4",
        "Orange": "#ff7f0e",
        "Green": "#2ca02c",
        "Red": "#d62728",
        "Purple": "#9467bd",
        "Brown": "#8c564b",
        "Pink": "#e377c2",
        "Gray": "#7f7f7f",
        "Olive": "#bcbd22",
        "Cyan": "#17becf",
        "Magenta": "#ff00ff",
        "Lime": "#00ff00",
        "Teal": "#008080",
        "Navy": "#000080",
        "Maroon": "#800000",
        "Yellow": "#ffff00",
        "Black": "#000000"
    }
    
    # Gather all stimulus onset times by type
    all_onsets = []
    odor_types = {"Any": []}  # Dictionary to store indices by odor type
    
    for odor, onset_interval in onsets.items():
        odor_types[odor] = []  # Initialize list for this odor type
        for time in enumerate(onset_interval.start):
            all_onsets.append((time[1], odor))
            odor_types["Any"].append(len(all_onsets) - 1)  # Add to "Any" list
            odor_types[odor].append(len(all_onsets) - 1)   # Add to type-specific list
    
    # Sort by time
    all_onsets.sort()
    
    # Re-index the odor_types dictionary after sorting
    odor_types = {"Any": list(range(len(all_onsets)))}
    for odor in unique_odors:
        odor_types[odor] = [i for i, (_, event_odor) in enumerate(all_onsets) if event_odor == odor]
    
    # Output area for plot
    plot_output = widgets.Output()
    
    # Function to generate the plot
    def create_plot(center_time, window_size=2.5):
        # Set time window
        tmin = center_time - window_size
        tmax = center_time + window_size
        
        # Create a new figure each time
        plt.ioff()  # Turn interactive mode off
        fig, ax = plt.subplots(figsize=figsize)
        
        # Find all events visible in this window
        visible_onsets = [(t, odor) for t, odor in all_onsets if tmin <= t <= tmax]
        
        # Row height adjustment to prevent overlap
        row_height = 0.9
        
        # Shade regions between onset and offset for all visible onsets
        for onset_time, odor in visible_onsets:
            offset_time = onset_time + 2.0  # 2 seconds after onset
            color = odor_colors[odor]
            
            # Add transparent shading between onset and offset
            if tmin <= offset_time <= tmax:  # Only if offset is within view
                ax.axvspan(onset_time, offset_time, ymin=0, ymax=1,
                           color=color, alpha=0.15)
            
            # Plot onset line (solid)
            ax.axvline(x=onset_time, ymin=0, ymax=1, 
                       color=color, linewidth=1.5, alpha=0.8)
            
            # Plot offset line (dotted)
            if tmin <= offset_time <= tmax:  # Only if offset is within view
                ax.axvline(x=offset_time, ymin=0, ymax=1, 
                           color=color, linewidth=1.2, alpha=0.6,
                           linestyle='--')
            
            # Add a small label above the line
            ax.text(onset_time, len(neuron_ids) + 0.2, 
                    f"{odor}", fontsize=8, ha='center', 
                    color=color, weight='bold')
        
        # Plot each neuron on its own row
        for i, neuron_id in enumerate(neuron_ids):
            # Get spikes for this neuron within the time window
            if isinstance(spikes[neuron_id], nap.Ts) or isinstance(spikes[neuron_id], nap.Tsd):
                # Create an interval set for the time window
                interval_set = nap.IntervalSet(start=[tmin], end=[tmax])
                
                # Use the interval set to restrict the spikes
                neuron_spikes = spikes[neuron_id].restrict(interval_set)
                spike_times = neuron_spikes.index
            else:
                # If it's not a Ts/Tsd object, filter manually
                spike_times = [t for t in spikes[neuron_id].index if tmin <= t <= tmax]
            
            # Plot spikes as vertical lines with adjusted height to prevent overlap
            if len(spike_times) > 0:
                y_values = np.ones_like(spike_times) * i
                ax.vlines(spike_times, y_values - row_height/2, y_values + row_height/2, 
                          colors='black', linewidth=1.2)
        
        # Highlight the center event
        center_event = None
        min_dist = float('inf')
        for t, odor in all_onsets:
            dist = abs(t - center_time)
            if dist < min_dist:
                min_dist = dist
                center_event = (t, odor)
                
        if center_event:
            ax.axvline(x=center_event[0], ymin=0, ymax=1, 
                       color=odor_colors[center_event[1]], linewidth=3, alpha=1.0)
            ax.text(center_event[0], len(neuron_ids) + 0.7, 
                    f"{center_event[1]} (SELECTED)", fontsize=10, ha='center', 
                    color=odor_colors[center_event[1]], weight='bold')
        
        # Set labels and limits
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Neuron', fontsize=10)
        ax.set_xlim(tmin, tmax)
        ax.set_ylim(-0.5, len(neuron_ids) + 1.0)
        
        # Set y-ticks with neuron count markers (0, 10, 20, etc.)
        # Find the step size for tick marks (10, 20, 50, etc. depending on neuron count)
        if len(neuron_ids) > 50:
            step = 20
        elif len(neuron_ids) > 20:
            step = 10
        else:
            step = 5
            
        ytick_positions = list(range(0, len(neuron_ids), step))
        if ytick_positions and ytick_positions[-1] != len(neuron_ids) - 1:
            ytick_positions.append(len(neuron_ids) - 1)  # Add the last neuron
            
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels([str(pos) for pos in ytick_positions])
        
        # Create simplified legend for odors (onset only)
        handles = []
        labels = []
        for odor in unique_odors:
            line = plt.Line2D([0], [0], color=odor_colors[odor], lw=2, label=odor)
            handles.append(line)
            labels.append(odor)
        ax.legend(handles=handles, labels=labels, loc='upper right', fontsize=9)
        
        # Add title with time window info
        ax.set_title(f'Raster Plot - Time Window: {tmin:.2f}s to {tmax:.2f}s', fontsize=12)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    # Function to update the display with the current event
    def update_display():
        with plot_output:
            plot_output.clear_output(wait=True)
            if len(all_onsets) > 0 and 0 <= current_event.value < len(all_onsets):
                selected_time = all_onsets[current_event.value][0]
                fig = create_plot(selected_time, window_input.value)
                display(fig)
                plt.close(fig)  # Close the figure to free memory
            else:
                print("Event index out of range.")
    
    # Function to navigate to events by type
    def navigate_by_type(direction, odor_filter, skip_count):
        current_type_indices = odor_types[odor_filter]
        if not current_type_indices:
            return  # No events of this type
            
        # Find the current position in the filtered list
        if current_event.value in current_type_indices:
            current_pos = current_type_indices.index(current_event.value)
        else:
            # Find the closest position
            if direction > 0:
                # Find the first event of this type after the current event
                filtered_future_indices = [i for i in current_type_indices if i > current_event.value]
                if filtered_future_indices:
                    current_event.value = filtered_future_indices[0]
                else:
                    current_event.value = current_type_indices[0]  # Wrap around to first
                return
            else:
                # Find the first event of this type before the current event
                filtered_past_indices = [i for i in current_type_indices if i < current_event.value]
                if filtered_past_indices:
                    current_event.value = filtered_past_indices[-1]
                else:
                    current_event.value = current_type_indices[-1]  # Wrap around to last
                return
        
        # Calculate the new position with skipping
        new_pos = current_pos + (direction * skip_count)
        if direction > 0:
            # Forward navigation
            if new_pos >= len(current_type_indices):
                new_pos = 0  # Wrap around to the beginning
        else:
            # Backward navigation
            if new_pos < 0:
                new_pos = len(current_type_indices) - 1  # Wrap around to the end
                
        # Set the new event index
        current_event.value = current_type_indices[new_pos]
    
    # Create the widgets with expanded widths
    current_event = widgets.IntText(
        value=0,
        description='Event:',
        layout=widgets.Layout(width='180px')
    )
    
    window_input = widgets.FloatText(
        value=window_size,
        description='Window (±s):',
        layout=widgets.Layout(width='200px')
    )
    
    skip_input = widgets.IntText(
        value=events_to_skip,
        description='Skip:',
        layout=widgets.Layout(width='150px')
    )
    
    # Dropdown for event type filtering with expanded width
    event_type_dropdown = widgets.Dropdown(
        options=["Any"] + unique_odors,
        value="Any",
        description='Type:',
        layout=widgets.Layout(width='200px')
    )
    
    # Navigation buttons with consistent width
    prev_button = widgets.Button(
        description='◀ Previous', 
        layout=widgets.Layout(width='150px')
    )
    
    next_button = widgets.Button(
        description='Next ▶', 
        layout=widgets.Layout(width='150px')
    )
    
    update_button = widgets.Button(
        description='Update Plot',
        button_style='primary',
        layout=widgets.Layout(width='150px')
    )
    
    # Create color selector dropdowns for each odor type
    color_selectors = []
    
    # Function to handle color changes
    def on_color_change(change):
        odor = change['owner'].description
        odor_colors[odor] = color_options[change['new']]
        update_display()
    
    # Create a dropdown for each odor type
    for odor in unique_odors:
        # Find the default color name from the hex value
        default_color_name = next((name for name, hex_value in color_options.items() 
                                  if hex_value == odor_colors[odor]), "Blue")
        
        color_selector = widgets.Dropdown(
            options=list(color_options.keys()),
            value=default_color_name,
            description=odor,
            layout=widgets.Layout(width='200px')
        )
        color_selector.observe(on_color_change, names='value')
        color_selectors.append(color_selector)
    
    # Event handlers for type-specific navigation
    def on_prev_clicked(b):
        navigate_by_type(-1, event_type_dropdown.value, skip_input.value)
        update_display()
        
    def on_next_clicked(b):
        navigate_by_type(1, event_type_dropdown.value, skip_input.value)
        update_display()
    
    def manual_update(b):
        update_display()
    
    # Handle event type dropdown changes
    def on_type_change(change):
        # Reset to first event of selected type when type changes
        if change['new'] != change['old'] and odor_types[change['new']]:
            current_event.value = odor_types[change['new']][0]
            update_display()
    
    # Connect event handlers
    prev_button.on_click(on_prev_clicked)
    next_button.on_click(on_next_clicked)
    update_button.on_click(manual_update)
    event_type_dropdown.observe(on_type_change, names='value')
    
    # Create color selector section with responsive layout
    # If there are many odors, organize in multiple columns
    # Create color selector section with responsive layout
    # If there are many odors, organize in multiple columns
    color_widget_rows = []
    colors_per_row = 3
    for i in range(0, len(color_selectors), colors_per_row):
        row = widgets.HBox(color_selectors[i:i+colors_per_row])
        color_widget_rows.append(row)
    
    color_widget_area = widgets.VBox(color_widget_rows)
    
    # Improved layout with better spacing and organization
    navigation_row1 = widgets.HBox([
        current_event, 
        skip_input
    ], layout=widgets.Layout(justify_content='flex-start', width='100%'))
    
    navigation_row2 = widgets.HBox([
        event_type_dropdown, 
        window_input
    ], layout=widgets.Layout(justify_content='flex-start', width='100%'))
    
    button_row = widgets.HBox([
        prev_button, 
        next_button, 
        update_button
    ], layout=widgets.Layout(justify_content='flex-start', width='100%'))
    
    # Organize the controls into sections
    navigation_section = widgets.VBox([
        widgets.HTML("<b>Navigation Controls:</b>"),
        navigation_row1,
        navigation_row2,
        button_row
    ], layout=widgets.Layout(margin='10px 0px 10px 0px', width='100%'))
    
    color_section = widgets.VBox([
        widgets.HTML("<b>Event Color Selection:</b>"),
        color_widget_area
    ], layout=widgets.Layout(margin='10px 0px 10px 0px', width='100%'))
    
    # Display the complete UI
    display(widgets.VBox([navigation_section, color_section, plot_output]))
    
    # Initial plot
    update_display()

def multi_raster2(spikes, onsets, window_size=2.5, events_to_skip=1, figsize=(15, 8), 
                 font_size=15, hist_height_ratio=0.2, bin_size=0.01):
    """
    Create a raster plot with colored vertical lines marking stimulus onsets and offsets,
    along with a histogram showing average activity.
    
    Parameters:
    -----------
    spikes : nap.Ts or nap.TsGroup
        Spike times for one or multiple neurons.
    onsets : dict of nap.IntervalSet
        Dictionary containing stimulus onset times with odor types as keys.
    window_size : float, optional
        Half-width of the time window to display (in seconds). Default is 2.5.
    events_to_skip : int, optional
        Number of events to skip when navigating. Default is 1.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (15, 8).
    font_size : int, optional
        Font size for labels, titles, and legends. Default is 15.
    hist_height_ratio : float, optional
        Height ratio of histogram to raster plot. Default is 0.2.
    bin_size : float, optional
        Bin size for histogram in seconds. Default is 0.01.
    """
    # Determine neuron count
    if isinstance(spikes, nap.TsGroup):
        neuron_ids = list(spikes.keys())
    else:  # Single Ts object
        neuron_ids = ["Neuron"]
        spikes = {"Neuron": spikes}
    
    # Create a color map for different odors - we'll allow these to be modified
    unique_odors = list(onsets.keys())
    default_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_odors)))
    odor_colors = {odor: to_hex(default_colors[i]) for i, odor in enumerate(unique_odors)}
    
    # Define a set of available colors for selection
    color_options = {
        "Blue": "#1f77b4",
        "Orange": "#ff7f0e",
        "Green": "#2ca02c",
        "Red": "#d62728",
        "Purple": "#9467bd",
        "Brown": "#8c564b",
        "Pink": "#e377c2",
        "Gray": "#7f7f7f",
        "Olive": "#bcbd22",
        "Cyan": "#17becf",
        "Magenta": "#ff00ff",
        "Lime": "#00ff00",
        "Teal": "#008080",
        "Navy": "#000080",
        "Maroon": "#800000",
        "Yellow": "#ffff00",
        "Black": "#000000"
    }
    
    # Gather all stimulus onset times by type
    all_onsets = []
    odor_types = {"Any": []}  # Dictionary to store indices by odor type
    
    for odor, onset_interval in onsets.items():
        odor_types[odor] = []  # Initialize list for this odor type
        for time in enumerate(onset_interval.start):
            all_onsets.append((time[1], odor))
            odor_types["Any"].append(len(all_onsets) - 1)  # Add to "Any" list
            odor_types[odor].append(len(all_onsets) - 1)   # Add to type-specific list
    
    # Sort by time
    all_onsets.sort()
    
    # Re-index the odor_types dictionary after sorting
    odor_types = {"Any": list(range(len(all_onsets)))}
    for odor in unique_odors:
        odor_types[odor] = [i for i, (_, event_odor) in enumerate(all_onsets) if event_odor == odor]
    
    # Create all widgets first
    # Output area for plot
    plot_output = widgets.Output()
    
    # Current event display
    current_event_info = widgets.HTML(
        value='<b>Current Event:</b> None',
        layout=widgets.Layout(width='250px')
    )
    
    # Create the widgets with expanded widths
    current_event = widgets.IntText(
        value=0,
        description='Event:',
        layout=widgets.Layout(width='120px')
    )
    
    window_input = widgets.FloatText(
        value=window_size,
        description='Window (±s):',
        layout=widgets.Layout(width='160px')
    )
    
    skip_input = widgets.IntText(
        value=events_to_skip,
        description='Skip:',
        layout=widgets.Layout(width='150px')  # Increased width
    )
    
    # Dropdown for event type filtering
    event_type_dropdown = widgets.Dropdown(
        options=["Any"] + unique_odors,
        value="Any",
        description='Type:',
        layout=widgets.Layout(width='180px')
    )
    
    # Navigation buttons
    prev_button = widgets.Button(
        description='◀ Previous', 
        layout=widgets.Layout(width='120px')
    )
    
    next_button = widgets.Button(
        description='Next ▶', 
        layout=widgets.Layout(width='120px')
    )
    
    update_button = widgets.Button(
        description='Update Plot',
        button_style='primary',
        layout=widgets.Layout(width='120px')
    )
    
    # Create color selectors for each odor type
    color_selectors = []
    for odor in unique_odors:
        # Find the default color name from the hex value
        default_color_name = next((name for name, hex_value in color_options.items() 
                                if hex_value == odor_colors[odor]), "Blue")
        
        color_selector = widgets.Dropdown(
            options=list(color_options.keys()),
            value=default_color_name,
            description=odor,
            layout=widgets.Layout(width='180px')
        )
        color_selectors.append(color_selector)
    
    # Function to compute PETH for all neurons around an event
    def compute_peth(center_time, window_size):
        # Set time window
        tmin = center_time - window_size
        tmax = center_time + window_size
        
        # Create bins
        bins = np.arange(tmin, tmax + bin_size, bin_size)
        bin_centers = bins[:-1] + bin_size/2
        
        # Initialize list to store each neuron's smoothed, normalized PETH
        ensemble_peths = []
        
        # Process each neuron
        for neuron_id in neuron_ids:
            # Get spikes for this neuron within the time window
            interval_set = nap.IntervalSet(start=[tmin], end=[tmax])
            if isinstance(spikes[neuron_id], nap.Ts) or isinstance(spikes[neuron_id], nap.Tsd):
                neuron_spikes = spikes[neuron_id].restrict(interval_set)
                spike_times = neuron_spikes.index
            else:
                # Filter manually if not a Ts/Tsd object
                spike_times = [t for t in spikes[neuron_id].index if tmin <= t <= tmax]
            
            # Count spikes in each bin
            hist, _ = np.histogram(spike_times, bins=bins)
            
            # Convert to firing rate
            firing_rate = hist / bin_size
            
            # Smooth the firing rate using our utility function
            smoothed_fr = smooth_peth(firing_rate, bin_size)
            
            ensemble_peths.append(smoothed_fr)
        
        # Stack and compute ensemble average (ignoring NaNs)
        if ensemble_peths:
            ensemble_peth = np.vstack(ensemble_peths)
            average_ensemble = np.nanmean(ensemble_peth, axis=0)
            
            # If all values were NaN, use the original histogram instead
            if np.all(np.isnan(average_ensemble)):
                # Compute a simple histogram as fallback
                histogram = np.zeros(len(bins) - 1)
                for neuron_id in neuron_ids:
                    interval_set = nap.IntervalSet(start=[tmin], end=[tmax])
                    if isinstance(spikes[neuron_id], nap.Ts) or isinstance(spikes[neuron_id], nap.Tsd):
                        neuron_spikes = spikes[neuron_id].restrict(interval_set)
                        spike_times = neuron_spikes.index
                    else:
                        spike_times = [t for t in spikes[neuron_id].index if tmin <= t <= tmax]
                    
                    hist, _ = np.histogram(spike_times, bins=bins)
                    histogram += hist
                
                # Normalize by number of neurons and bin size to get average firing rate
                histogram = histogram / (len(neuron_ids) * bin_size)
                return bin_centers, histogram
            
            return bin_centers, average_ensemble
        else:
            # If no valid PETHs, return zeros
            return bin_centers, np.zeros(len(bin_centers))
    
    # Function to generate the plot
    def create_plot(center_time, window_size=2.5):
        # Set time window
        tmin = center_time - window_size
        tmax = center_time + window_size
        
        try:
            # Create a new figure with GridSpec for raster and histogram
            plt.ioff()  # Turn interactive mode off
            fig = plt.figure(figsize=figsize)
            
            # Calculate the height ratio between histogram and raster
            gs = GridSpec(2, 1, height_ratios=[hist_height_ratio, 1-hist_height_ratio])
            
            # Create two subplots
            ax_hist = fig.add_subplot(gs[0])  # Top subplot for histogram
            ax_raster = fig.add_subplot(gs[1], sharex=ax_hist)  # Bottom subplot for raster
            
            # Find all events visible in this window
            visible_onsets = [(t, odor) for t, odor in all_onsets if tmin <= t <= tmax]
            
            # Row height adjustment to prevent overlap
            row_height = 0.9
            
            # Add shading and lines for each visible event in both plots
            for onset_time, odor in visible_onsets:
                offset_time = onset_time + 2.0  # 2 seconds after onset
                color = odor_colors[odor]
                
                # Add transparent shading between onset and offset
                if tmin <= offset_time <= tmax:  # Only if offset is within view
                    ax_raster.axvspan(onset_time, offset_time, ymin=0, ymax=1,
                               color=color, alpha=0.15)
                    ax_hist.axvspan(onset_time, offset_time, ymin=0, ymax=1,
                               color=color, alpha=0.15)
                
                # Plot onset line (solid)
                ax_raster.axvline(x=onset_time, ymin=0, ymax=1, 
                           color=color, linewidth=1.5, alpha=0.8)
                ax_hist.axvline(x=onset_time, ymin=0, ymax=1, 
                           color=color, linewidth=1.5, alpha=0.8)
                
                # Plot offset line (dotted)
                if tmin <= offset_time <= tmax:  # Only if offset is within view
                    ax_raster.axvline(x=offset_time, ymin=0, ymax=1, 
                               color=color, linewidth=1.2, alpha=0.6,
                               linestyle='--')
                    ax_hist.axvline(x=offset_time, ymin=0, ymax=1, 
                               color=color, linewidth=1.2, alpha=0.6,
                               linestyle='--')
                
            # Plot spikes only for a subset of neurons if there are many
            max_neurons_to_plot = 200  # Limit to improve performance
            plotted_neurons = neuron_ids[:max_neurons_to_plot] if len(neuron_ids) > max_neurons_to_plot else neuron_ids
            
            # Plot each neuron on its own row in raster plot
            for i, neuron_id in enumerate(plotted_neurons):
                # Get spikes for this neuron within the time window
                spike_times = []
                try:
                    if isinstance(spikes[neuron_id], nap.Ts) or isinstance(spikes[neuron_id], nap.Tsd):
                        # Create an interval set for the time window
                        interval_set = nap.IntervalSet(start=[tmin], end=[tmax])
                        
                        # Use the interval set to restrict the spikes
                        neuron_spikes = spikes[neuron_id].restrict(interval_set)
                        spike_times = neuron_spikes.index
                    else:
                        # If it's not a Ts/Tsd object, filter manually
                        spike_times = [t for t in spikes[neuron_id].index if tmin <= t <= tmax]
                        
                    # Plot spikes as vertical lines with adjusted height to prevent overlap
                    if len(spike_times) > 0:
                        y_values = np.ones_like(spike_times) * i
                        ax_raster.vlines(spike_times, y_values - row_height/2, y_values + row_height/2, 
                                  colors='black', linewidth=1.2)
                except Exception as e:
                    pass
            
            # Find the center event (closest to the center time)
            center_event = None
            min_dist = float('inf')
            for t, odor in all_onsets:
                dist = abs(t - center_time)
                if dist < min_dist:
                    min_dist = dist
                    center_event = (t, odor)
            
            # Compute and plot the histogram for the visible window
            try:
                bin_centers, histogram = compute_peth(center_time, window_size)
                
                # Always plot the PETH line in black
                ax_hist.plot(bin_centers, histogram, linewidth=2, color='black')
                
                # If we have a center event, update the event info display
                if center_event:
                    color = odor_colors[center_event[1]]
                    
                    # Highlight in raster plot - solid line at event onset
                    ax_raster.axvline(x=center_event[0], ymin=0, ymax=1, 
                               color=color, linewidth=3, alpha=1.0)
                    
                    # Highlight in histogram - solid line at event onset
                    ax_hist.axvline(x=center_event[0], ymin=0, ymax=1, 
                               color=color, linewidth=3, alpha=1.0)
                    
                    # Update the current event info HTML
                    current_event_info.value = f'<b>Current Event:</b> {center_event[1]} at {center_event[0]:.2f}s'
            except Exception as e:
                pass
            
            # Set labels and limits
            ax_raster.set_xlabel('Time (s)', fontsize=font_size)
            ax_raster.set_ylabel('Neuron', fontsize=font_size)
            ax_raster.set_xlim(tmin, tmax)
            ax_raster.set_ylim(-0.5, min(len(neuron_ids), max_neurons_to_plot) + 1.0)
            
            # Set histogram labels
            ax_hist.set_ylabel('Normalized Rate', fontsize=font_size)
            ax_hist.tick_params(axis='both', which='major', labelsize=font_size-2)
            ax_raster.tick_params(axis='both', which='major', labelsize=font_size-2)
            
            # Hide x-axis labels on top plot
            plt.setp(ax_hist.get_xticklabels(), visible=False)
            
            # Set y-ticks with neuron count markers
            if len(neuron_ids) > 50:
                step = 20
            elif len(neuron_ids) > 20:
                step = 10
            else:
                step = 5
                
            ytick_positions = list(range(0, min(len(neuron_ids), max_neurons_to_plot), step))
            if ytick_positions and ytick_positions[-1] != min(len(neuron_ids), max_neurons_to_plot) - 1:
                ytick_positions.append(min(len(neuron_ids), max_neurons_to_plot) - 1)  # Add the last neuron
                
            ax_raster.set_yticks(ytick_positions)
            ax_raster.set_yticklabels([str(pos) for pos in ytick_positions])
            
            # Create simplified legend for odors (onset only)
            handles = []
            labels = []
            for odor in unique_odors:
                line = plt.Line2D([0], [0], color=odor_colors[odor], lw=2, label=odor)
                handles.append(line)
                labels.append(odor)
            ax_raster.legend(handles=handles, labels=labels, loc='upper right', fontsize=font_size)
            
            # Add title with time window info
            fig.suptitle(f'Raster Plot - Time Window: {tmin:.2f}s to {tmax:.2f}s', 
                        fontsize=font_size+2)
            
            # Adjust layout
            plt.tight_layout()
            fig.subplots_adjust(top=0.92, hspace=0.05)  # Reduce space between subplots
            
            return fig
            
        except Exception as e:
            # Return a simple error figure instead
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Error creating plot: {e}", 
                    ha='center', va='center', fontsize=12)
            return fig
    
    # Function to update the display with the current event
    def update_display(current_event_widget, plot_output_widget, all_onsets_list, window_size_widget):
        with plot_output_widget:
            plot_output_widget.clear_output(wait=True)
            
            if len(all_onsets_list) > 0 and 0 <= current_event_widget.value < len(all_onsets_list):
                selected_time = all_onsets_list[current_event_widget.value][0]
                
                try:
                    fig = create_plot(selected_time, window_size_widget.value)
                    display(fig)
                    plt.close(fig)
                except Exception as e:
                    pass
    
    # Function to handle color changes
    def on_color_change(change):
        odor = change['owner'].description
        odor_colors[odor] = color_options[change['new']]
        update_display(current_event, plot_output, all_onsets, window_input)
    
    # Connect color selector change events
    for selector in color_selectors:
        selector.observe(on_color_change, names='value')
    
    # Function to navigate to events by type
    def navigate_by_type(direction, odor_filter, skip_count, current_event_widget, all_onsets_list):
        current_type_indices = odor_types[odor_filter]
        if not current_type_indices:
            return  # No events of this type
            
        # Find the current position in the filtered list
        if current_event_widget.value in current_type_indices:
            current_pos = current_type_indices.index(current_event_widget.value)
        else:
            # Find the closest position
            if direction > 0:
                # Find the first event of this type after the current event
                filtered_future_indices = [i for i in current_type_indices if i > current_event_widget.value]
                if filtered_future_indices:
                    current_event_widget.value = filtered_future_indices[0]
                else:
                    current_event_widget.value = current_type_indices[0]  # Wrap around to first
                return
            else:
                # Find the first event of this type before the current event
                filtered_past_indices = [i for i in current_type_indices if i < current_event_widget.value]
                if filtered_past_indices:
                    current_event_widget.value = filtered_past_indices[-1]
                else:
                    current_event_widget.value = current_type_indices[-1]  # Wrap around to last
                return
        
        # Calculate the new position with skipping
        new_pos = current_pos + (direction * skip_count)
        if direction > 0:
            # Forward navigation
            if new_pos >= len(current_type_indices):
                new_pos = 0  # Wrap around to the beginning
        else:
            # Backward navigation
            if new_pos < 0:
                new_pos = len(current_type_indices) - 1  # Wrap around to the end
                
        # Set the new event index
        current_event_widget.value = current_type_indices[new_pos]
    
    # Event handlers for type-specific navigation
    def on_prev_clicked(b):
        navigate_by_type(-1, event_type_dropdown.value, skip_input.value, current_event, all_onsets)
        update_display(current_event, plot_output, all_onsets, window_input)
        
    def on_next_clicked(b):
        navigate_by_type(1, event_type_dropdown.value, skip_input.value, current_event, all_onsets)
        update_display(current_event, plot_output, all_onsets, window_input)
    
    def manual_update(b):
        update_display(current_event, plot_output, all_onsets, window_input)
    
    # Handle event type dropdown changes
    def on_type_change(change):
        # Reset to first event of selected type when type changes
        if change['new'] != change['old'] and odor_types[change['new']]:
            current_event.value = odor_types[change['new']][0]
            update_display(current_event, plot_output, all_onsets, window_input)
    
    # Connect event handlers
    prev_button.on_click(on_prev_clicked)
    next_button.on_click(on_next_clicked)
    update_button.on_click(manual_update)
    event_type_dropdown.observe(on_type_change, names='value')
    
    # Create color widget rows
    color_widget_rows = []
    colors_per_row = 4
    for i in range(0, len(color_selectors), colors_per_row):
        row = widgets.HBox(color_selectors[i:i+colors_per_row])
        color_widget_rows.append(row)
    
    color_widget_area = widgets.VBox(color_widget_rows)
    
    # Create a flexible layout for all controls in one row
    # First, create individual sections
    navigation_controls = widgets.VBox([
        widgets.HTML("<b>Navigation Controls:</b>"),
        widgets.HBox([current_event, skip_input], layout=widgets.Layout(width='auto')),
        widgets.HBox([event_type_dropdown, window_input], layout=widgets.Layout(width='auto')),
        widgets.HBox([prev_button, next_button, update_button], layout=widgets.Layout(width='auto')),
        current_event_info
    ])
    
    color_controls = widgets.VBox([
        widgets.HTML("<b>Event Color Selection:</b>"),
        color_widget_area
    ])
    
    # Put all control sections in a single flexbox row
    controls_row = widgets.HBox([
        navigation_controls,
        color_controls
    ], layout=widgets.Layout(
        flex_flow='row wrap',
        justify_content='space-around',
        width='100%',
        margin='0px 0px 10px 0px'
    ))
    
    # Display the complete UI with controls in one flexible row
    display(widgets.VBox([
        controls_row,
        plot_output
    ]))
    
    # Initial update_display call
    try:
        update_display(current_event, plot_output, all_onsets, window_input)
    except Exception as e:
        # Try to display something helpful
        with plot_output:
            plot_output.clear_output(wait=True)
            plt.figure()
            plt.title("Error occurred during initial plot")
            plt.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
            display(plt.gcf())
            plt.close()
    
    # Return the widgets for possible external use
    return {
        'plot_output': plot_output,
        'current_event': current_event,
        'window_input': window_input,
        'skip_input': skip_input,
        'event_type_dropdown': event_type_dropdown,
        'prev_button': prev_button,
        'next_button': next_button,
        'update_button': update_button,
        'color_selectors': color_selectors
    }