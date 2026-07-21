#!/usr/bin/env python3
"""
Neuropixels LFP Extraction and Kilosort4 Processing

This script performs automatic bad channel detection, LFP extraction, and runs Kilosort4
on Neuropixels data. It processes each probe in the dataset, extracts LFP data, and 
runs spike sorting.

Usage:
    python extract_lfp_and_run_ks4_quadBase.py --file_params path/to/params.txt
    python extract_lfp_and_run_ks4_quadBase.py --batch_params path/to/batch_list.txt
"""

import os
import sys
import json
import logging
import numpy as np
import scipy.io as scio
import argparse
import shutil
from datetime import datetime
import spikeinterface.full as si
import hdf5storage

# Default parameters
DEFAULT_PARAMS = {
    'CATGT_OUTPUT_DIR': None,  # Required
    'OUTPUT_DIR': None,        # Required
    'SESSION_ID': None,        # Required
    'FAST_STORAGE_DIR': None,  # Required
    'PROBE_INDICES': [0],
    'EXTRACT_LFP_FOR': None,    # Required
    'RUN_KILOSORT_FOR': None,   # Required
    'OVERRIDE_BAD_CHANNELS': False,
    'MANUAL_BAD_CHANNELS_FILE': None,
    'MAX_BAD_CHANNELS': 25,
    'OVERRIDE_LFP_CHANNELS': False,
    'MANUAL_LFP_CHANNELS_FILE': None,
    'LFP_SAMPLE_RATE': 2500,
    'LFP_HIGH_PASS': 1,
    'LFP_LOW_PASS': 400,
    'LFP_SPACING': 100,
    'INTER_SHANK_SPACE': 250,
    'AP_HIGH_PASS': 400.0,
    'KS_PARAMS': {
        'delete_recording_dat': True,
        'use_binary_file': True
    },
    'LFP_NUM_JOBS': 6,
    'SI_NUM_JOBS': 8,
    'SI_CHUNK_DURATION': '1s',
    'LFP_CHUNK_DURATION': '30s'
}

# Required parameters
REQUIRED_PARAMS = ['CATGT_OUTPUT_DIR', 'OUTPUT_DIR', 'SESSION_ID', 'FAST_STORAGE_DIR']

def parse_params_file(params_file):
    """Parse parameters from a file"""
    try:
        params = DEFAULT_PARAMS.copy()
        
        with open(params_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            try:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if key in params:
                    # Handle different parameter types
                    if key in ['PROBE_INDICES', 'EXTRACT_LFP_FOR', 'RUN_KILOSORT_FOR']:
                        # Parse list of integers
                        if value.lower() == 'none':
                            params[key] = None  # Keep as None to use default
                        elif not value.strip():
                            params[key] = []  # Empty list if value is empty
                        else:
                            params[key] = [int(x.strip()) for x in value.split(',')]
                    elif key in ['OVERRIDE_BAD_CHANNELS', 'OVERRIDE_LFP_CHANNELS']:
                        # Parse boolean
                        params[key] = value.lower() == 'true'
                    elif key in ['MAX_BAD_CHANNELS', 'LFP_SAMPLE_RATE', 'LFP_HIGH_PASS', 
                                'LFP_LOW_PASS', 'LFP_SPACING', 'INTER_SHANK_SPACE', 
                                'AP_HIGH_PASS', 'LFP_NUM_JOBS', 'SI_NUM_JOBS']:
                        # Parse numeric
                        params[key] = float(value)
                        # Convert to int if it's supposed to be an integer
                        if key in ['MAX_BAD_CHANNELS', 'LFP_SAMPLE_RATE', 'LFP_SPACING', 
                                  'INTER_SHANK_SPACE', 'LFP_NUM_JOBS', 'SI_NUM_JOBS']:
                            params[key] = int(params[key])
                    elif key == 'KS_PARAMS':
                        # Parse JSON
                        params[key] = json.loads(value)
                    else:
                        # String parameters
                        params[key] = value
                else:
                    print(f"Warning: Unknown parameter '{key}' in params file")
            except Exception as e:
                print(f"Error parsing line '{line}': {e}")
        
        # Check required parameters
        missing_params = [param for param in REQUIRED_PARAMS if params[param] is None]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")
        
        # Set default values for EXTRACT_LFP_FOR and RUN_KILOSORT_FOR if not specified
        # Only set default if parameter is None, not if it's an empty list
        if params['EXTRACT_LFP_FOR'] is None:
            params['EXTRACT_LFP_FOR'] = params['PROBE_INDICES']

        if params['RUN_KILOSORT_FOR'] is None:
            params['RUN_KILOSORT_FOR'] = params['PROBE_INDICES']
        
        return params
        
    except Exception as e:
        print(f"Error parsing parameters file '{params_file}': {e}")
        sys.exit(1)

# Set up logging
def setup_logging(output_dir, session_id):
    log_file = os.path.join(output_dir, f"{session_id}_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Reset the root logger and remove all handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def detect_bad_channels(raw_rec, probe_idx, logger, params):
    """
    Detect bad channels for a specific probe
    
    Args:
        raw_rec: SpikeInterface recording object
        probe_idx: Index of the probe
        logger: Logger object
        params: Dictionary of parameters
    
    Returns:
        tuple: (bad_channel_ids, False if successful or number of bad channels below threshold)
    """
    logger.info(f"Detecting bad channels for probe {probe_idx}...")
    
    try:
        # First want to split by shank, which is essential if using the default method (coherence+psd)
        split_rec = raw_rec.split_by("group")
        # Run the bad channel detection algorithm shank-wise - it will automatically high-pass at 300 Hz
        bad_channel_ids = []
        channel_labels = []
        for shank, sub_rec in split_rec.items():
            logger.info(f"Detecting bad channels for shank {shank} ...")
            ids, labels = si.detect_bad_channels(sub_rec)
            logger.info(f"Detected {len(ids)} bad channels for shank {shank}: {ids}")
            bad_channel_ids.extend(ids)
            channel_labels.extend(labels)
        
        # Convert to list if needed
        if isinstance(bad_channel_ids, np.ndarray):
            bad_channel_ids = bad_channel_ids.tolist()
        
        # Log the detected bad channels
        logger.info(f"Detected {len(bad_channel_ids)} bad channels for probe {probe_idx}")
        for bad_channel_id in bad_channel_ids:
            try:
                index = int(bad_channel_id.split('AP')[1])
                logger.info(f"  {bad_channel_id}, {channel_labels[index]}")
            except (IndexError, ValueError):
                logger.info(f"  {bad_channel_id}")
        
        # Check if manual curation is needed
        needs_manual_curation = len(bad_channel_ids) > params['MAX_BAD_CHANNELS']
        if needs_manual_curation:
            logger.warning(f"Number of bad channels ({len(bad_channel_ids)}) exceeds threshold ({params['MAX_BAD_CHANNELS']})")
            logger.warning("Manual curation is recommended")
        
        return bad_channel_ids, needs_manual_curation
        
    except Exception as e:
        logger.error(f"Exception detecting bad channels: {e}")
        return [], True  # Assume manual curation is needed on error

def save_bad_channels(bad_channel_ids, catgt_output_dir, probe_idx, logger):
    """
    Save bad channel IDs to a JSON file in the CatGT output directory
    
    Args:
        bad_channel_ids: List of bad channel IDs
        catgt_output_dir: Path to the CatGT output directory
        probe_idx: Index of the probe
        logger: Logger object
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create the output file path in the CatGT directory
        output_file = os.path.join(catgt_output_dir, f"imec{probe_idx}_bad_channels.json")
        
        # Save the bad channels to a JSON file
        with open(output_file, 'w') as f:
            json.dump({'bad_channel_ids': bad_channel_ids}, f, indent=4)
        
        logger.info(f"Saved bad channel IDs to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Exception saving bad channel IDs: {e}")
        return False

def extract_lfp(raw_rec, bad_channel_ids, probe_idx, logger, params):
    """
    Extract LFP data from raw recording
    
    Args:
        raw_rec: SpikeInterface raw recording object
        bad_channel_ids: List of bad channel IDs to exclude
        probe_idx: Index of the probe
        logger: Logger object
        params: Dictionary of parameters
    
    Returns:
        tuple: (final_channels, final_lfp, final_tvec, final_fs, final_depths, shank_ids)
    """
    try:
        logger.info(f"Extracting LFP data for probe {probe_idx}...")
        
        # Remove bad channels
        lfp_rec = raw_rec.remove_channels(bad_channel_ids)

        # Change the global job kwargs to increase chunk duration - doesn't change the output, just
        ### saves us from seeing an ugly warning
        si.set_global_job_kwargs(chunk_duration=params['LFP_CHUNK_DURATION'])
        logger.info(f"Changed global job kwargs for lfp filtering - increased chunk duration to {params['LFP_CHUNK_DURATION']}")
        logger.info(f"Global job kwargs are now: {si.get_global_job_kwargs()}")

        # bandpass filter for lfp extraction
        logger.info(f"Applying bandpass filter ({params['LFP_HIGH_PASS']}-{params['LFP_LOW_PASS']} Hz)...")
        margin_ms_lfp = params['LFP_HIGH_PASS']*5*1000 # high-pass freq threshold * 5, expressed in ms (e.g. 1Hz --> 5000 ms)
        lfp_rec = si.bandpass_filter(lfp_rec, 
                                     freq_min=params['LFP_HIGH_PASS'], 
                                     freq_max=params['LFP_LOW_PASS'],
                                     margin_ms = margin_ms_lfp,
                                     ignore_low_freq_error=True)
        
        # Resample for LFP extraction
        logger.info(f"Resampling to {params['LFP_SAMPLE_RATE']} Hz...")
        lfp_rec = si.resample(lfp_rec, 
                              resample_rate = params['LFP_SAMPLE_RATE'],
                              margin_ms = margin_ms_lfp)
        
        if params['OVERRIDE_LFP_CHANNELS']:
            # Use manual LFP channel selection
            logger.info(f"Using manual LFP channel selection from: {params['MANUAL_LFP_CHANNELS_FILE']}")
            
            try:
                with open(params['MANUAL_LFP_CHANNELS_FILE'], 'r') as f:
                    manual_data = json.load(f)
                
                # Try different possible keys for LFP channels
                if f'imec{probe_idx}' in manual_data:
                    manual_channels = manual_data[f'imec{probe_idx}']['channel_ids']
                elif f'probe{probe_idx}' in manual_data:
                    manual_channels = manual_data[f'probe{probe_idx}']['channel_ids']
                elif 'channel_ids' in manual_data:
                    manual_channels = manual_data['channel_ids']
                else:
                    logger.error(f"Could not find LFP channel data for probe {probe_idx} in manual file")
                    return None, None, None, None, None, None
                
                # Convert to numpy array if needed
                if not isinstance(manual_channels, np.ndarray):
                    manual_channels = np.array(manual_channels)
                
                # Validate that these channels exist in the recording
                available_channels = set(lfp_rec.channel_ids)
                valid_channels = [ch for ch in manual_channels if ch in available_channels]
                
                if len(valid_channels) < len(manual_channels):
                    logger.warning(f"{len(manual_channels) - len(valid_channels)} manual LFP channels were not found in the recording")
                
                if len(valid_channels) == 0:
                    logger.error("No valid LFP channels found in manual selection")
                    return None, None, None, None, None, None
                
                final_channels = np.array(valid_channels)
                logger.info(f"Selected {len(final_channels)} manual LFP channels")
                
                # Find indices of these channels
                channel_indices = []
                for ch in final_channels:
                    try:
                        idx = np.where(lfp_rec.channel_ids == ch)[0][0]
                        channel_indices.append(idx)
                    except (IndexError, ValueError):
                        logger.warning(f"Channel {ch} not found in recording")
                
                # Get channel locations and calculate shank IDs
                all_locs = lfp_rec.get_channel_locations()
                final_depths = [(all_locs[idx][1] + 175) for idx in channel_indices] # EFBG - adding 175 microns such that the tip of the probe is at 0 microns depth (for NPX2)
                shank_ids = [int(all_locs[idx][0] // params['INTER_SHANK_SPACE']) for idx in channel_indices]
                
            except Exception as e:
                logger.error(f"Error loading manual LFP channels: {e}")
                return None, None, None, None, None, None
                
        else:
            # Automatic channel selection for NPX2
            logger.info(f"Selecting channels with {params['LFP_SPACING']} um spacing...")
            keep_idx = []
            for iShank in range(4):  # 4 shanks for Neuropixels 2.0
                idx_depth = [(idx, x[1]) for idx, x in enumerate(lfp_rec.get_channel_locations()) 
                             if int(x[0] // params['INTER_SHANK_SPACE']) == iShank]
                
                # If no channels for this shank, skip
                if len(idx_depth) == 0:
                    continue
                    
                depths = [x[1] for x in idx_depth]
                idx = [x[0] for x in idx_depth]
                
                # EFBG - new method of getting queried_depths, accounting for recordings where multiple different portions of a single shank are recorded (e.g. dorsal and ventral hc)
                depths_array = np.array(depths)
                sorted_depths = np.sort(depths_array)
                gap_threshold = 100; # in microns - Threshold to identify gaps in the depth distribution (in microns) - theoretically channels should only be 15 microns apart, but made larger in case of bad channels
                depth_diffs = np.diff(sorted_depths)
                gap_indices = np.where(np.abs(depth_diffs) > gap_threshold)[0]
                # If there are gaps, split the depths into segments
                if len(gap_indices) > 0:
                    queried_depths = []
                    start_idx = 0
                    for gap_idx in gap_indices:
                        segment_depths = sorted_depths[start_idx:gap_idx+1]
                        segment_queried = np.arange(min(segment_depths), max(segment_depths), params['LFP_SPACING'])
                        queried_depths.extend(segment_queried)
                        start_idx = gap_idx + 1
                    # Add the last segment
                    if start_idx < len(sorted_depths):
                        segment_depths = sorted_depths[start_idx:]
                        segment_queried = np.arange(min(segment_depths), max(segment_depths), params['LFP_SPACING'])
                        queried_depths.extend(segment_queried)
                else:
                    # No gaps, use all depths
                    queried_depths = np.arange(min(depths), max(depths), params['LFP_SPACING']).tolist()

                
                # Add the max depth if not already there
                if queried_depths[-1] < max(depths) - 50:
                    queried_depths.append(max(depths))
                    
                # Find the closest channels to the requested depths
                for b in queried_depths:
                    min_diff = float('inf')
                    closest_index = None
                    for i, a in enumerate(depths):
                        diff = abs(a - b)
                        if diff < min_diff:
                            min_diff = diff
                            closest_index = i
                        elif diff == min_diff:
                            closest_index = min(closest_index, i)
                    # Only add a channel if not already present - EFBG added
                    if idx[closest_index] not in keep_idx:
                        keep_idx.append(idx[closest_index])
                    
            # Get the selected channels
            final_channels = lfp_rec.channel_ids[np.asarray(keep_idx)]
            logger.info(f"Selected {len(final_channels)} channels for LFP automatically")
            
            # Get channel locations and calculate shank IDs
            all_locs = lfp_rec.get_channel_locations()
            final_depths = [(all_locs[idx][1] + 175) for idx in keep_idx] # EFBG - adding 175 microns such that the tip of the probe is at 0 microns depth (for NPX2)
            shank_ids = [int(all_locs[idx][0] // params['INTER_SHANK_SPACE']) for idx in keep_idx]
        
        # Create a temporary folder for faster extraction
        temp_lfp_folder = os.path.join(params['FAST_STORAGE_DIR'], f"{params['SESSION_ID']}_temp_lfp_imec{probe_idx}")
        os.makedirs(temp_lfp_folder, exist_ok=True)
        
        # Extract LFP data using a temporary binary file for speed
        logger.info(f"Creating temporary binary file for faster LFP extraction...")
        job_kwargs = dict(n_jobs=params['LFP_NUM_JOBS'], chunk_duration=params['LFP_CHUNK_DURATION'], progress_bar=True)
        temp_lfp_subchannel = lfp_rec.select_channels(channel_ids=final_channels)
        temp_lfp = temp_lfp_subchannel.save(
            folder=temp_lfp_folder, 
            format="binary", 
            return_scaled=True, 
            cast_unsigned=True,
            overwrite=True,
            **job_kwargs
        ) # I think return_scaled and cast_unsinged are both depreciated, but they don't throw an error
        
        # Get traces from the temporary recording
        logger.info("Extracting LFP traces from temporary file...")
        final_lfp = temp_lfp.get_traces(channel_ids=final_channels, return_in_uV=True)
        
        # Get other metadata
        final_tvec = lfp_rec.get_times()
        final_tvec = final_tvec - final_tvec[0] # because spikeinterface adds the "First Sample's time"
        final_fs = lfp_rec.get_sampling_frequency()
        
        # Save LFP times to the CatGT output folder in both formats
        lfp_times_npy_path = os.path.join(params['CATGT_OUTPUT_DIR'], f"imec{probe_idx}_lfp_times.npy")

        # Save as .npy file
        np.save(lfp_times_npy_path, final_tvec)
        logger.info(f"Saved LFP times as .npy to {lfp_times_npy_path}")
        
        # Clean up the temporary folder 
        ### This still doesn't work, I think if we wanted it to it would have to go after 
        ### we save the lfp data to the .mat file
        logger.info("Cleaning up temporary LFP extraction files...")
        try:
            shutil.rmtree(temp_lfp_folder)
        except Exception as e:
            logger.warning(f"Could not remove temporary LFP folder {temp_lfp_folder}: {e}")
        
        return final_channels, final_lfp, final_tvec, final_fs, final_depths, shank_ids
        
    except Exception as e:
        logger.error(f"Exception extracting LFP data: {e}")
        return None, None, None, None, None, None

def run_kilosort(raw_rec, bad_channel_ids, probe_idx, logger, params):
    """
    Run Kilosort4 on the recording
    
    Args:
        raw_rec: SpikeInterface raw recording object
        bad_channel_ids: List of bad channel IDs to exclude
        probe_idx: Index of the probe
        logger: Logger object
        params: Dictionary of parameters
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Running Kilosort4 for probe {probe_idx}...")
        
        # make sure the chunk duration is back to 1s for everything kilosort
        si.set_global_job_kwargs(chunk_duration = '1s')

        # Filter the recording
        logger.info(f"Applying high-pass filter at {params['AP_HIGH_PASS']} Hz...")
        filt_rec = si.highpass_filter(raw_rec, freq_min=params['AP_HIGH_PASS'])
        
        # Remove bad channels
        logger.info(f"Removing {len(bad_channel_ids)} bad channels...")
        filt_rec = filt_rec.remove_channels(bad_channel_ids)
        
        # Define output folders and make session-specific names
        session_specific_folder = f"{params['SESSION_ID']}_imec{probe_idx}"
        
        # SI preprocess folder (on fast storage)
        si_working_folder = os.path.join(params['FAST_STORAGE_DIR'], f"{session_specific_folder}_si_preprocess")
        
        # KS4 output folder (on fast storage)
        ks_working_folder = os.path.join(params['FAST_STORAGE_DIR'], session_specific_folder)
        
        # Create folders if they don't exist
        os.makedirs(si_working_folder, exist_ok=True)
        os.makedirs(ks_working_folder, exist_ok=True)
        
        # Save the preprocessed recording
        logger.info(f"Saving filtered recording to {si_working_folder}...")
        job_kwargs = dict(n_jobs=params['SI_NUM_JOBS'], chunk_duration=params['SI_CHUNK_DURATION'], progress_bar=True)
        saved_rec = filt_rec.save(folder=si_working_folder, format='binary', overwrite=True, **job_kwargs)
        
        # Run Kilosort4
        logger.info(f"Running Kilosort4 with output to {ks_working_folder}...")
        si.run_sorter('kilosort4', saved_rec, folder=ks_working_folder, 
                      verbose=True, remove_existing_folder=True, **params['KS_PARAMS'])
        
        logger.info(f"Kilosort4 completed for probe {probe_idx}")
        return True
        
    except Exception as e:
        logger.error(f"Exception running Kilosort: {e}")
        return False

def process_probe(probe_idx, logger, params):
    """
    Process a single probe: detect bad channels, extract LFP, run Kilosort
    
    Args:
        probe_idx: Index of the probe to process
        logger: Logger object
        params: Dictionary of parameters
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Processing probe {probe_idx}...")
    
    try:
        # Read the raw data
        logger.info(f"Reading raw data from {params['CATGT_OUTPUT_DIR']} for stream imec{probe_idx}.ap...")
        raw_rec = si.read_spikeglx(params['CATGT_OUTPUT_DIR'], stream_name=f"imec{probe_idx}.ap")
        
        # Determine bad channels
        if params['OVERRIDE_BAD_CHANNELS']:
            # Load bad channels from manual file
            logger.info(f"Using manual bad channels from: {params['MANUAL_BAD_CHANNELS_FILE']}")
            
            try:
                with open(params['MANUAL_BAD_CHANNELS_FILE'], 'r') as f:
                    manual_data = json.load(f)
                
                # Try different possible keys for bad channels
                if f'imec{probe_idx}' in manual_data:
                    bad_channel_ids = manual_data[f'imec{probe_idx}']['bad_channel_ids']
                elif f'probe{probe_idx}' in manual_data:
                    bad_channel_ids = manual_data[f'probe{probe_idx}']['bad_channel_ids']
                elif 'bad_channel_ids' in manual_data:
                    bad_channel_ids = manual_data['bad_channel_ids']
                else:
                    logger.error(f"Could not find bad channel data for probe {probe_idx} in manual file")
                    return False
                
                needs_manual_curation = False
                
            except Exception as e:
                logger.error(f"Error loading manual bad channels: {e}")
                return False
        else:
            # Detect bad channels automatically
            bad_channel_ids, needs_manual_curation = detect_bad_channels(raw_rec, probe_idx, logger, params)
        
        # Save bad channels to JSON file in the CatGT output directory
        save_bad_channels(bad_channel_ids, params['CATGT_OUTPUT_DIR'], probe_idx, logger)
        
        # Check if manual curation is needed
        if needs_manual_curation:
            logger.warning(f"Skipping processing for probe {probe_idx} due to high bad channel count or detection error")
            return False
        
        # Extract LFP data if this probe is in the list
        if probe_idx in params['EXTRACT_LFP_FOR']:
            # Extract LFP data
            final_channels, final_lfp, final_tvec, final_fs, final_depths, shank_ids = extract_lfp(
                raw_rec, bad_channel_ids, probe_idx, logger, params)
            
            if final_lfp is None:
                logger.error(f"LFP extraction failed for probe {probe_idx}")
                return False
            
            # Save LFP data to MATLAB file
            lfp_mat_fname = os.path.join(params['OUTPUT_DIR'], f"imec{probe_idx}_clean_lfp.mat")
            logger.info(f"Saving LFP data to {lfp_mat_fname}")
            
            hdf5storage.savemat(
                lfp_mat_fname,
                {
                    'depths': np.asarray(final_depths, dtype=np.float64),
                    'channel_ids': np.asarray(final_channels.tolist(), dtype='U15').reshape(-1,1), # added reshape after another formatting failure
                    'lfp_traces': np.squeeze(np.asarray(final_lfp, dtype=np.float32)), # I changed this line to match the style im using, but it shouldn't make a functional difference
                    'lfp_tvec': final_tvec,
                    'lfp_fs': final_fs,
                    'shank_ids': np.asarray(shank_ids, dtype=np.int32)
                },
                format='7.3',
                oned_as='row'
            )
            
            logger.info(f"LFP data extraction completed for probe {probe_idx}")
        else:
            logger.info(f"Skipping LFP extraction for probe {probe_idx} (not in EXTRACT_LFP_FOR list)")
        
        # Run Kilosort if this probe is in the list
        if probe_idx in params['RUN_KILOSORT_FOR']:
            # Run Kilosort
            success = run_kilosort(raw_rec, bad_channel_ids, probe_idx, logger, params)
            if not success:
                logger.error(f"Kilosort processing failed for probe {probe_idx}")
                return False
                
            logger.info(f"Kilosort processing completed for probe {probe_idx}")
        else:
            logger.info(f"Skipping Kilosort for probe {probe_idx} (not in RUN_KILOSORT_FOR list)")
        
        logger.info(f"Successfully processed probe {probe_idx}")
        return True
        
    except Exception as e:
        logger.error(f"Exception processing probe {probe_idx}: {e}")
        return False

def process_session(params_file):
    """Process a single session with the given parameters"""
    # Parse parameters
    params = parse_params_file(params_file)
    
    # Setup logging
    logger = setup_logging(params['OUTPUT_DIR'], params['SESSION_ID'])
    
    logger.info(f"Processing session: {params['SESSION_ID']}")
    logger.info(f"Parameters file: {params_file}")
    logger.info(f"CatGT output directory: {params['CATGT_OUTPUT_DIR']}")
    logger.info(f"Output directory: {params['OUTPUT_DIR']}")
    logger.info(f"Fast storage for KS4: {params['FAST_STORAGE_DIR']}")
    logger.info(f"Processing probes: {params['PROBE_INDICES']}")
    logger.info(f"Extracting LFP for probes: {params['EXTRACT_LFP_FOR']}")
    logger.info(f"Running Kilosort for probes: {params['RUN_KILOSORT_FOR']}")
    
    # Make sure output directories exist
    os.makedirs(params['OUTPUT_DIR'], exist_ok=True)
    os.makedirs(params['FAST_STORAGE_DIR'], exist_ok=True)
    
    # Process each probe in the list
    all_success = True
    for probe_idx in params['PROBE_INDICES']:
        success = process_probe(probe_idx, logger, params)
        if not success:
            all_success = False
            logger.warning(f"Processing failed or was skipped for probe {probe_idx}")
        else:
            logger.info(f"Processing completed successfully for probe {probe_idx}")
    
    if all_success:
        logger.info(f"All probes processed successfully for session {params['SESSION_ID']}")
        print(f"\nAll probes processed successfully for session {params['SESSION_ID']}!")
    else:
        logger.warning(f"Processing completed with warnings or errors for some probes in session {params['SESSION_ID']}")
        print(f"\nProcessing completed with warnings or errors for some probes in session {params['SESSION_ID']}. Check the log file for details.")
    
    return all_success

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process Neuropixels data: extract LFP and run Kilosort')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file_params', type=str, help='Path to parameters file for a single session')
    group.add_argument('--batch_params', type=str, help='Path to a file listing multiple parameter files')
    args = parser.parse_args()
    
    if args.file_params:
        # Process a single session
        process_session(args.file_params)
    else:
        # Process multiple sessions from a batch file
        with open(args.batch_params, 'r') as f:
            params_files = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        
        print(f"Processing {len(params_files)} sessions from batch file: {args.batch_params}")
        
        success_count = 0
        for i, params_file in enumerate(params_files):
            print(f"\nProcessing session {i+1}/{len(params_files)}: {params_file}")
            if process_session(params_file):
                success_count += 1
        
        print(f"\nBatch processing completed: {success_count}/{len(params_files)} sessions processed successfully")

if __name__ == "__main__":
    main()
