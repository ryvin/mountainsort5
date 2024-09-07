import os
import time
import uuid
import shutil
from pathlib import Path
import numpy as np
import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw
import mountainsort5 as ms5
from mountainsort5.util import create_cached_recording
import matplotlib.pyplot as plt

def load_npy_file(file_path):
    data = np.load(file_path)
    if data.ndim == 1:
        data = data.reshape(-1, 1)  # Reshape to 2D if it's 1D
    elif data.ndim == 2 and data.shape[0] == 1:
        data = data.T  # Transpose if it's 1 x N instead of N x 1
    return data.astype(np.float32)  # Changed to float32 to avoid potential int16 overflow

def remove_duplicate_spikes(recording, tolerance=0.9999):
    traces = recording.get_traces()
    unique_traces, unique_indices = np.unique(traces, axis=0, return_index=True)
    if len(unique_traces) < len(traces):
        print(f"Removed {len(traces) - len(unique_traces)} duplicate spikes")
        # Sort the unique indices to maintain the original order
        unique_indices.sort()
        # Create a new recording with only the unique frames
        return recording.frame_slice(start_frame=unique_indices[0], end_frame=unique_indices[-1])
    return recording

def create_unique_temp_dir():
    temp_dir = Path(os.environ.get('TEMP', '/tmp')) / f'mountainsort5_{uuid.uuid4().hex}'
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir

def load_ground_truth(file_path, samplerate):
    # Load the ground truth data
    ground_truth_data = np.load(file_path)
    
    # Extract spike times and labels
    spike_times = ground_truth_data[:, 1].astype(int)
    spike_labels = ground_truth_data[:, 2].astype(int)
    
    # Create a dictionary of spike trains
    spike_trains = {}
    for label in np.unique(spike_labels):
        spike_trains[label] = spike_times[spike_labels == label]
    
    # Create a SortingExtractor
    sorting_true = se.NumpySorting.from_times_labels(spike_times=spike_trains, sampling_frequency=samplerate)
    
    return sorting_true

def main():
    # Data file information
    data_info = {
        "params": {
            "samplerate": 24000,
            "scale_factor": 0.01,
            "spike_sign": -1
        }
    }

    # Load the raw data from the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(script_dir, "C_Easy1_noise005.npy")
    traces = load_npy_file(raw_data_path)

    print(f"Data shape: {traces.shape}")
    print(f"Data type: {traces.dtype}")
    print(f"Data range: {traces.min()} to {traces.max()}")

    # Apply scale factor
    traces = traces * data_info["params"]["scale_factor"]

    # Create a SpikeInterface recording object
    recording = se.NumpyRecording(traces, sampling_frequency=data_info["params"]["samplerate"])
    
    # Set channel locations explicitly for a single channel
    channel_locations = np.array([[0.0, 0.0]])  # x, y coordinates for a single channel
    recording.set_channel_locations(channel_locations)

    print(f'Recording: {recording.get_num_channels()} channels; {recording.get_total_duration():.2f} sec')

    timer = time.time()

    # Preprocessing
    recording_filtered = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording_preprocessed = spre.whiten(recording_filtered)
    
    # Remove duplicate spikes
    recording_preprocessed = remove_duplicate_spikes(recording_preprocessed)

    # Create a unique temporary directory
    temp_dir = create_unique_temp_dir()

    try:
        # Cache the recording to the temporary directory for efficient reading
        recording_cached = create_cached_recording(recording_preprocessed, folder=str(temp_dir))

        # Sorting
        print('Starting MountainSort5')
        sorting = ms5.sorting_scheme1(
            recording_cached,
            sorting_parameters=ms5.Scheme1SortingParameters(
                detect_sign=data_info["params"]["spike_sign"],
                detect_threshold=6,  # Increased threshold
                detect_time_radius_msec=0.5,
                snippet_T1=20,
                snippet_T2=20,
                npca_per_channel=3,
                npca_per_subdivision=10
            )
        )
        assert isinstance(sorting, si.BaseSorting)

        elapsed_sec = time.time() - timer
        duration_sec = recording.get_total_duration()
        print(f'Elapsed time for sorting: {elapsed_sec:.2f} sec -- x{(duration_sec / elapsed_sec):.2f} speed compared with real time')

        # Print sorting results
        print('Sorting results:')
        print(f'Found {len(sorting.get_unit_ids())} units')
        for unit_id in sorting.get_unit_ids():
            print(f'  Unit {unit_id}: {len(sorting.get_unit_spike_train(unit_id))} spikes')

        # Load ground truth sorting
        ground_truth_path = os.path.join(script_dir, "C_Easy1_noise005.firings_true.npy")
        if os.path.exists(ground_truth_path):
            sorting_true = load_ground_truth(ground_truth_path, data_info["params"]["samplerate"])

            # Compare with ground truth
            print('Comparing with ground truth')
            comparison = sc.compare_sorter_to_ground_truth(gt_sorting=sorting_true, tested_sorting=sorting)
            print(comparison.get_performance())

            # Visualizations
            print('Generating visualizations')

            # Raster plot
            fig = plt.figure(figsize=(12, 6))
            sw.plot_rasters(sorting, edge_cmap='viridis', figsize=(12, 6))
            plt.title('Spike Raster Plot')
            plt.savefig(os.path.join(script_dir, 'raster_plot.png'))
            plt.close(fig)

            # Confusion matrix
            fig = plt.figure(figsize=(10, 8))
            sw.plot_confusion_matrix(comparison, count_text=True)
            plt.title('Confusion Matrix')
            plt.savefig(os.path.join(script_dir, 'confusion_matrix.png'))
            plt.close(fig)

            # Agreement matrix
            fig = plt.figure(figsize=(10, 8))
            sw.plot_agreement_matrix(comparison, ordered=True)
            plt.title('Agreement Matrix')
            plt.savefig(os.path.join(script_dir, 'agreement_matrix.png'))
            plt.close(fig)

            print('Visualizations saved in the script directory')
        else:
            print("Ground truth file not found. Skipping comparison and visualization.")

    finally:
        # Clean up the temporary directory
        print(f"Attempting to remove temporary directory: {temp_dir}")
        try:
            # Force close any open file handles
            import gc
            gc.collect()
            # Try to remove the directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Temporary directory {temp_dir} has been removed.")
        except Exception as e:
            print(f"Error removing temporary directory {temp_dir}: {e}")
            print("You may need to manually remove this directory.")

if __name__ == '__main__':
    main()