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

def create_unique_temp_dir():
    temp_dir = Path(os.environ.get('TEMP', '/tmp')) / f'mountainsort5_{uuid.uuid4().hex}'
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir

def load_npy_file(file_path):
    data = np.load(file_path)
    if data.ndim == 1:
        data = data.reshape(-1, 1)  # Reshape to 2D if it's 1D
    elif data.ndim == 2 and data.shape[0] == 1:
        data = data.T  # Transpose if it's 1 x N instead of N x 1
    return data.astype(np.float32)  # Ensure float32 data type

def remove_duplicate_spikes(recording, tolerance=0.9999):
    traces = recording.get_traces()
    unique_traces, unique_indices = np.unique(traces, axis=0, return_index=True)
    if len(unique_traces) < len(traces):
        print(f"Removed {len(traces) - len(unique_traces)} duplicate spikes")
        unique_indices.sort()
        return recording.frame_slice(start_frame=unique_indices[0], end_frame=unique_indices[-1])
    return recording

def load_ground_truth(file_path, samplerate):
    ground_truth_data = np.load(file_path)
    spike_times = ground_truth_data[:, 1].astype(int)
    spike_labels = ground_truth_data[:, 2].astype(int)
    unit_ids = np.unique(spike_labels)
    spike_trains = [spike_times[spike_labels == unit_id] for unit_id in unit_ids]
    sorting_true = se.NumpySorting.from_times_labels(
        times_list=spike_trains,
        labels_list=unit_ids.tolist(),
        sampling_frequency=samplerate
    )
    return sorting_true

def force_single_segment(sorting):
    all_spike_trains = []
    all_unit_ids = []
    for unit_id in sorting.get_unit_ids():
        spike_train = []
        for segment_index in range(sorting.get_num_segments()):
            segment_spike_train = sorting.get_unit_spike_train(unit_id, segment_index=segment_index)
            spike_train.extend(segment_spike_train)
        all_spike_trains.append(np.sort(np.array(spike_train)))
        all_unit_ids.append(unit_id)
    
    print(f"Number of units: {len(all_unit_ids)}")
    print(f"Number of spike trains: {len(all_spike_trains)}")
    for i, (unit_id, spike_train) in enumerate(zip(all_unit_ids, all_spike_trains)):
        print(f"Unit {unit_id}: {len(spike_train)} spikes")
        if i >= 4:  # Print only first 5 units to avoid clutter
            print("...")
            break
    
    return se.NumpySorting.from_times_labels(
        times_list=all_spike_trains,
        labels_list=all_unit_ids,
        sampling_frequency=sorting.get_sampling_frequency()
    )

def ensure_same_segments(sorting1, sorting2):
    print(f"Initial number of segments - sorting1: {sorting1.get_num_segments()}, sorting2: {sorting2.get_num_segments()}")
    
    print("Forcing both sortings to have a single segment...")
    print("Sorting 1 (Ground Truth) details:")
    sorting1 = force_single_segment(sorting1)
    print("\nSorting 2 (MountainSort5 output) details:")
    sorting2 = force_single_segment(sorting2)
    
    print(f"\nFinal number of segments - sorting1: {sorting1.get_num_segments()}, sorting2: {sorting2.get_num_segments()}")
    
    if sorting1.get_num_segments() != sorting2.get_num_segments():
        raise ValueError("Failed to equalize the number of segments between sortings.")
    
    return sorting1, sorting2

def align_and_force_single_segment(sorting1, sorting2):
    print("Aligning sortings and forcing single segment...")
    
    # Get all spikes from sorting1 (ground truth)
    all_spikes_1 = []
    all_labels_1 = []
    for unit_id in sorting1.get_unit_ids():
        for segment_index in range(sorting1.get_num_segments()):
            spikes = sorting1.get_unit_spike_train(unit_id, segment_index=segment_index)
            all_spikes_1.extend(spikes)
            all_labels_1.extend([unit_id] * len(spikes))
    
    # Get all spikes from sorting2 (MountainSort5 output)
    all_spikes_2 = []
    all_labels_2 = []
    for unit_id in sorting2.get_unit_ids():
        spikes = sorting2.get_unit_spike_train(unit_id)
        all_spikes_2.extend(spikes)
        all_labels_2.extend([unit_id] * len(spikes))
    
    # Create new single-segment sortings
    new_sorting1 = se.NumpySorting.from_times_labels(
        times_list=[np.array(all_spikes_1)],
        labels_list=[np.array(all_labels_1)],
        sampling_frequency=sorting1.get_sampling_frequency()
    )
    
    new_sorting2 = se.NumpySorting.from_times_labels(
        times_list=[np.array(all_spikes_2)],
        labels_list=[np.array(all_labels_2)],
        sampling_frequency=sorting2.get_sampling_frequency()
    )
    
    print(f"New sorting1 (Ground Truth): {len(new_sorting1.get_unit_ids())} units, {new_sorting1.get_num_segments()} segment(s)")
    print(f"New sorting2 (MountainSort5): {len(new_sorting2.get_unit_ids())} units, {new_sorting2.get_num_segments()} segment(s)")
    
    return new_sorting1, new_sorting2

def main():
    # Data file information
    data_info = {
        "geom": [0.0, 0.0],
        "params": {
            "samplerate": 24000,
            "scale_factor": 0.01,
            "spike_sign": -1
        }
    }

    # Load the raw data
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
    channel_locations = np.array([data_info["geom"]])
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
        # Cache the recording
        recording_cached = create_cached_recording(recording_preprocessed, folder=str(temp_dir))

        # Sorting
        print('Starting MountainSort5')
        sorting = ms5.sorting_scheme1(
            recording_cached,
            sorting_parameters=ms5.Scheme1SortingParameters(
                detect_sign=data_info["params"]["spike_sign"],
                detect_threshold=6,
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

            # Align sortings and force single segment
            sorting_true, sorting = align_and_force_single_segment(sorting_true, sorting)

            # Compare with ground truth
            print('Comparing with ground truth')
            comparison = sc.compare_sorter_to_ground_truth(gt_sorting=sorting_true, tested_sorting=sorting)
            print(comparison.get_performance())

            # Visualizations
            print('Generating visualizations')

            # Raster plot
            fig, ax = plt.subplots(figsize=(12, 6))
            sw.plot_rasters(sorting, ax=ax)
            ax.set_title('Spike Raster Plot')
            plt.savefig(os.path.join(script_dir, 'raster_plot.png'))
            plt.close(fig)

            # Confusion matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            sw.plot_confusion_matrix(comparison, ax=ax)
            ax.set_title('Confusion Matrix')
            plt.savefig(os.path.join(script_dir, 'confusion_matrix.png'))
            plt.close(fig)

            # Agreement matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            sw.plot_agreement_matrix(comparison, ax=ax)
            ax.set_title('Agreement Matrix')
            plt.savefig(os.path.join(script_dir, 'agreement_matrix.png'))
            plt.close(fig)

            print('Visualizations saved in the script directory')
        else:
            print("Ground truth file not found. Skipping comparison and visualization.")
               

    finally:
        # Clean up the temporary directory
        print(f"Attempting to remove temporary directory: {temp_dir}")
        try:
            import gc
            gc.collect()
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Temporary directory {temp_dir} has been removed.")
        except Exception as e:
            print(f"Error removing temporary directory {temp_dir}: {e}")
            print("You may need to manually remove this directory.")

if __name__ == '__main__':
    main()