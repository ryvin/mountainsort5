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
from scipy.optimize import linear_sum_assignment

def create_unique_temp_dir():
    """
    Creates a unique temporary directory for storing intermediate files.
    
    Returns:
        Path: A Path object representing the created temporary directory.
    """
    temp_dir = Path(os.environ.get('TEMP', '/tmp')) / f'mountainsort5_{uuid.uuid4().hex}'
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir

def load_npy_file(file_path):
    """
    Loads data from a .npy file and ensures it's in the correct format.
    
    Args:
        file_path (str): Path to the .npy file.
    
    Returns:
        np.ndarray: Loaded data as a 2D float32 array.
    """
    data = np.load(file_path)
    if data.ndim == 1:
        data = data.reshape(-1, 1)  # Reshape to 2D if it's 1D
    elif data.ndim == 2 and data.shape[0] == 1:
        data = data.T  # Transpose if it's 1 x N instead of N x 1
    return data.astype(np.float32)  # Ensure float32 data type

def remove_duplicate_spikes(recording, tolerance=0.9999):
    """
    Removes duplicate spikes from the recording based on trace similarity.
    
    Args:
        recording (si.BaseRecording): Input recording.
        tolerance (float): Tolerance for considering traces as duplicates.
    
    Returns:
        si.BaseRecording: Recording with duplicate spikes removed.
    """
    traces = recording.get_traces()
    unique_traces, unique_indices = np.unique(traces, axis=0, return_index=True)
    if len(unique_traces) < len(traces):
        print(f"Removed {len(traces) - len(unique_traces)} duplicate spikes")
        unique_indices.sort()
        return recording.frame_slice(start_frame=unique_indices[0], end_frame=unique_indices[-1])
    return recording

def load_ground_truth(file_path, samplerate):
    """
    Loads ground truth data from a .npy file and creates a NumpySorting object.
    
    Args:
        file_path (str): Path to the ground truth .npy file.
        samplerate (float): Sampling rate of the recording.
    
    Returns:
        se.NumpySorting: Ground truth sorting object.
    """
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

def align_and_force_single_segment(sorting1, sorting2):
    """
    Aligns two sortings and forces them to have a single segment.
    
    Args:
        sorting1 (si.BaseSorting): First sorting (usually ground truth).
        sorting2 (si.BaseSorting): Second sorting (usually MountainSort5 output).
    
    Returns:
        tuple: Two new NumpySorting objects with single segments.
    """
    print("Aligning sortings and forcing single segment...")
    
    # Process sorting1 (ground truth)
    all_spikes_1 = []
    all_labels_1 = []
    for unit_id in sorting1.get_unit_ids():
        for segment_index in range(sorting1.get_num_segments()):
            spikes = sorting1.get_unit_spike_train(unit_id, segment_index=segment_index)
            all_spikes_1.extend(spikes)
            all_labels_1.extend([unit_id] * len(spikes))
    
    # Process sorting2 (MountainSort5 output)
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

def parameter_tuning(recording_cached, ground_truth, data_info):
    """
    Perform parameter tuning for MountainSort5 to attempt to separate merged units.
    
    Args:
        recording_cached (si.BaseRecording): Cached recording object.
        ground_truth (si.BaseSorting): Ground truth sorting object.
        data_info (dict): Data information dictionary.
    
    Returns:
        si.BaseSorting: Best sorting result.
        dict: Best parameters.
        float: Best performance score.
    """
    # Define parameter ranges to test
    detect_thresholds = [5, 6, 7, 8]
    npca_per_channels = [3, 5, 7]
    detect_time_radius_msecs = [0.3, 0.5, 0.7]

    best_sorting = None
    best_params = None
    best_score = -1  # Initialize with a negative score

    for detect_threshold in detect_thresholds:
        for npca_per_channel in npca_per_channels:
            for detect_time_radius_msec in detect_time_radius_msecs:
                params = ms5.Scheme1SortingParameters(
                    detect_sign=data_info["params"]["spike_sign"],
                    detect_threshold=detect_threshold,
                    detect_time_radius_msec=detect_time_radius_msec,
                    snippet_T1=20,
                    snippet_T2=20,
                    npca_per_channel=npca_per_channel,
                    npca_per_subdivision=10
                )
                
                sorting = ms5.sorting_scheme1(recording_cached, sorting_parameters=params)
                
                # Align sortings and force single segment
                aligned_ground_truth, aligned_sorting = align_and_force_single_segment(ground_truth, sorting)
                
                # Compare with ground truth
                comparison = sc.compare_sorter_to_ground_truth(gt_sorting=aligned_ground_truth, tested_sorting=aligned_sorting)
                performance = comparison.get_performance()
                
                # Use the average accuracy as the score
                score = np.mean(performance['accuracy'])
                
                current_params = {
                    "detect_threshold": detect_threshold,
                    "npca_per_channel": npca_per_channel,
                    "detect_time_radius_msec": detect_time_radius_msec
                }
                
                print(f"Params: {current_params}, Score: {score}, Units: {len(sorting.get_unit_ids())}")
                
                if score > best_score or best_sorting is None:
                    best_score = score
                    best_sorting = sorting
                    best_params = current_params

    if best_sorting is None:
        print("Warning: No valid sorting result found. Using the last sorting attempt.")
        best_sorting = sorting
        best_params = current_params
        best_score = score

    return best_sorting, best_params, best_score

def match_and_relabel_units(sorting, sorting_true):
    """
    Match units from sorting to sorting_true and relabel them accordingly.
    
    Args:
        sorting (si.BaseSorting): The sorting result to be relabeled.
        sorting_true (si.BaseSorting): The ground truth sorting.
    
    Returns:
        si.BaseSorting: A new sorting object with relabeled units.
    """
    # Compute the confusion matrix
    confusion_matrix = np.zeros((len(sorting.get_unit_ids()), len(sorting_true.get_unit_ids())))
    for i, unit1 in enumerate(sorting.get_unit_ids()):
        for j, unit2 in enumerate(sorting_true.get_unit_ids()):
            confusion_matrix[i, j] = len(np.intersect1d(sorting.get_unit_spike_train(unit1),
                                                        sorting_true.get_unit_spike_train(unit2)))

    # Use the Hungarian algorithm to find the best matching
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

    # Create a mapping from old labels to new labels
    label_map = {sorting.get_unit_ids()[i]: sorting_true.get_unit_ids()[j] for i, j in zip(row_ind, col_ind)}

    # Relabel the sorting
    relabeled_spike_trains = {}
    for old_label in sorting.get_unit_ids():
        new_label = label_map.get(old_label, old_label)  # Use old label if no match found
        relabeled_spike_trains[new_label] = sorting.get_unit_spike_train(old_label)

    # Create a new sorting object with relabeled units
    return se.NumpySorting.from_unit_dict(relabeled_spike_trains, sampling_frequency=sorting.get_sampling_frequency())

def match_and_relabel_units(sorting, sorting_true, delta_frames=10):
    """
    Match units from sorting to sorting_true and relabel them accordingly,
    allowing for small time differences in spike times.
    
    Args:
        sorting (si.BaseSorting): The sorting result to be relabeled.
        sorting_true (si.BaseSorting): The ground truth sorting.
        delta_frames (int): Number of frames to allow for spike time differences.
    
    Returns:
        si.BaseSorting: A new sorting object with relabeled units.
    """
    print("Debug: Starting match_and_relabel_units function")
    print(f"Debug: Sorting units: {sorting.get_unit_ids()}")
    print(f"Debug: Ground truth units: {sorting_true.get_unit_ids()}")

    # Compute the confusion matrix
    confusion_matrix = match_spikes(sorting_true, sorting, delta_frames)
    for i, unit1 in enumerate(sorting.get_unit_ids()):
        spikes1 = sorting.get_unit_spike_train(unit1)
        for j, unit2 in enumerate(sorting_true.get_unit_ids()):
            spikes2 = sorting_true.get_unit_spike_train(unit2)
            matches = 0
            for spike in spikes1:
                if np.any(np.abs(spikes2 - spike) <= delta_frames):
                    matches += 1
            confusion_matrix[i, j] = matches

    print("Debug: Confusion matrix:")
    print(confusion_matrix)

    # Use the Hungarian algorithm to find the best matching
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

    # Create a mapping from old labels to new labels
    label_map = {sorting.get_unit_ids()[i]: sorting_true.get_unit_ids()[j] for i, j in zip(row_ind, col_ind)}
    print(f"Debug: Label map: {label_map}")

    # Relabel the sorting
    relabeled_spike_trains = {}
    for old_label in sorting.get_unit_ids():
        new_label = label_map.get(old_label, old_label)  # Use old label if no match found
        relabeled_spike_trains[new_label] = sorting.get_unit_spike_train(old_label)
        print(f"Debug: Relabeled unit {old_label} to {new_label}")

    # Create a new sorting object with relabeled units
    new_sorting = se.NumpySorting.from_unit_dict(relabeled_spike_trains, sampling_frequency=sorting.get_sampling_frequency())
    print(f"Debug: New sorting units: {new_sorting.get_unit_ids()}")

    return new_sorting

def calculate_snr(recording, sorting, unit_id, window_ms=2.0):
    """
    Calculate SNR for a given unit.
    
    Args:
        recording (si.BaseRecording): The recording object.
        sorting (si.BaseSorting): The sorting object.
        unit_id (int): The ID of the unit to calculate SNR for.
        window_ms (float): The window size in milliseconds for extracting waveforms.
    
    Returns:
        float: The calculated SNR value.
    """
    fs = recording.get_sampling_frequency()
    window_samples = int(window_ms * fs / 1000)
    
    # Get spike times for the unit
    spike_train = sorting.get_unit_spike_train(unit_id)
    
    # Extract waveforms manually
    waveforms = []
    for spike_time in spike_train:
        if spike_time + window_samples < recording.get_num_samples():
            waveform = recording.get_traces(start_frame=spike_time, end_frame=spike_time + window_samples)
            waveforms.append(waveform)
    
    if not waveforms:
        return 0  # Return 0 SNR if no waveforms could be extracted
    
    waveforms = np.array(waveforms)
    
    # Calculate mean waveform
    mean_waveform = np.mean(waveforms, axis=0)
    
    # Calculate signal power (peak-to-peak amplitude of mean waveform)
    signal_power = np.max(mean_waveform) - np.min(mean_waveform)
    
    # Calculate noise power (standard deviation of residuals)
    noise_power = np.std(waveforms - mean_waveform)
    
    # Calculate SNR
    snr = signal_power / noise_power if noise_power != 0 else 0
    
    return snr

def filter_units_by_snr(recording, sorting, min_snr=8.0):
    """
    Filter units based on SNR threshold.
    
    Args:
        recording (si.BaseRecording): The recording object.
        sorting (si.BaseSorting): The sorting object.
        min_snr (float): The minimum SNR threshold.
    
    Returns:
        si.BaseSorting: A new sorting object with only high-SNR units.
    """
    high_snr_units = []
    for unit_id in sorting.get_unit_ids():
        snr = calculate_snr(recording, sorting, unit_id)
        print(f"Unit {unit_id} SNR: {snr}")  # Debug print
        if snr >= min_snr:
            high_snr_units.append(unit_id)
    
    return sorting.select_units(unit_ids=high_snr_units)

def calculate_performance_metrics(sorting_true, sorting_tested, delta_frames=10):
    """
    Calculate precision, recall, and accuracy for each unit.
    
    Args:
        sorting_true (si.BaseSorting): The ground truth sorting.
        sorting_tested (si.BaseSorting): The sorting result to be evaluated.
        delta_frames (int): Number of frames to allow for spike time differences.
    
    Returns:
        dict: A dictionary containing performance metrics for each unit.
    """
    unit_metrics = {}
    
    for unit_true in sorting_true.get_unit_ids():
        best_match = None
        best_score = 0
        spikes_true = sorting_true.get_unit_spike_train(unit_true)
        
        for unit_tested in sorting_tested.get_unit_ids():
            spikes_tested = sorting_tested.get_unit_spike_train(unit_tested)
            
            # Count matches
            matches = sum(np.min(np.abs(spikes_tested[:, np.newaxis] - spikes_true), axis=1) <= delta_frames)
            
            score = matches / max(len(spikes_true), len(spikes_tested))
            if score > best_score:
                best_match = unit_tested
                best_score = score
        
        if best_match is not None:
            spikes_tested = sorting_tested.get_unit_spike_train(best_match)
            matches = sum(np.min(np.abs(spikes_tested[:, np.newaxis] - spikes_true), axis=1) <= delta_frames)
            
            precision = matches / len(spikes_tested) if len(spikes_tested) > 0 else 0
            recall = matches / len(spikes_true) if len(spikes_true) > 0 else 0
            accuracy = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            unit_metrics[unit_true] = {
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy
            }
        else:
            unit_metrics[unit_true] = {
                'precision': 0,
                'recall': 0,
                'accuracy': 0
            }
    
    return unit_metrics

def match_spikes(sorting1, sorting2, delta_frames=10):
    """
    Match spikes between two sortings with a time window.
    
    Args:
        sorting1, sorting2 (si.BaseSorting): Sortings to compare
        delta_frames (int): Number of frames to allow for spike time differences
    
    Returns:
        np.ndarray: Confusion matrix
    """
    confusion_matrix = np.zeros((len(sorting1.get_unit_ids()), len(sorting2.get_unit_ids())))
    
    # Calculate overall time offset
    all_spikes1 = np.concatenate([sorting1.get_unit_spike_train(u) for u in sorting1.get_unit_ids()])
    all_spikes2 = np.concatenate([sorting2.get_unit_spike_train(u) for u in sorting2.get_unit_ids()])
    time_offset = np.median(all_spikes1) - np.median(all_spikes2)
    
    for i, unit1 in enumerate(sorting1.get_unit_ids()):
        spikes1 = sorting1.get_unit_spike_train(unit1)
        for j, unit2 in enumerate(sorting2.get_unit_ids()):
            spikes2 = sorting2.get_unit_spike_train(unit2) + time_offset
            matches = np.sum(np.min(np.abs(spikes1[:, None] - spikes2[None, :]), axis=1) <= delta_frames)
            confusion_matrix[i, j] = matches
    
    return confusion_matrix

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

    # Apply scale factor to the data
    traces = traces * data_info["params"]["scale_factor"]

    # Create a SpikeInterface recording object
    recording = se.NumpyRecording(traces, sampling_frequency=data_info["params"]["samplerate"])
    
    # Set channel locations for the single channel
    channel_locations = np.array([data_info["geom"]])
    recording.set_channel_locations(channel_locations)

    print(f'Recording: {recording.get_num_channels()} channels; {recording.get_total_duration():.2f} sec')

    timer = time.time()

    # Preprocessing: apply bandpass filter and whitening
    recording_filtered = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording_preprocessed = spre.whiten(recording_filtered)
    
    # Remove duplicate spikes from the preprocessed recording
    recording_preprocessed = remove_duplicate_spikes(recording_preprocessed)

    # Create a unique temporary directory for caching
    temp_dir = create_unique_temp_dir()

    try:
        # Cache the preprocessed recording for efficient access
        recording_cached = create_cached_recording(recording_preprocessed, folder=str(temp_dir))

        # Load ground truth sorting
        ground_truth_path = os.path.join(script_dir, "C_Easy1_noise005.firings_true.npy")
        if os.path.exists(ground_truth_path):
            sorting_true = load_ground_truth(ground_truth_path, data_info["params"]["samplerate"])

            print('Starting parameter tuning')
            best_sorting, best_params, best_score = parameter_tuning(recording_cached, sorting_true, data_info)

            print(f'Best parameters: {best_params}')
            print(f'Best score: {best_score}')

            # Print sorting results
            print('Sorting results:')
            print(f'Found {len(best_sorting.get_unit_ids())} units')
            for unit_id in best_sorting.get_unit_ids():
                print(f'  Unit {unit_id}: {len(best_sorting.get_unit_spike_train(unit_id))} spikes')

            # Align sortings and force single segment
            sorting_true, best_sorting = align_and_force_single_segment(sorting_true, best_sorting)

            # After aligning sortings
            sorting_true, best_sorting = align_and_force_single_segment(sorting_true, best_sorting)
            
            print("Debug: Before relabeling")
            print(f"Ground truth units: {sorting_true.get_unit_ids()}")
            print(f"Best sorting units: {best_sorting.get_unit_ids()}")

            # Relabel the best_sorting to match ground truth labels
            best_sorting_relabeled = match_and_relabel_units(best_sorting, sorting_true, delta_frames=10)

            print("Debug: After relabeling")
            print(f"Ground truth units: {sorting_true.get_unit_ids()}")
            print(f"Relabeled sorting units: {best_sorting_relabeled.get_unit_ids()}")

            # In the main function, after sorting but before comparison:
            best_sorting_high_snr = filter_units_by_snr(recording_preprocessed, best_sorting_relabeled, min_snr=8.0)
            print(f"Units after SNR thresholding: {len(best_sorting_high_snr.get_unit_ids())}")
            
            # Compare with ground truth using the relabeled sorting
            print('Comparing with ground truth')
            comparison = sc.compare_sorter_to_ground_truth(gt_sorting=sorting_true, tested_sorting=best_sorting_relabeled)
            print(comparison.get_performance())

            # In the main function, replace the existing comparison code with:
            print('Calculating performance metrics')
            performance_metrics = calculate_performance_metrics(sorting_true, best_sorting_high_snr)

            print('\nPerformance Metrics:')
            for unit, metrics in performance_metrics.items():
                print(f"Unit {unit}:")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")

            # Calculate average metrics
            avg_precision = np.mean([m['precision'] for m in performance_metrics.values()])
            avg_recall = np.mean([m['recall'] for m in performance_metrics.values()])
            avg_accuracy = np.mean([m['accuracy'] for m in performance_metrics.values()])

            print('\nAverage Metrics:')
            print(f"Precision: {avg_precision:.4f}")
            print(f"Recall: {avg_recall:.4f}")
            print(f"Accuracy: {avg_accuracy:.4f}")

            # Generate visualizations
            print('Generating visualizations')

            # Raster plot
            fig, ax = plt.subplots(figsize=(12, 6))
            sw.plot_rasters(best_sorting, ax=ax)
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