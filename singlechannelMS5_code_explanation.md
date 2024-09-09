# Detailed Explanation of Single-Channel Spike Sorting Code

## 1. Import Statements
The script imports necessary libraries, including:
- Standard Python libraries (os, time, uuid, shutil, pathlib)
- NumPy for numerical operations
- SpikeInterface (si) and its submodules for spike sorting operations
- MountainSort5 (ms5) for the sorting algorithm
- Matplotlib for visualization
- scipy.optimize for the Hungarian algorithm used in unit matching

## 2. Utility Functions

### create_unique_temp_dir()
- Creates a unique temporary directory using UUID for storing intermediate files

### load_npy_file(file_path)
- Loads data from a .npy file, ensuring 2D float32 format

### remove_duplicate_spikes(recording, tolerance=0.9999)
- Removes duplicate spikes from the recording based on trace similarity

### load_ground_truth(file_path, samplerate)
- Loads ground truth data from a .npy file and creates a NumpySorting object

### align_and_force_single_segment(sorting1, sorting2)
- Aligns two sortings and forces them to have a single segment

### parameter_tuning(recording_cached, ground_truth, data_info)
- Performs parameter tuning for MountainSort5 to optimize sorting results
- Tests various combinations of detect_threshold, npca_per_channel, and detect_time_radius_msec
- Returns the best sorting result, best parameters, and best score

### match_and_relabel_units(sorting, sorting_true, delta_frames=10)
- Matches units from the sorting result to ground truth units
- Relabels units allowing for small time differences in spike times
- Uses the Hungarian algorithm for optimal matching

### calculate_snr(recording, sorting, unit_id, window_ms=2.0)
- Calculates Signal-to-Noise Ratio (SNR) for a given unit
- Extracts waveforms manually and computes SNR based on signal and noise power

### filter_units_by_snr(recording, sorting, min_snr=8.0)
- Filters units based on an SNR threshold
- Returns a new sorting object with only high-SNR units

### calculate_performance_metrics(sorting_true, sorting_tested, delta_frames=10)
- Calculates precision, recall, and accuracy for each unit
- Compares the tested sorting against the ground truth

## 3. Main Function

### Data Loading and Preprocessing
- Loads raw data and applies scaling
- Creates a SpikeInterface recording object
- Applies bandpass filtering and whitening
- Removes duplicate spikes

### Spike Sorting and Parameter Tuning
- Caches the preprocessed recording
- Performs parameter tuning to find optimal MountainSort5 parameters
- Applies MountainSort5 algorithm with the best parameters

### Ground Truth Comparison and Unit Matching
- Loads ground truth data
- Aligns and relabels sorted units to match ground truth
- Filters units based on SNR threshold

### Performance Evaluation
- Compares sorting results with ground truth
- Calculates detailed performance metrics (precision, recall, accuracy) for each unit
- Computes average performance metrics across all units

### Visualization
- Generates visualizations:
  1. Raster plot of sorted spikes
  2. Confusion matrix comparing sorted results to ground truth
  3. Agreement matrix showing similarity between sorted and ground truth units
- Saves visualizations as PNG files

### Cleanup
- Removes the temporary directory used during processing

## 4. Key Enhancements

1. **Parameter Tuning**: Systematically tests different MountainSort5 parameters to optimize sorting results.
2. **SNR Calculation and Filtering**: Implements SNR calculation for each unit and filters out low-SNR units.
3. **Unit Matching and Relabeling**: Uses the Hungarian algorithm to optimally match and relabel sorted units to ground truth units.
4. **Detailed Performance Metrics**: Calculates precision, recall, and accuracy for each unit, providing a comprehensive evaluation of sorting performance.

## 5. Workflow

1. Load and preprocess raw data
2. Perform parameter tuning for MountainSort5
3. Apply MountainSort5 with optimal parameters
4. Match and relabel sorted units to ground truth
5. Filter units based on SNR
6. Calculate detailed performance metrics
7. Generate and save visualizations

This updated script provides a more robust and comprehensive workflow for single-channel spike sorting, incorporating parameter optimization, SNR-based unit filtering, and detailed performance evaluation.
