import numpy as np
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import mountainsort5 as ms5
from mountainsort5.util import create_cached_recording
from tempfile import TemporaryDirectory
import os

def load_npy_file(file_path, dtype=None):
    data = np.load(file_path)
    if dtype:
        data = data.astype(dtype)
    if data.ndim == 1:
        data = data.reshape(-1, 1)  # Reshape to 2D if it's 1D
    elif data.ndim > 2:
        raise ValueError("The .npy file should contain 1D or 2D data")
    return data

# Ask for file path
last_file = r"C:\Code\mountainsort5\C_Easy1_noise005.npy"
file_path = input(f"Enter the path to your recording file (or press Enter to use the last file: {last_file}): ").strip() or last_file

print(f"Loading file: {file_path}")

# Ask for data type
dtype_choice = input("Choose data type (1 for float32, 2 for int16, or press Enter for no change): ").strip()
if dtype_choice == '1':
    dtype = np.float32
elif dtype_choice == '2':
    dtype = np.int16
else:
    dtype = None

try:
    # Load the .npy file
    traces = load_npy_file(file_path, dtype)
    
    print(f"Data shape: {traces.shape}")
    print(f"Data type: {traces.dtype}")
    print(f"Data range: {traces.min()} to {traces.max()}")

    # Create a SpikeInterface recording object
    sampling_frequency = 30000  # You might want to make this configurable
    recording = se.NumpyRecording(traces, sampling_frequency=sampling_frequency)

    print(f"Recording duration: {recording.get_total_duration()} seconds")

    # Ask user if they want to preprocess
    preprocess = input("Do you want to preprocess the data? (y/n): ").strip().lower() == 'y'

    if preprocess:
        # Preprocessing
        recording_filtered = spre.bandpass_filter(recording, freq_min=300, freq_max=6000, dtype=np.float32)
        recording_preprocessed = spre.whiten(recording_filtered)
    else:
        recording_preprocessed = recording

    with TemporaryDirectory() as tmpdir:
        # Cache the recording
        recording_cached = create_cached_recording(recording_preprocessed, folder=tmpdir)

        # Run MountainSort5 using Scheme 1
        sorting = ms5.sorting_scheme1(
            recording=recording_cached,
            sorting_parameters=ms5.Scheme1SortingParameters(
                detect_sign=-1,  # Adjust based on your data
                detect_threshold=5.5,
                detect_time_radius_msec=0.5,
                snippet_T1=20,
                snippet_T2=20,
                npca_per_channel=3,
                npca_per_subdivision=10
            )
        )

    # Print basic information about the sorting results
    print(f"Found {len(sorting.get_unit_ids())} units")
    for unit_id in sorting.get_unit_ids():
        print(f"  Unit {unit_id}: {len(sorting.get_unit_spike_train(unit_id))} spikes")

    # You can now save or further analyze the sorting results

except Exception as e:
    print(f"Error: {str(e)}")