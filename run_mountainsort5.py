import os
import json
from tempfile import TemporaryDirectory
import numpy as np
import spikeinterface as si
import spikeinterface.preprocessing as spre
import mountainsort5 as ms5
from mountainsort5.util import create_cached_recording

def load_last_file():
    if os.path.exists('last_file.json'):
        with open('last_file.json', 'r') as f:
            return json.load(f)['last_file']
    return None

def save_last_file(file_path):
    with open('last_file.json', 'w') as f:
        json.dump({'last_file': file_path}, f)

def load_recording(file_path):
    try:
        if file_path.endswith('.npy'):
            # Load NumPy array
            data = np.load(file_path)
            if data.ndim == 1:
                data = data.reshape(1, -1)  # Reshape to 2D for single-channel data
            elif data.shape[0] > data.shape[1]:
                data = data.T  # Transpose if samples are in the first dimension
            return si.NumpyRecording(data, sampling_frequency=30000)  # Assume 30kHz sampling rate
        else:
            # Try to load using SpikeInterface
            return si.load_extractor(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def run_mountainsort5(file_path, scheme='2'):
    # Load the recording
    recording = load_recording(file_path)
    if recording is None:
        return

    # Print recording information
    print(f"Recording duration: {recording.get_total_duration()} seconds")
    print(f"Number of channels: {recording.get_num_channels()}")
    print(f"Number of samples: {recording.get_num_samples()}")

    # Ask user for preprocessing options
    do_filter = input("Apply bandpass filter? (y/n): ").lower() == 'y'
    do_whiten = input("Apply whitening? (y/n): ").lower() == 'y'

    recording_preprocessed = recording

    if do_filter:
        freq_min = float(input("Enter minimum frequency for bandpass filter (default 300): ") or 300)
        freq_max = float(input("Enter maximum frequency for bandpass filter (default 6000): ") or 6000)
        recording_preprocessed = spre.bandpass_filter(recording_preprocessed, freq_min=freq_min, freq_max=freq_max, dtype=np.float32)

    if do_whiten and recording.get_num_channels() > 1:
        recording_preprocessed = spre.whiten(recording_preprocessed)
    elif do_whiten:
        print("Whitening is not applicable for single-channel recordings. Skipping this step.")

    with TemporaryDirectory(dir=os.path.dirname(file_path)) as tmpdir:
        # cache the recording to a temporary directory for efficient reading
        recording_cached = create_cached_recording(recording_preprocessed, folder=tmpdir)

        # Adjust sorting parameters for single-channel recording if necessary
        if recording.get_num_channels() == 1:
            sorting_params = {
                "detect_channel_radius": 0,
                "snippet_T1": 20,
                "snippet_T2": 20,
            }
        else:
            sorting_params = {}

        if scheme == '1':
            sorting = ms5.sorting_scheme1(
                recording=recording_cached,
                sorting_parameters=ms5.Scheme1SortingParameters(**sorting_params)
            )
        elif scheme == '2':
            sorting = ms5.sorting_scheme2(
                recording=recording_cached,
                sorting_parameters=ms5.Scheme2SortingParameters(**sorting_params)
            )
        elif scheme == '3':
            sorting = ms5.sorting_scheme3(
                recording=recording_cached,
                sorting_parameters=ms5.Scheme3SortingParameters(**sorting_params)
            )
        else:
            raise ValueError("Invalid scheme. Choose 1, 2, or 3.")

    # Print basic information
    print(sorting)
    print(f"Found {len(sorting.unit_ids)} units")
    
    # Export results
    output_folder = os.path.join(os.path.dirname(file_path), f'ms5_scheme{scheme}_output')
    si.export_to_phy(recording_preprocessed, sorting, output_folder=output_folder)
    print(f"Results exported to: {output_folder}")

def find_recording_file(directory):
    for file in os.listdir(directory):
        if file.endswith(('.npy', '.bin', '.dat')):  # Add more extensions if needed
            return os.path.join(directory, file)
    return None

def main():
    while True:
        last_file = load_last_file()
        
        if last_file:
            print(f"Last used file: {last_file}")
            use_last = input("Use this file? (y/n): ").lower() == 'y'
            if use_last:
                file_path = last_file
            else:
                file_path = input("Enter the path to your recording file (or press Enter to search current directory): ")
        else:
            file_path = input("Enter the path to your recording file (or press Enter to search current directory): ")
        
        if not file_path:
            file_path = find_recording_file(os.getcwd())
            if file_path:
                print(f"Found recording file: {file_path}")
            else:
                print("No recording file found in the current directory.")
                continue

        if os.path.exists(file_path):
            save_last_file(file_path)
            scheme = input("Enter sorting scheme (1, 2, or 3): ")
            try:
                run_mountainsort5(file_path, scheme)
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                retry = input("Do you want to try again? (y/n): ").lower() == 'y'
                if not retry:
                    break
        else:
            print("File not found. Please check the path and try again.")
            retry = input("Do you want to try again? (y/n): ").lower() == 'y'
            if not retry:
                break

if __name__ == "__main__":
    main()