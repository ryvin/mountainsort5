import os
import time
import uuid
import shutil
from pathlib import Path
import numpy as np
import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.comparison as sc
import mountainsort5 as ms5
from mountainsort5.util import create_cached_recording
import spikeforest as sf
from generate_visualization_output import generate_visualization_output

def create_unique_temp_dir():
    temp_dir = Path(os.environ.get('TEMP', '/tmp')) / f'mountainsort5_{uuid.uuid4().hex}'
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir

def safe_remove(path, retries=5, delay=1):
    for i in range(retries):
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
            return True
        except Exception as e:
            print(f"Attempt {i+1} to remove {path} failed: {e}")
            if i < retries - 1:
                time.sleep(delay)
    print(f"Failed to remove {path} after {retries} attempts")
    return False

def main():
    paired_kampff_uri = 'sha1://b8b571d001f9a531040e79165e8f492d758ec5e0?paired-kampff-spikeforest-recordings.json'

    recordings = sf.load_spikeforest_recordings(paired_kampff_uri)

    rec = recordings[1]

    print(f'{rec.study_name}/{rec.recording_name} {rec.num_channels} channels; {rec.duration_sec} sec')

    print('Loading recording and sorting_true')
    recording = rec.get_recording_extractor()
    sorting_true = rec.get_sorting_true_extractor()

    channel_locations = recording.get_channel_locations()
    for m in range(channel_locations.shape[0]):
        print(f'Channel {recording.channel_ids[m]}: {channel_locations[m, 0]} {channel_locations[m, 1]}')

    timer = time.time()

    recording_filtered = spre.bandpass_filter(recording, freq_min=300, freq_max=6000, dtype=np.float32)
    recording_preprocessed: si.BaseRecording = spre.whiten(recording_filtered)

    # Create a unique temporary directory
    temp_dir = create_unique_temp_dir()
    
    try:
        recording_cached = create_cached_recording(recording_preprocessed, folder=str(temp_dir))

        print('Starting MountainSort5')
        sorting = ms5.sorting_scheme1(
            recording_cached,
            sorting_parameters=ms5.Scheme1SortingParameters(
                detect_channel_radius=100,
                snippet_mask_radius=100
            )
        )
        assert isinstance(sorting, si.BaseSorting)

        elapsed_sec = time.time() - timer
        duration_sec = recording.get_total_duration()
        print(f'Elapsed time for sorting: {elapsed_sec:.2f} sec -- x{(duration_sec / elapsed_sec):.2f} speed compared with real time for {recording.get_num_channels()} channels')

        print('Comparing with truth')
        comparison: sc.GroundTruthComparison = sc.compare_sorter_to_ground_truth(gt_sorting=sorting_true, tested_sorting=sorting)
        print(comparison.get_performance())

        if os.getenv('GENERATE_VISUALIZATION_OUTPUT') == '1':
            generate_visualization_output(rec=rec, recording_preprocessed=recording_preprocessed, sorting=sorting, sorting_true=sorting_true)

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