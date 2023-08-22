import os
import sys
import numpy as np
# Get the current working directory
cwd = os.getcwd()

# Append the relative path to video_synchronization
sys.path.append(os.path.join(cwd, 'video_synchronization'))

import utils
from utils import (
    get_audio_frame_rate,
    load_video_and_extract_audio,
    convert_to_mono,
    calculate_cross_correlation,
    find_offset,
    butter_bandpass_filter,
    get_num_sample,
    get_sorted_files,
    SAMPLE_RATE
)

# Important! Insert the path to the folder in which the models' parameters are saved
# If running from the root directory
models_path = r'.'  # This will point to the current directory, which is the root


lab_dir = os.path.join(models_path, "data", "Lab_webm")
ronen_dir = os.path.join(models_path, "data", "Ronen_webm")
labels_csv = os.path.join(models_path, "data", "labels.xlsx")


lab_files = get_sorted_files(lab_dir)
ronen_files = get_sorted_files(ronen_dir)

for lab_f, ronen_f in zip(lab_files, ronen_files):
    sample_num = get_num_sample(lab_f)
    ronen_sample_num = get_num_sample(ronen_f)
    print("Syncing video number: " + sample_num)
    if ronen_sample_num == sample_num:
        ronen = os.path.join(ronen_dir, ronen_f)
        lab = os.path.join(lab_dir, lab_f)

        # Define the frequency range for bird noises
        lowcut = 1000.0
        highcut = 2500.0

        # Load the videos and extract the audio
        ronen_video, ronen_audio = load_video_and_extract_audio(ronen)
        lab_video, lab_audio = load_video_and_extract_audio(lab)

        # Convert to mono
        ronen_audio = convert_to_mono(ronen_audio)
        lab_audio = convert_to_mono(lab_audio)

        # Normalize the audio
        ronen_audio = ronen_audio / np.max(np.abs(ronen_audio))
        lab_audio = lab_audio / np.max(np.abs(lab_audio))

        # Apply the bandpass filter
        ronen_audio = butter_bandpass_filter(ronen_audio, lowcut, highcut, SAMPLE_RATE, order=6)
        lab_audio = butter_bandpass_filter(lab_audio, lowcut, highcut, SAMPLE_RATE, order=6)

        cross_correlation = calculate_cross_correlation(ronen_audio, lab_audio)

        offset = find_offset(cross_correlation, len(ronen_audio))
        if offset > 0:
            offset_seconds = offset / SAMPLE_RATE
            lab_video = lab_video.set_start(offset_seconds, change_end=False)
            ronen_video = ronen_video.set_end(lab_video.duration)
            print('cutting Lab video')
        elif offset < 0:
            offset = -offset
            offset_seconds = offset / SAMPLE_RATE
            ronen_video = ronen_video.set_start(offset_seconds, change_end=False)
            lab_video = lab_video.set_end(ronen_video.duration)
            print('cutting Ronen video')
        else:
            print('no offset')

        # Define the output directories
        lab_output_dir = os.path.join(models_path, "data", "lab_synced")
        ronen_output_dir = os.path.join(models_path, "data", "ronen_synced")

        # Define the output file paths
        lab_output_file = os.path.join(lab_output_dir, f"lab_sync_video_{sample_num}.webm")
        ronen_output_file = os.path.join(ronen_output_dir, f"ronen_sync_video_{sample_num}.webm")

        # Save the synchronized videos
        lab_video.write_videofile(lab_output_file)
        ronen_video.write_videofile(ronen_output_file)

    else:
        print(f"No matching file found for sample number {sample_num} in 'ronen_dir'. Skipping...")
