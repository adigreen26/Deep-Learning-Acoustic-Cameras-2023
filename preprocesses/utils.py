import moviepy.editor as mpy
import os
from scipy.signal import butter, lfilter
import cv2
import numpy as np
import matplotlib.pyplot as plt

SAMPLE_RATE = 44100


def get_audio_frame_rate(video_path):
    # Load the video file
    video = mpy.VideoFileClip(video_path)

    # Return the audio frame rate
    return video.audio.fps


def load_video_and_extract_audio(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    video = mpy.VideoFileClip(file_path)
    audio = video.audio.to_soundarray(fps=SAMPLE_RATE)
    return video, audio


def convert_to_mono(audio):
    if audio.ndim == 2:
        return np.mean(audio, axis=1)
    return audio


def calculate_cross_correlation(audio1, audio2):
    return np.correlate(audio1, audio2, "full")


def find_offset(cross_correlation, audio_length):
    return cross_correlation.argmax() - (audio_length - 1)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def get_num_sample(filename):
    num_sample = filename.rsplit('_', 1)[1]
    num_sample = num_sample.split('.', 1)[0]
    return num_sample


def get_sorted_files(directory):
    # Get a sorted list of files in the directory based on sample number
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return sorted(files, key=lambda x: int(get_num_sample(x)))


def list_folder_contents_sorted(folder_path):
    contents = os.listdir(folder_path)
    sorted_contents = sorted(contents)  # Sort the contents alphabetically
    print(len(contents))
    for item in sorted_contents:
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            print(f"File: {item}")
        elif os.path.isdir(item_path):
            print(f"Directory: {item}")


def plot_label_and_image(image_path, label_path):
    image = cv2.imread(image_path)
    label = np.load(label_path)
    print("Label values:", label)
    plt.figure(figsize=(6, 3))
    plt.plot()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Image")
    plt.axis("off")
    plt.tight_layout()
    plt.show()