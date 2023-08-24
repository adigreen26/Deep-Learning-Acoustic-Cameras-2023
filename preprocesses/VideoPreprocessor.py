# Standard libraries
import os
import random

# Third-party libraries
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class VideoPreprocessor:
    def __init__(self, lab_source_dir, ronen_source_dir, target_dir, labels_csv, num_frames=5, crop_area=None):
        self.lab_source_dir = lab_source_dir
        self.ronen_source_dir = ronen_source_dir
        self.target_dir = target_dir
        self.num_frames = num_frames
        self.crop_area = crop_area  # Format: (top, left, bottom, right)

        self.labels = pd.read_excel(labels_csv, usecols=range(4)).set_index('filename')
        self.labels = self.labels.astype(float)
        self.num_videos = 0

        # Create the target directory if it doesn't exist
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)

        self.train_indices = []
        self.test_indices = []

    def preprocess_videos(self):
        lab_files = sorted(os.listdir(self.lab_source_dir))
        ronen_files = sorted(os.listdir(self.ronen_source_dir))

        assert len(lab_files) == len(ronen_files), "Number of lab and ronen videos must match."

        for lab_file, ronen_file in zip(lab_files, ronen_files):
            lab_video_path = os.path.join(self.lab_source_dir, lab_file)
            ronen_video_path = os.path.join(self.ronen_source_dir, ronen_file)

            lab_frames = self.load_and_preprocess(lab_video_path)
            ronen_frames = self.load_and_preprocess(ronen_video_path)

            for idx, (lab_frame, ronen_frame) in enumerate(zip(lab_frames, ronen_frames)):
                lab_label = self.get_label(ronen_file)
                ronen_label = self.get_label(ronen_file)

                lab_filename = f"{lab_file.split('.')[0]}_{idx:02d}.jpg"
                ronen_filename = f"{ronen_file.split('.')[0]}_{idx:02d}.jpg"

                self.save_processed_frame(lab_frame, lab_label, lab_filename, is_lab=True)
                self.save_processed_frame(ronen_frame, ronen_label, ronen_filename, is_lab=False)

    def train_test_split_videos(self, test_size=0.2, random_state=None):
        # Discretize the labels into bins
        bins = pd.cut(self.labels['x'], bins=10)

        # Generate a sequence of indices from 0 to num_videos - 1
        indices = list(range(self.num_videos))

        # Split the indices into training indices and test indices
        self.train_indices, self.test_indices = train_test_split(
            indices, test_size=test_size, stratify=bins, random_state=random_state)

        self.train_indices = self.train_indices
        self.test_indices = self.test_indices

    def load_and_preprocess(self, path):
        cap = cv2.VideoCapture(path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.crop_area:
                    frame = frame[self.crop_area[0]:self.crop_area[2], self.crop_area[1]:self.crop_area[3]]
                frames.append(frame)
            else:
                break

        cap.release()

        if len(frames) == 0:
            return None  # No frames were loaded

        # Check if num_frames is greater than the total number of frames
        if self.num_frames >= len(frames):
            averaged_frame = np.mean(frames, axis=0).astype(np.uint8)
            return [averaged_frame]

        # Split frames into segments, average them, and return
        num_segments = len(frames) // self.num_frames
        segments = [frames[i:i + self.num_frames] for i in range(0, len(frames), self.num_frames)]
        averaged_frames = [np.mean(segment, axis=0).astype(np.uint8) for segment in segments]

        return averaged_frames

    def get_label(self, filename):
        return self.labels.loc[filename, ['x', 'y', 'z']].values

    def save_processed_frame(self, frame, label, filename, is_lab):
        parts = filename.split('_')
        video_index = int(parts[3]) - 1
        if video_index in self.train_indices:
            target_dir = os.path.join(self.target_dir, "train", "lab" if is_lab else "ronen")
        elif video_index in self.test_indices:
            target_dir = os.path.join(self.target_dir, "test", "lab" if is_lab else "ronen")
        else:
            raise ValueError("Frame index not found in train or test indices")

        target_path = os.path.join(target_dir, filename)
        cv2.imwrite(target_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Save the corresponding label as a .npy file
        label_path = os.path.join(target_dir, f"{filename.split('.')[0]}_label.npy")
        np.save(label_path, label)