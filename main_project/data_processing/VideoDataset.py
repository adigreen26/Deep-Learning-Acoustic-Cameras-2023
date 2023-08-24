# Contains the VideoDataset(Dataset) class and related data loading functions
import os
import cv2  # pip install opencv-python
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class VideoDataset(Dataset):
    def __init__(self, lab_dir, ronen_dir, labels_csv):
        self.lab_files = sorted(os.listdir(lab_dir))
        self.ronen_files = sorted(os.listdir(ronen_dir))
        self.lab_dir = lab_dir
        self.ronen_dir = ronen_dir

        # Load the labels from the CSV file
        self.labels = pd.read_excel(labels_csv, usecols=range(4)).set_index('filename')
        # Convert the labels to floats
        self.labels = self.labels.astype(float)

    def __len__(self):
        return len(self.lab_files)

    def __getitem__(self, idx):
        lab_file = self.lab_files[idx]
        ronen_file = self.ronen_files[idx]

        # Check that the files are correctly synchronized
        assert lab_file.split('_')[-1] == ronen_file.split('_')[-1], \
            f"Files are not synchronized: {lab_file}, {ronen_file}"

        # Load the videos using OpenCV
        lab_video = self.load_and_crop_video(os.path.join(self.lab_dir, lab_file))
        ronen_video = self.load_and_crop_video(os.path.join(self.ronen_dir, ronen_file))

        # Get the label for this video
        label = self.labels.loc[ronen_file, ['x', 'y', 'z']].values

        # Print out the label array
        #print(f"Label for {ronen_file}: {label}")

        label = torch.tensor(label, dtype=torch.float)

        return lab_video, ronen_video, label

    def train_test_split_videos(self, test_size=0.2, random_state=None):
        # Discretize the labels into bins
        bins = pd.cut(self.labels['x'], bins=10)

        # Generate a sequence of indices from 0 to len(self) - 1
        indices = list(range(len(self)))

        # Split the indices into training indices and test indices
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, stratify=bins, random_state=random_state)

        # Create two new datasets: one for training and one for testing
        train_dataset = torch.utils.data.Subset(self, train_indices)
        test_dataset = torch.utils.data.Subset(self, test_indices)

        return train_dataset, test_dataset

    def load_and_crop_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []

        # Define the crop area (top, left, bottom, right)
        top, left, bottom, right = 0, 0, 285, 385

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                # Convert the frame from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Crop the frame
                frame = frame[top:bottom, left:right]
                # Append the cropped frame
                frames.append(frame)
            else:
                break

        cap.release()

        # Convert the list of frames to a single numpy array
        frames = np.array(frames)

        # Calculate the average of all frames
        average_frame = np.mean(frames, axis=0).astype(np.uint8)

        # Convert the averaged frame from BGR to RGB and to float32
        average_frame_rgb = cv2.cvtColor(average_frame, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Convert the averaged frame to a PyTorch tensor
        average_frame_tensor = torch.from_numpy(average_frame_rgb)

        return average_frame_tensor