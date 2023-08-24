# Contains the VideoDataset(Dataset) class and related data loading functions
import os
import cv2  # pip install opencv-python
import numpy as np
import os
import cv2
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, data_dir):
        self.lab_dir = os.path.join(data_dir, "lab")
        self.ronen_dir = os.path.join(data_dir, "ronen")
        self.lab_files = sorted([file for file in os.listdir(self.lab_dir) if file.endswith(".jpg")])
        self.ronen_files = sorted([file for file in os.listdir(self.ronen_dir) if file.endswith(".jpg")])

    def __len__(self):
        return len(self.lab_files)

    def __getitem__(self, idx):
        lab_file = self.lab_files[idx]
        ronen_file = self.ronen_files[idx]

        # Check that the files are correctly synchronized
        assert lab_file.split('_')[3] == ronen_file.split('_')[3], \
            f"Files are not synchronized: {lab_file}, {ronen_file}"

        # Load the images using OpenCV
        lab_image = cv2.imread(os.path.join(self.lab_dir, lab_file))
        ronen_image = cv2.imread(os.path.join(self.ronen_dir, ronen_file))

        # Load corresponding label
        label_file = os.path.splitext(ronen_file)[0] + '_label.npy'

        label = np.load(os.path.join(self.ronen_dir, label_file))

        # lab_image = torch.tensor(lab_image, dtype=torch.float32).permute(2, 0, 1)  # Permute for PyTorch format
        # ronen_image = torch.tensor(ronen_image, dtype=torch.float32).permute(2, 0, 1)  # Permute for PyTorch format
        label = torch.tensor(label, dtype=torch.float32)

        # Normalize images if needed (e.g., using ImageNet mean and std)
        # lab_image = normalize_lab_image(lab_image)
        # ronen_image = normalize_ronen_image(ronen_image)

        return lab_image, ronen_image, label

    def group_frames_by_video(self):
        videos = {}
        for lab_file, ronen_file in zip(self.lab_files, self.ronen_files):
            video_name = lab_file.split('_')[3]
            if video_name not in videos:
                videos[video_name] = []
            videos[video_name].append((lab_file, ronen_file))
        return videos


class TrainVideoDataset(VideoDataset):
    def __init__(self, data_dir):
        super().__init__(data_dir)
        # You can further customize the dataset for training if needed


class TestVideoDataset(VideoDataset):
    def __init__(self, data_dir):
        super().__init__(data_dir)
        self.videos = self.group_frames_by_video()

    def evaluate_videos(self, model):
        video_losses = []
        for video_name, video_frames in self.videos.items():
            video_loss = self.evaluate_video(video_frames, model)
            video_losses.append(video_loss)
        avg_video_loss = sum(video_losses) / len(video_losses)
        return avg_video_loss

    def evaluate_video(self, video_frames, model):
        # video_frames is a list of frames belonging to the video
        video_losses = []
        for lab_file, ronen_file in video_frames:
            lab_image = cv2.imread(os.path.join(self.lab_dir, lab_file))
            ronen_image = cv2.imread(os.path.join(self.ronen_dir, ronen_file))

            # Calculate loss for this frame and add to video_losses
            loss = calculate_loss(model, lab_image, ronen_image)
            video_losses.append(loss)

        # Calculate average loss for the entire video
        avg_video_loss = sum(video_losses) / len(video_losses)

        return avg_video_loss