# Contains the main training loop, settings before training, etc.
import sys
import os
# Get the current working directory
cwd = os.getcwd()
# Append the relative path to preprocesses
sys.path.append(os.path.join(cwd, 'main_project'))

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from matplotlib.pyplot import imsave
import numpy as np
from data_processing.dataloader import TrainVideoDataset, TestVideoDataset
from models.cnn_frame import CNNFrame
from training.train_helpers import EarlyStopping, calculate_average_saliency_maps, plot_saliency_maps
import settings

# Directory paths
target_dir = settings.TARGET_DIR
train_dir = settings.TRAIN_DIR
test_dir = settings.TEST_DIR


# Create instances of the train and test dataset classes
train_dataset = TrainVideoDataset(train_dir)
test_dataset = TestVideoDataset(test_dir)

batch_size = settings.BATCH_SIZE
# Instantiate the data loader
# Create data loaders for train and test datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Train for CNNFrame
model = CNNFrame(num_classes=3)
criterion = nn.MSELoss()
lr = settings.LEARNING_RATE
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Instantiate the learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=settings.LR_DECAY_STEP_SIZE, gamma=settings.LR_DECAY_GAMMA)
early_stopping = EarlyStopping(tolerance=settings.EARLY_STOP_TOLERANCE, min_delta=settings.EARLY_STOP_MIN_DELTA)

# Number of epochs
num_epochs = settings.NUM_EPOCHS

# Use GPU if available
device = settings.DEVICE
model = model.to(device)

train_losses = []
test_losses = []
best_loss = float('inf')  # Initialize best loss as positive infinity

# Create directories to save the images
if not os.path.exists("saliency_maps"):
    os.mkdir("saliency_maps")
if not os.path.exists("frames"):
    os.mkdir("frames")

print("Training...")
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0

    for lab_video, ronen_video, label in train_loader:
        lab_frame = lab_video.permute(0, 3, 1, 2)  # [batch, channels, height, width]
        ronen_frame = ronen_video.permute(0, 3, 1, 2)  # [batch, channels, height, width]
        label = label.to(device).float()  # Move label tensor to device and ensure it's float
        lab_frame, ronen_frame = lab_frame.to(device).float(), ronen_frame.to(device).float()
        optimizer.zero_grad()
        outputs = model(lab_frame, ronen_frame)

        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Update learning rate scheduler at the end of each epoch
    scheduler.step()

    # Evaluation
    model.eval()
    total_test_loss = 0.0
    num_videos = len(test_dataset.videos)  # Number of videos

    for video_name, video_frames in test_dataset.videos.items():
        lab_frames, ronen_frames = zip(*video_frames)  # Unzip frames and labels
        label_file = os.path.splitext(video_frames[0][1])[0] + '_label.npy'
        label = np.load(os.path.join(test_dataset.ronen_dir, label_file))
        label = torch.tensor(label, dtype=torch.float32).to(device)

        lab_frames_tensors = []
        ronen_frames_tensors = []

        for lab_file, ronen_file in video_frames:
            # Load the images and perform preprocessing
            lab_image = cv2.imread(os.path.join(test_dataset.lab_dir, lab_file))
            ronen_image = cv2.imread(os.path.join(test_dataset.ronen_dir, ronen_file))

            lab_image = torch.tensor(lab_image, dtype=torch.float32).to(device).unsqueeze(0)
            ronen_image = torch.tensor(ronen_image, dtype=torch.float32).to(device).unsqueeze(0)

            lab_frames_tensors.append(lab_image)
            ronen_frames_tensors.append(ronen_image)

        lab_frames = torch.cat(lab_frames_tensors, dim=0).permute(0, 3, 1, 2)
        ronen_frames = torch.cat(ronen_frames_tensors, dim=0).permute(0, 3, 1, 2)

        # Forward pass through the model to get predictions
        outputs = model(lab_frames, ronen_frames)
        outputs = torch.mean(outputs, dim=0)  # get avrege prediction

        # Calculate the loss using the loaded label
        loss = criterion(outputs, label)
        total_test_loss += loss.item()  # Accumulate the loss

    avg_total_test_loss = total_test_loss / num_videos  # Calculate the average test loss

    # Save the model if it has the lowest test loss so far
    if avg_total_test_loss < best_loss:
        best_loss = avg_total_test_loss
        torch.save(model.state_dict(), 'best_model_cnn_droput005_aug50.pth')

    print(
        f"Epoch [{epoch + 1}/{num_epochs}],Train Loss: {avg_train_loss:.4f}, Test Loss (across all videos): {avg_total_test_loss:.4f}")

    # Plot saliency maps for specific epochs
    if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
        for lab_video, ronen_video, label in test_loader:
            lab_frame = lab_video.permute(0, 3, 1, 2).to(device).float()
            ronen_frame = ronen_video.permute(0, 3, 1, 2).to(device).float()
            # Take only the first 4 images in the batch
            lab_frame_subset = lab_frame[:4]
            ronen_frame_subset = ronen_frame[:4]

            # lab_saliency, ronen_saliency = calculate_saliency_maps(model, lab_frame, ronen_frame)
            lab_saliency, ronen_saliency = calculate_average_saliency_maps(model, lab_frame_subset, ronen_frame_subset)
            # Normalize the saliency maps for each video in the batch separately
            lab_saliency_normalized = np.zeros_like(lab_saliency)
            ronen_saliency_normalized = np.zeros_like(ronen_saliency)

            # Normalize the saliency maps for each video in the batch separately
            for i in range(lab_frame_subset.shape[0]):
                lab_saliency_normalized[i] = (lab_saliency[i] - lab_saliency[i].min()) / (
                            lab_saliency[i].max() - lab_saliency[i].min())
                ronen_saliency_normalized[i] = (ronen_saliency[i] - ronen_saliency[i].min()) / (
                            ronen_saliency[i].max() - ronen_saliency[i].min())

            # Normalize the frames to [0, 1]
            lab_frame_normalized = lab_frame_subset.cpu() / 255.0
            ronen_frame_normalized = ronen_frame_subset.cpu() / 255.0

            plot_saliency_maps(lab_frame_normalized, ronen_frame_normalized, lab_saliency_normalized,
                               ronen_saliency_normalized)

            # Inside the plotting section:
            for idx, (lab, ronen, lab_sal, ronen_sal) in enumerate(zip(lab_frame_normalized, ronen_frame_normalized,
                                                                       lab_saliency_normalized,
                                                                       ronen_saliency_normalized)):
                # Save frames
                imsave(os.path.join("frames", f"epoch_{epoch + 1}_lab_frame_{idx}.png"), lab.permute(1, 2, 0).numpy())
                imsave(os.path.join("frames", f"epoch_{epoch + 1}_ronen_frame_{idx}.png"),
                       ronen.permute(1, 2, 0).numpy())

                # Save saliency maps
                imsave(os.path.join("saliency_maps", f"epoch_{epoch + 1}_lab_saliency_{idx}.png"),
                       np.transpose(lab_sal, (1, 2, 0)))
                imsave(os.path.join("saliency_maps", f"epoch_{epoch + 1}_ronen_saliency_{idx}.png"),
                       np.transpose(ronen_sal, (1, 2, 0)))
            break

    # Update early stopping object
    if early_stopping(avg_train_loss, avg_total_test_loss):
        print("Early stopping triggered!")
        break

    torch.cuda.empty_cache()

print("Training finished!")






