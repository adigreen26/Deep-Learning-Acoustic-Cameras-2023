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
from data_processing.VideoDataset import VideoDataset
from models.cnn_frame import CNNFrame
from training.train_helpers import EarlyStopping, calculate_average_saliency_maps, plot_saliency_maps
import settings

lab_dir = settings.LAB_DIR
ronen_dir = settings.RONEN_DIR
labels_csv = settings.LABELS_CSV

# Instantiate the dataset
dataset = VideoDataset(lab_dir, ronen_dir, labels_csv)
# Split the dataset into a training set and a test set
train_dataset, test_dataset = dataset.train_test_split_videos(test_size=settings.TEST_SPLIT_SIZE, random_state=settings.RANDOM_SEED)

batch_size = settings.BATCH_SIZE
# Instantiate the data loader
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
    with torch.no_grad():
        for lab_video, ronen_video, label in test_loader:  # Using test_loader for test data
            lab_frame = lab_video.permute(0, 3, 1, 2)  # [batch, channels, height, width]
            ronen_frame = ronen_video.permute(0, 3, 1, 2)  # [batch, channels, height, width]
            label = label.to(device).float()  # Move label tensor to device and ensure it's float
            lab_frame, ronen_frame = lab_frame.to(device).float(), ronen_frame.to(device).float()
            outputs = model(lab_frame, ronen_frame)
            loss = criterion(outputs, label)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    # Save the model if it has the lowest test loss so far
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        model_save_path = os.path.join(settings.MODEL_SAVE_DIR, 'best_model_cnn_droput005.pth')
        torch.save(model.state_dict(), model_save_path)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

    # Plot saliency maps for specific epochs
    if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
        for lab_video, ronen_video, label in test_loader:
            lab_frame = lab_video.permute(0, 3, 1, 2).to(device).float()
            ronen_frame = ronen_video.permute(0, 3, 1, 2).to(device).float()

            # lab_saliency, ronen_saliency = calculate_saliency_maps(model, lab_frame, ronen_frame)
            lab_saliency, ronen_saliency = calculate_average_saliency_maps(model, lab_frame, ronen_frame)
            # Normalize the saliency maps for each video in the batch separately
            lab_saliency_normalized = np.zeros_like(lab_saliency)
            ronen_saliency_normalized = np.zeros_like(ronen_saliency)

            # Normalize the saliency maps for each video in the batch separately
            for i in range(lab_frame.shape[0]):
                lab_saliency_normalized[i] = (lab_saliency[i] - lab_saliency[i].min()) / (
                            lab_saliency[i].max() - lab_saliency[i].min())
                ronen_saliency_normalized[i] = (ronen_saliency[i] - ronen_saliency[i].min()) / (
                            ronen_saliency[i].max() - ronen_saliency[i].min())

            # Normalize the frames to [0, 1]
            lab_frame_normalized = lab_frame.cpu() / 255.0
            ronen_frame_normalized = ronen_frame.cpu() / 255.0

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
    if early_stopping(avg_train_loss, avg_test_loss):
        print("Early stopping triggered!")
        break

    torch.cuda.empty_cache()

print("Training finished!")






