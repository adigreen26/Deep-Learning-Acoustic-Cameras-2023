import settings
from data_processing.dataloader import TrainVideoDataset, TestVideoDataset
from models.cnn_frame import CNNFrame
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
# Get the current working directory
cwd = os.getcwd()
# Append the relative path to preprocesses
sys.path.append(os.path.join(cwd, 'main_project'))


def predict_and_compute_loss(data_loader, model, criterion):
    model.eval()
    predictions = []
    test_losses = []
    true_labels = []

    with torch.no_grad():
        for lab_video, ronen_video, label in data_loader:
            lab_frame = lab_video.permute(0, 3, 1, 2).to(device).float()  # [batch, channels, height, width]
            ronen_frame = ronen_video.permute(0, 3, 1, 2).to(device).float()  # [batch, channels, height, width]
            label = label.to(device).float()  # Move label tensor to device and ensure it's float

            outputs = model(lab_frame, ronen_frame)
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(label.cpu().numpy())

            loss = criterion(outputs, label)
            test_losses.append(loss.item())

    return predictions, test_losses, true_labels


# Directory paths
target_dir = settings.TARGET_DIR
train_dir = settings.TRAIN_DIR
test_dir = settings.TEST_DIR
what_model = "saved_data_best_model_cnn_droput01_aug50" #name the folder of the model
model_dir = os.path.join(settings.MODEL_SAVE_DIR, what_model)

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

# Use GPU if available
device = settings.DEVICE
model = model.to(device)

with open(os.path.join(model_dir,"losses.txt"), "r") as f:
    lines = f.readlines()[8:]  # Skip metadata and header
    train_losses = [float(line.split("\t")[1]) for line in lines]
    test_losses = [float(line.split("\t")[2]) for line in lines]

best_epoch = test_losses.index(min(test_losses)) + 1
print(f"The best epoch based on test loss is: {best_epoch}")

# Assuming test_loader, model, and criterion are defined
test_predictions, test_losses, true_labels = predict_and_compute_loss(test_loader, model, criterion)
print("Test Losses:", np.mean(test_losses))

# Plot the losses
plt.figure(figsize=(5, 3))
plt.plot(train_losses[5:], label="Train Loss")
plt.plot(test_losses[5:], label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Train and Test Losses Over Epochs")
plt.show()