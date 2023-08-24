# Contains hyperparameters, paths, and other configurations
import os
import torch

# Directory paths
models_path = r'.'  # This will point to the current directory, which is the root of the project
TARGET_DIR = os.path.join(models_path, "data", "augmentation")
TRAIN_DIR = os.path.join(TARGET_DIR, "train")
TEST_DIR = os.path.join(models_path, "test")


# Hyperparameters
LEARNING_RATE = 0.01  # Learning rate for the optimizer
BATCH_SIZE = 4  # Number of samples per batch
NUM_EPOCHS = 70  # Total number of training epochs
DROPOUT_RATE = 0.05  # Dropout rate for regularization

# Early stopping parameters
EARLY_STOP_TOLERANCE = 5  # Number of epochs with no improvement after which training will be stopped
EARLY_STOP_MIN_DELTA = 1  # Minimum change in the monitored quantity to qualify as an improvement

# Learning rate scheduler parameters
LR_DECAY_STEP_SIZE = 5  # Period of learning rate decay
LR_DECAY_GAMMA = 0.5  # Multiplicative factor of learning rate decay

# Model parameters
NUM_CLASSES = 3  # Number of output classes
INPUT_SIZE = (285, 385, 3)  # Dimensions of the input data (height, width, channels)

# Logging settings
LOG_FREQUENCY = 10  # Log every 10 batches, for example

# Device settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use CUDA if available, else use CPU

# Miscellaneous settings
RANDOM_SEED = 42  # Seed for random number generation for reproducibility
TEST_SPLIT_SIZE = 0.2  # Fraction of data to be used as test set

# Directory for saving trained models
MODEL_SAVE_DIR = os.path.join(models_path, "saved_models")

