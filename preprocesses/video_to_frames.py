import sys
import os
from VideoPreprocessor import VideoPreprocessor
from utils import (
    plot_label_and_image,
    list_folder_contents_sorted
)
# Get the current working directory
cwd = os.getcwd()
# Append the relative path to preprocesses
sys.path.append(os.path.join(cwd, 'preprocesses'))

num_frames = 50 # number of frames you avrege on
top, left, bottom, right = 0, 0, 285, 385

# Update these paths and parameters
models_path = r'.'  # This will point to the current directory, which is the root
lab_source_dir = os.path.join("data", "lab_synced")
ronen_source_dir = os.path.join("data", "ronen_synced")
target_dir = os.path.join("data", "augmentation")
labels_csv = os.path.join("data", "labels.xlsx")


crop_area = (top, left, bottom, right)  # Set crop area or leave as None
# Create an instance of the VideoPreprocessor class
preprocessor = VideoPreprocessor(
    lab_source_dir=lab_source_dir,
    ronen_source_dir=ronen_source_dir,
    target_dir=target_dir,
    labels_csv=labels_csv,
    num_frames=num_frames,
    crop_area=crop_area
)

# Run the preprocessing
preprocessor.num_videos = len(os.listdir(preprocessor.lab_source_dir))
preprocessor.train_test_split_videos(test_size=0.2, random_state=42)
preprocessor.preprocess_videos()

visualize = False
# If you want to visualize
if visualize:
    target_dir = os.path.join("data", "augmentation")
    # Replace 'path_to_folder' with the actual path of the folder you want to list
    folder_path = os.path.join(target_dir, "test", "lab")
    file_name = "lab_sync_video_104_04"
    image_path = os.path.join(folder_path, f"{file_name}.jpg")
    label_path = os.path.join(folder_path, f"{file_name}_label.npy")
    plot_label_and_image(image_path, label_path)

    # Replace 'path_to_folder' with the actual path of the folder you want to list
    folder_path = os.path.join(target_dir, "test", "ronen")
    file_name = "ronen_sync_video_104_04"
    image_path = os.path.join(folder_path, f"{file_name}.jpg")
    label_path = os.path.join(folder_path, f"{file_name}_label.npy")
    plot_label_and_image(image_path, label_path)