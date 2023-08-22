## Deep-Learning-Acoustic-Cameras-2023

Acoustic Insights into Bird Sounds: Utilizing Dual Acoustic Cameras for 3D Positioning Analysis.
Final Project in the course Deep Learning (05107255) by Raja Giryes.
Authors: Sarah Shitrit and Adi Green.


## Directory Structure
**`video_synchronization/`**:
  - `video_synchronization.py`: Main script for video synchronization
  - `utils.py`: Utility functions for video synchronization

**`main_project/`**:
- `data_processing/`
  - `dataloader.py`: Contains the VideoDataset(Dataset) class and related data loading functions 
  - `data_helpers.py`: Contains helper functions like plot label distribution, display random frame, etc. 
- `models/` 
  - `cnn_frame.py`: Contains the CNNFrame model definition 
- `saved_models/`
  - `model.pth`: Pre-trained model weights 
- `training/` 
  - `train_helpers.py`: Contains training helper functions like early stopping, calculation of saliency maps, etc. 
  - `train.py`: Contains the main training loop and settings before training 
- `settings.py`: Configuration and settings for the project


## Getting Started

1. **Setup Environment**: Ensure you have Python 3.10 installed. It's recommended to create a virtual environment for this project.
2. **Install Dependencies**: Install the required packages using `pip install -r requirements.txt` (Note: You'll need to create a `requirements.txt` file with all the necessary packages).
3. **Video Synchronization**: Before processing the data, ensure that the videos are synchronized. Use the `video_synchronization.py` script located in the `video_synchronization/` directory outside of the `main_project` directory.
4. **Data Preparation**: Place your video data in the `data/` directory outside of the `main_project` directory. Ensure the videos are in the appropriate `Lab_webm` and `Ronen_webm` subdirectories.
5. **Training**: To train the model, navigate to the `training/` directory and run `train.py`.
6. **Evaluation**: After training, you can evaluate the model's performance using the saved weights in the `saved_models/` directory.

## Configuration

All project settings and configurations


