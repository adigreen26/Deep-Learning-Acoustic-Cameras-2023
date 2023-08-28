## Deep-Learning-Acoustic-Cameras-2023

Acoustic Insights into Bird Sounds: Utilizing Dual Acoustic Cameras for 3D Positioning Analysis.
Final Project in the course Deep Learning (05107255) by Raja Giryes.
Authors: Sarah Shitrit and Adi Green.


## Directory Structure
**`preprocesses/`**:
  - `video_synchronization.py`: Main script for video synchronization
  - `video_to_frames.py`: Takes synced videos and averages them to make a bigger dataset. Saves the dataset into new folders called "augmented".
  - `VideoPreprocessor.py`: Class to load and process videos to frames.
  - `utils.py`: Utility functions for all preprocesses

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
- `load_model_and_predict.py`: Loading pre-trained model, plot losses and predict


## Getting Started

1. **Setup Environment**: Ensure you have Python 3.10 installed. It's recommended to create a virtual environment for this project.
2. **Install Dependencies**: Install the required packages using `pip install -r requirements.txt` (Note: You'll need to create a `requirements.txt` file with all the necessary packages).
3. **Video Synchronization**: Before processing the data, ensure that the videos are synchronized. Use the `video_synchronization.py` script located in the `preprocesses/` directory outside of the `main_project` directory.
4. **Videos to Frames**: After synchronization, decide the configuration of the data and save it to folder. Use the `video_to_frames.py` script located in the `preprocesses/` directory outside of the `main_project` directory.
5. **Data Preparation**: Place your video data in the `data/` directory outside the `main_project` directory. Ensure the videos are in the appropriate subdirectories.
6. **Training**: To train the model, navigate to the `training/` directory and run `train.py`.
7. **Evaluation**: After training, you can evaluate the model's performance using the saved weights in the `saved_models/` directory using `load_model_and_predict.py` in `main_project/` directory. Place your models in the `saved_models/` directory inside `main_project/`.

## Data
The data is saved in [Google Drive Link](https://drive.google.com/drive/folders/14aMe9iEvR_HsLhjcINDB2XZdYrNwYngk?usp=sharing
). 

## Saved Models

The saved models are saved in [Google Drive Link](https://drive.google.com/drive/folders/1-FOlWwH1NtWjpx_2I4yfisFRTnYAjOdf?usp=sharing
). 
- These are the setting for the `video_to_frames.py` (num_frames), and dropout for the model.
For `num_frames=50` is data is in folder `augmentation`. 
For `num_frames=500` is data is in folder `averaged`.
  1. saved_data_best_model_cnn_droput01_aug50: 
     + `num_frames=50`
     + `dropout = 0.1`
     + `weight_decay = 0`
  2. saved_data_cnn_droput005_aug50:
     + `num_frames=50`
     + `dropout = 0.05`
     + `weight_decay = 0`
  3. saved_data_cnn_dropout005:
     + `num_frames=500`
     + `dropout = 0.05`
     + `weight_decay = 0`
  4. saved_data_cnn_nodropout_noregula:
     + `num_frames=500`
     + `dropout = 0`
     + `weight_decay = 0`
  5. saved_data_cnn_nodropout_regulaL2:
     + `num_frames=500`
     + `dropout = 0`
     + `weight_decay = 10e-4`



