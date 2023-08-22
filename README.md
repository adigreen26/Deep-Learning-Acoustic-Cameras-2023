## Deep-Learning-Acoustic-Cameras-2023

Acoustic Insights into Bird Sounds: Utilizing Dual Acoustic Cameras for 3D Positioning Analysis. Final Project in the course Deep Learning (05107255) by Raja Giryes. Authors: Sarah Shitrit and Adi Green.

### Components/Steps:

#### 1. Video Synchronization:
- **Purpose**: 
  - Synchronize videos based on their audio tracks.
  
- **Key Functions**:
  - `get_audio_frame_rate`: Extracts audio frame rate from a video.
  - `load_video_and_extract_audio`: Loads a video and extracts its audio.
  - `convert_to_mono`: Converts stereo audio to mono.
  - `calculate_cross_correlation`: Computes the cross-correlation between two audio signals.
  - `find_offset`: Determines the offset between two audio signals based on their cross-correlation.
  - `butter_bandpass`: Designs a bandpass filter using the Butterworth method.
  - `butter_bandpass_filter`: Filters an audio signal using a bandpass filter.
  - `get_num_sample`: Extracts the sample number from a filename.
  - `get_sorted_files`: Returns a sorted list of files in a directory based on sample number.
  
- **Process**:
  1. Load videos and extract audio.
  2. Convert stereo audio to mono.
  3. Normalize audio amplitude.
  4. Apply a bandpass filter.
  5. Calculate cross-correlation to find offset.
  6. Adjust video start time based on offset.
  7. Write synced videos to new files.


## Usage:
    How can someone use your project? Any specific instructions or commands they should know?

## Dependencies:    
    List of libraries or tools used in the project.



- 
