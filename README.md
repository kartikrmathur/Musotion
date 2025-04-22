<<<<<<< HEAD
# Facial Expression-Based Music Player

This application uses facial expression recognition to detect user emotions and play music that matches the detected mood.

## Project Overview

The application detects facial expressions through a webcam, classifies the emotion (happy, sad, angry, etc.), and automatically plays music that matches the detected mood from predefined playlists.

## Implementation Phases

### Phase 1: Core Functionality (MVP) - Complete ✓

- **Facial Expression Detection**

  - Uses OpenCV and a custom trained deep learning model to detect emotions
  - Recognizes 7 emotions: happy, sad, neutral, angry, surprised, fear, and disgust
  - Processes webcam feed at 5-10 FPS

- **Music Player**

  - Uses Pygame for audio playback
  - Plays music from predefined emotion-based playlists
  - Basic controls: play/pause, next/previous track

- **User Interface**
  - Shows live webcam feed with emotion overlay
  - Displays current song information
  - Simple controls for music playback

### Phase 2: Enhanced Features - Planned

- **Advanced Emotion Detection**

  - Improve accuracy with additional preprocessing
  - Add confidence threshold for more stability
  - Use facial landmarks for better feature extraction

- **Custom Playlists**

  - Settings menu to assign music folders to emotions
  - Support for various audio formats
  - Auto-scan music library for compatible files

- **Rich GUI**

  - Album art display
  - Progress bar and volume control
  - Visualization of emotion detection confidence

- **Performance Optimization**
  - Multi-threading for smooth video and audio processing
  - Optimize frame rate and resolution settings
  - Reduce resource usage

### Phase 3: Advanced Features - Future

- **User Preferences**

  - User mood tracking and visualization
  - Personalized music recommendations based on past emotions
  - Settings persistence

- **Integration**

  - Export playlists to standard formats
  - Integration with music streaming services
  - Share functionality

- **Cross-Platform Support**
  - Package as standalone application
  - Web interface option
  - Mobile compatibility

## Setup and Requirements

### Prerequisites

- Python 3.8 or higher
- Webcam

### Required Packages

```bash
pip install tensorflow keras pandas numpy pygame opencv-python imutils deepface
```

### Directory Structure

```
music/
├── angry/      # MP3 files for angry emotion
├── disgust/    # MP3 files for disgust emotion
├── fear/       # MP3 files for fear emotion
├── happy/      # MP3 files for happy emotion
├── neutral/    # MP3 files for neutral emotion
├── sad/        # MP3 files for sad emotion
└── surprise/   # MP3 files for surprise emotion
```

## Usage

1. Run the application:

   ```bash
   python Expression_Based_music_player.py
   ```

2. Controls:
   - **SPACE**: Play/pause music
   - **N**: Next song
   - **P**: Previous song
   - **Q**: Quit

## Implementation Details

### Emotion Detection Model

The model uses a Convolutional Neural Network (CNN) trained on the FER2013 dataset, which contains grayscale images of faces labeled with emotions.

### Music Player

The music player uses Pygame's mixer module to handle audio playback. It automatically selects songs from the appropriate emotion folder based on detected facial expressions.

## Troubleshooting

- If webcam doesn't initialize, check camera connection and permissions
- If music doesn't play, ensure MP3 files are placed in the correct emotion folders
- For model errors, verify the trained model file exists in the output_model directory
=======
# Musotion - Emotion Recognition & Music Player

A Python application that recognizes facial emotions using deep learning and plays music based on detected emotions.

## Project Components

### 1. Emotion Recognition Model (emotionRecTrain.py)

Python script using Keras + TensorFlow to train a custom machine learning model to recognize emotions from facial images.

#### Training

Required arguments:

- `csv_file` - path to the FER2013 dataset CSV file
- `export_path` - path to save the trained model artifacts

Optional arguments:

- `batch_size` - batch size during training
- `n_epochs` - number of training epochs
- `debug` - Will override script arguments for batch_size and n_epochs to 10 and 1

Example commands:

```bash
# Training with sample dataset in debug mode
python emotionRecTrain.py --csv_file=sample_fer2013.csv --export_path=output_model --debug

# Training with full dataset
python emotionRecTrain.py --csv_file=path_to_fer2013.csv --export_path=output_model

# Setting custom parameters
python emotionRecTrain.py --csv_file=path_to_fer2013.csv --export_path=output_model --batch_size=50
python emotionRecTrain.py --csv_file=path_to_fer2013.csv --export_path=output_model --n_epochs=100
```

### 2. Emotion Demo Application (emotion_demo.py)

A diagnostic tool that checks your system setup and demonstrates data preprocessing.

```bash
python emotion_demo.py
```

### 3. Music Player Module (music_player.py)

A standalone module that manages music playback based on emotional states.

```bash
python music_player.py
```

This script can also be run directly to test music playback for each emotion.

### 4. Emotion-Based Music Player (emotion_music_player.py)

The main application that combines emotion recognition and music playback. It detects your emotion through the webcam and plays appropriate music.

```bash
python emotion_music_player.py
```

Controls:

- Press 'q' to quit
- Press '+' to increase volume
- Press '-' to decrease volume
- Press 'm' to mute/unmute

## Setup and Installation

1. Create a Python environment (recommended Python 3.10 or 3.11)
2. Install required packages:
   ```bash
   pip install --user tensorflow keras pandas numpy pygame opencv-python imutils
   ```
3. Download the FER2013 dataset or use the provided sample dataset
4. Organize your music files in the following structure:
   ```
   music/
   ├── angry/      # MP3 files for angry emotion
   ├── disgust/    # MP3 files for disgust emotion
   ├── fear/       # MP3 files for fear emotion
   ├── happy/      # MP3 files for happy emotion
   ├── neutral/    # MP3 files for neutral emotion
   ├── sad/        # MP3 files for sad emotion
   └── surprise/   # MP3 files for surprise emotion
   ```

## Running the Project

1. Train the emotion recognition model:

   ```bash
   python emotionRecTrain.py --csv_file=sample_fer2013.csv --export_path=output_model
   ```

2. Add your MP3 files to each emotion folder in the `music` directory

3. Run the emotion-based music player:
   ```bash
   python emotion_music_player.py
   ```

## Model Architecture

The emotion recognition model uses a VGG16 pre-trained model for feature extraction, followed by custom dense layers for emotion classification. The model is trained to recognize 7 different emotions from facial images.

## Dataset

The model uses the FER2013 dataset, which contains grayscale images of faces labeled with emotions. A sample version of the dataset is included in the project as `sample_fer2013.csv`.

## References

For more details about the emotion recognition model and architecture, read the article:
[Training a TensorFlow Model to Recognize Emotions](https://medium.com/@jsflo.dev/training-a-tensorflow-model-to-recognize-emotions-a20c3bcd6468)
>>>>>>> e845925b6297e856d3840686af965c6579551143
