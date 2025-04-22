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
