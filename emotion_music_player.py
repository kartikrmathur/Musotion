import cv2
import numpy as np
import os
import imutils
import time
import pygame
from tensorflow import keras

import operator
from music_player import MusicPlayer

# Check if the haarcascade file exists in the expected location
cascade_paths = [
    'haarcascade_frontalface_alt2.xml',  # Current directory
    os.path.join('Musotion', 'haarcascade_frontalface_alt2.xml'),  # Musotion subdirectory
    os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_alt2.xml')  # Script directory
]

cascade_file = None
for path in cascade_paths:
    if os.path.exists(path):
        cascade_file = path
        break
    
# Make sure the cascade file exists
if not cascade_file:
    print("Error: Could not find haarcascade_frontalface_alt2.xml")
    print("Please download the file from: https://github.com/opencv/opencv/tree/master/data/haarcascades")
    exit(1)

print(f"Using cascade file: {cascade_file}")

# Initialize face cascade classifier
try:
    faceCascade = cv2.CascadeClassifier(cascade_file)
except Exception as e:
    print(f"Error loading cascade classifier: {e}")
    exit(1)

# Load the emotion recognition model
model_paths = [
    # Look for the specific model file
    os.path.join('output_model', 'emotion_model.keras'),
    os.path.join('output_model', 'emotion_model.h5'),
    # Try the directory as fallback
    os.path.join('output_model', 'emotion_model'),
    'output_model',
    # Musotion subdirectory
    os.path.join('Musotion', 'output_model', 'emotion_model.keras'),
    os.path.join('Musotion', 'output_model', 'emotion_model.h5'),
    os.path.join('Musotion', 'output_model')
]

model_path = None
for path in model_paths:
    if os.path.exists(path):
        model_path = path
        break

# Check if model exists, if not, instruct user to train it
if not model_path:
    print("Error: Model file not found.")
    print("Please train the model first using:")
    print("python simple_model.py --csv_file=sample_fer2013_converted.csv --export_path=output_model --debug")
    exit(1)

print(f"Using model path: {model_path}")

try:
    # Try to load the model from the specified path
    if model_path.endswith('.keras') or model_path.endswith('.h5'):
        # Load as a regular Keras model
        model = keras.models.load_model(model_path)
    else:
        # Try TFSMLayer for SavedModel format
        model = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please train the model first using:")
    print("python simple_model.py --csv_file=sample_fer2013_converted.csv --export_path=output_model --debug")
    exit(1)

# Initialize music player
music_player = MusicPlayer()

# Define emotion labels
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
font = cv2.FONT_HERSHEY_SIMPLEX

# Function to get emoji image path
def get_emoji_path(emotion):
    emoji_paths = [
        os.path.join('emoji', f"{emotion}.png"),  # Current directory
        os.path.join('Musotion', 'emoji', f"{emotion}.png"),  # Musotion subdirectory
        os.path.join(os.path.dirname(__file__), 'emoji', f"{emotion}.png")  # Script directory
    ]
    
    # Try each path
    for emoji_path in emoji_paths:
        if os.path.exists(emoji_path):
            try:
                emoji_img = cv2.imread(emoji_path)
                # Verify image was loaded correctly
                if emoji_img is not None and emoji_img.size > 0:
                    return emoji_img
                else:
                    print(f"Warning: Could not load emoji image: {emoji_path}")
            except Exception as e:
                print(f"Error loading emoji image {emoji_path}: {e}")
    
    # If emoji directory doesn't exist, create it
    emoji_dir = os.path.join(os.path.dirname(__file__), 'emoji')
    if not os.path.exists(emoji_dir):
        try:
            os.makedirs(emoji_dir, exist_ok=True)
            print(f"Created emoji directory: {emoji_dir}")
        except Exception as e:
            print(f"Warning: Could not create emoji directory: {e}")
    
    print(f"Emoji for {emotion} not found, using placeholder")
    # Create a placeholder colored image based on emotion
    placeholder = np.zeros((150, 150, 3), dtype=np.uint8)
    
    # Set color based on emotion
    if emotion == 'happy':
        placeholder[:] = (0, 255, 255)  # Yellow
    elif emotion == 'sad':
        placeholder[:] = (255, 0, 0)    # Blue
    elif emotion == 'angry':
        placeholder[:] = (0, 0, 255)    # Red
    elif emotion == 'fear':
        placeholder[:] = (255, 0, 255)  # Purple
    elif emotion == 'disgust':
        placeholder[:] = (0, 255, 0)    # Green
    elif emotion == 'surprise':
        placeholder[:] = (0, 255, 255)  # Yellow
    else:  # neutral
        placeholder[:] = (200, 200, 200)  # Gray
        
    # Write emotion text
    cv2.putText(placeholder, emotion, (10, 75), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return placeholder

def draw_music_controls(frame, width, height):
    """Draw music player controls UI on the frame"""
    # Get song info
    song_info = music_player.get_current_song_info()
    
    # Background for controls
    control_height = 80
    cv2.rectangle(frame, (0, height-control_height), (width, height), (0, 0, 0), -1)
    
    # Draw separator line
    cv2.line(frame, (0, height-control_height), (width, height-control_height), (50, 50, 50), 2)
    
    # Display song information
    if song_info:
        # Song title and status
        status_text = f"{song_info['status']}: {song_info['file_name']}"
        cv2.putText(frame, status_text, (10, height-control_height+20), 
                    font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Playlist position
        position_text = f"Song {song_info['index']}/{song_info['total']}"
        cv2.putText(frame, position_text, (10, height-control_height+40), 
                    font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Time position (approximate)
        time_text = f"Position: {song_info['position']:.1f}s"
        cv2.putText(frame, time_text, (10, height-control_height+60), 
                    font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "No music playing", (10, height-control_height+30), 
                    font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Display controls
    controls_text = "Controls: SPACE=Play/Pause | N=Next | P=Previous | F=Forward | B=Backward | Q=Quit"
    cv2.putText(frame, controls_text, (width//2 - 300, height-10), 
                font, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

def main():
    # Initialize webcam
    print("Initializing webcam...")
    
    # Try multiple camera indices if the first one fails
    webcam_initialized = False
    video_capture = None
    
    # Try camera indices 0, 1, and 2
    for camera_index in range(3):
        try:
            video_capture = cv2.VideoCapture(camera_index)
            if video_capture.isOpened():
                print(f"Webcam initialized successfully on camera index {camera_index}")
                webcam_initialized = True
                break
            else:
                video_capture.release()  # Release before trying the next one
        except Exception as e:
            print(f"Error trying camera index {camera_index}: {e}")
    
    if not webcam_initialized or video_capture is None:
        print("Error: Could not initialize any webcam.")
        print("Please check your camera connection and permissions.")
        return
    
    # Dictionary to keep track of detected emotions
    emotion_stats = {emotion: 0 for emotion in emotions}
    current_emotion = None
    last_emotion_change = time.time()
    
    # Min frames to detect emotion before changing music
    min_frames_to_confirm = 5
    emotion_confirmed = False
    
    # Get webcam frame dimensions
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("\nMusic Player Controls:")
    print("  - SPACE: Play/Pause music")
    print("  - N: Next song")
    print("  - P: Previous song")
    print("  - F: Skip forward 10 seconds")
    print("  - B: Skip backward 10 seconds")
    print("  - +/-: Increase/decrease volume")
    print("  - M: Mute/unmute")
    print("  - Q: Quit")
    print("\nLooking for your face...")
    
    # Main loop
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            
            if not ret or frame is None:
                print("Error: Failed to capture frame from webcam.")
                print("Attempting to reconnect...")
                # Try to reconnect to the webcam
                video_capture.release()
                video_capture = cv2.VideoCapture(0)
                if not video_capture.isOpened():
                    print("Failed to reconnect to webcam. Exiting.")
                    break
                continue
                
            # Resize the frame to improve speed
            try:
                frame = imutils.resize(frame, width=640)
            except Exception as e:
                print(f"Error resizing frame: {e}")
                continue
            
            # Get dimensions for UI
            height, width = frame.shape[:2]
            
            # Create a copy for displaying results
            display_frame = frame.copy()
            
            # Convert to gray-scale for face detection
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                print(f"Error converting frame to grayscale: {e}")
                continue
            
            # Detect faces
            try:
                faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
            except Exception as e:
                print(f"Error in face detection: {e}")
                faces = []
            
            # Process detected faces
            for (x, y, w, h) in faces:
                # Draw rectangle around the face
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                try:
                    # Extract and preprocess face
                    if y >= 0 and x >= 0 and y+h <= frame.shape[0] and x+w <= frame.shape[1]:
                        face_crop = frame[y:y+h, x:x+w]
                        face_crop = cv2.resize(face_crop, (48, 48))
                        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                        # Reshape and normalize as done in simple_model.py
                        face_crop = face_crop.astype('float32') / 255.0
                        face_input = np.zeros((1, 48, 48, 1))
                        face_input[0, :, :, 0] = face_crop
                        
                        # Predict emotion
                        predictions = model.predict(face_input, verbose=0)
                        emotion_idx = np.argmax(predictions)
                        detected_emotion = emotions[emotion_idx]
                        confidence = float(predictions[0][emotion_idx] * 100)
                        
                        # Update emotion counter
                        emotion_stats[detected_emotion] += 1
                        
                        # Display emotion and confidence on the frame
                        label = f"{detected_emotion}: {confidence:.1f}%"
                        cv2.putText(display_frame, label, (x, y-10), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        print("Warning: Face coordinates out of bounds")
                except Exception as e:
                    print(f"Error processing face: {e}")
                    continue
            
            # Find the most common emotion
            if sum(emotion_stats.values()) > 0:
                dominant_emotion = max(emotion_stats.items(), key=operator.itemgetter(1))[0]
                
                # Display stats
                stat_y = 30
                cv2.putText(display_frame, "Emotion Stats:", (10, stat_y), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                stat_y += 25
                
                for emotion, count in emotion_stats.items():
                    if count > 0:
                        percentage = count / sum(emotion_stats.values()) * 100
                        cv2.putText(display_frame, f"{emotion}: {percentage:.1f}%", 
                                    (10, stat_y), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                        stat_y += 25
                
                # Display dominant emotion
                cv2.putText(display_frame, f"Dominant: {dominant_emotion}", 
                            (10, stat_y), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                
                # Get emoji for the dominant emotion
                emoji_img = get_emoji_path(dominant_emotion)
                if isinstance(emoji_img, np.ndarray):
                    # Resize emoji to fit the display
                    emoji_img = imutils.resize(emoji_img, width=150)
                    
                    # Display emoji image
                    cv2.imshow('Emotion', emoji_img)
                    
                    # Check if Emotion window was closed
                    if cv2.getWindowProperty('Emotion', cv2.WND_PROP_VISIBLE) < 1:
                        cv2.namedWindow('Emotion', cv2.WINDOW_NORMAL)
                
                # Change music if the emotion has been stable for a while
                if (current_emotion != dominant_emotion and 
                    emotion_stats[dominant_emotion] >= min_frames_to_confirm):
                    
                    current_time = time.time()
                    # Only change music if it's been at least 5 seconds since the last change
                    if current_time - last_emotion_change >= 5.0:
                        print(f"\nChanging to {dominant_emotion} music...")
                        music_player.play_music(dominant_emotion)
                        current_emotion = dominant_emotion
                        last_emotion_change = current_time
                        
                        # Reset emotion counters after changing music
                        emotion_stats = {emotion: 0 for emotion in emotions}
            
            # Draw music controls UI
            draw_music_controls(display_frame, width, height)
            
            # Display the frame
            cv2.imshow('Emotion Recognition & Music Player', display_frame)
            
            # Check if window was closed (X button clicked)
            if cv2.getWindowProperty('Emotion Recognition & Music Player', cv2.WND_PROP_VISIBLE) < 1:
                print("\nWindow closed by user")
                break
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            # Quit on 'q'
            if key == ord('q'):
                break
                
            # Play/Pause toggle (spacebar)
            elif key == 32:  # spacebar
                music_player.toggle_pause()
                
            # Next song
            elif key == ord('n'):
                music_player.next_song()
                
            # Previous song
            elif key == ord('p'):
                music_player.previous_song()
                
            # Skip forward 10 seconds
            elif key == ord('f'):
                music_player.skip_forward(10)
                
            # Skip backward 10 seconds
            elif key == ord('b'):
                music_player.skip_backward(10)
                
            # Volume controls
            elif key == ord('+'):
                # Get current volume and increase it
                current_volume = pygame.mixer.music.get_volume()
                music_player.set_volume(min(1.0, current_volume + 0.1))
                print(f"Volume increased to {pygame.mixer.music.get_volume():.1f}")
                
            elif key == ord('-'):
                # Get current volume and decrease it
                current_volume = pygame.mixer.music.get_volume()
                music_player.set_volume(max(0.0, current_volume - 0.1))
                print(f"Volume decreased to {pygame.mixer.music.get_volume():.1f}")
                
            elif key == ord('m'):
                # Toggle mute
                current_volume = pygame.mixer.music.get_volume()
                if current_volume > 0:
                    # Save current volume before muting
                    if not hasattr(music_player, 'pre_mute_volume'):
                        music_player.pre_mute_volume = current_volume
                    music_player.set_volume(0.0)
                    print("Muted")
                else:
                    # Restore previous volume or set to default
                    restore_volume = getattr(music_player, 'pre_mute_volume', 0.5)
                    music_player.set_volume(restore_volume)
                    print(f"Unmuted (Volume: {restore_volume:.1f})")
    
    finally:
        # Clean up
        print("\nClosing application...")
        video_capture.release()
        cv2.destroyAllWindows()
        music_player.stop_music()
        print("Done!")

if __name__ == "__main__":
    main() 