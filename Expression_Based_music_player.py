import cv2
import numpy as np
import os
import time
import pygame
from tensorflow import keras
import threading
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import io
import random
import mutagen
from music_settings import MusicSettings
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mediapipe as mp

class EmotionMusicPlayer:
    def __init__(self):
        # Initialize pygame for music playback
        pygame.mixer.init()
        
        # Load settings
        self.settings = MusicSettings()
        
        # Initialize face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        
        # Also initialize cascade as backup
        cascade_file = 'haarcascade_frontalface_alt2.xml'
        if not os.path.exists(cascade_file):
            raise FileNotFoundError(f"Cascade file {cascade_file} not found. Please download it from OpenCV repository.")
        self.face_cascade = cv2.CascadeClassifier(cascade_file)
        
        # Load the emotion recognition model
        model_path = os.path.join('output_model', 'emotion_model.keras')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Please train the model first.")
        self.model = keras.models.load_model(model_path)
        
        # Define emotion labels (matching the model's output)
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Initialize music directory structure
        self.music_root = "music"
        if not os.path.exists(self.music_root):
            os.makedirs(self.music_root, exist_ok=True)
            # Create subdirectories for each emotion
            for emotion in self.emotions:
                os.makedirs(os.path.join(self.music_root, emotion), exist_ok=True)
            print(f"Created music directories at {self.music_root}. Please add MP3 files for each emotion.")
        
        # Music playback variables
        self.current_emotion = None
        self.current_playlist = []
        self.current_index = -1
        self.playing = False
        self.paused = False
        self.music_thread = None
        self.stop_event = threading.Event()
        self.current_position = 0
        self.song_length = 0
        
        # Mood tracking
        self.mood_history = []
        self.last_mood_time = None
        
        # Set initial volume
        pygame.mixer.music.set_volume(self.settings.get_volume())
        
        # Initialize GUI elements
        self.root = None
        self.cam_frame = None
        self.album_art = None
        self.progress_bar = None
        self.progress_var = None
        self.volume_var = None
        self.current_song_label = None
        self.emotion_label = None
        self.emotion_confidence = None
    
    def get_songs_for_emotion(self, emotion):
        """Get list of songs for a specific emotion"""
        # Get custom folder for this emotion from settings
        emotion_folder = self.settings.get_folder_for_emotion(emotion)
        if not os.path.exists(emotion_folder):
            print(f"Warning: Folder for {emotion} not found. Creating it now.")
            os.makedirs(emotion_folder, exist_ok=True)
            return []
        
        # Get supported formats from settings
        formats = self.settings.get_supported_formats()
        
        # Find all matching files
        songs = []
        for f in os.listdir(emotion_folder):
            if any(f.lower().endswith(fmt) for fmt in formats):
                songs.append(os.path.join(emotion_folder, f))
        
        if not songs:
            print(f"No songs found for {emotion}. Please add audio files to {emotion_folder}")
        
        return songs
    
    def play_music_for_emotion(self, emotion):
        """Play music based on detected emotion"""
        if emotion == self.current_emotion and (self.playing or self.paused):
            # Already playing the right emotion music
            return
        
        # Stop current playback if any
        self.stop_music()
        
        # Record this emotion in history
        now = datetime.now()
        self.mood_history.append((now, emotion))
        self.last_mood_time = now
        
        # Trim history to last 24 hours
        one_day_ago = now.timestamp() - (24 * 60 * 60)
        self.mood_history = [(t, e) for t, e in self.mood_history 
                            if t.timestamp() > one_day_ago]
        
        # Get songs for this emotion
        songs = self.get_songs_for_emotion(emotion)
        if not songs:
            print(f"No songs available for {emotion}")
            return
        
        # Set up new playlist
        self.current_emotion = emotion
        self.current_playlist = songs
        self.current_index = random.randint(0, len(songs) - 1)
        
        # Start playback in a separate thread
        self.play_current_song()
    
    def play_current_song(self):
        """Start playing the current song in the playlist"""
        if not self.current_playlist or self.current_index < 0:
            return
        
        self.stop_event.clear()
        self.music_thread = threading.Thread(
            target=self._play_song_thread, 
            args=(self.current_playlist[self.current_index],)
        )
        self.music_thread.daemon = True
        self.music_thread.start()
        
        # Update UI with song info
        self.update_song_info()
    
    def _play_song_thread(self, song_path):
        """Thread function to play a song"""
        try:
            print(f"Playing: {os.path.basename(song_path)}")
            pygame.mixer.music.load(song_path)
            pygame.mixer.music.play()
            self.playing = True
            self.paused = False
            
            # Get song length
            try:
                audio = mutagen.File(song_path)
                if audio:
                    self.song_length = audio.info.length
                else:
                    self.song_length = 0
            except:
                self.song_length = 0
            
            # Wait for the song to finish or for stop_event
            while pygame.mixer.music.get_busy() and not self.stop_event.is_set():
                self.current_position = pygame.mixer.music.get_pos() / 1000.0
                time.sleep(0.1)
            
            # If song ended naturally and not stopped
            if not self.stop_event.is_set() and not self.paused:
                self.play_next_song()
                
        except Exception as e:
            print(f"Error playing music: {e}")
            self.playing = False
    
    def stop_music(self):
        """Stop current music playback"""
        if self.playing or self.paused:
            self.stop_event.set()
            pygame.mixer.music.stop()
            self.playing = False
            self.paused = False
            
            # Wait for thread to finish
            if self.music_thread and self.music_thread.is_alive():
                self.music_thread.join(1.0)
    
    def play_next_song(self):
        """Play the next song in the playlist"""
        if not self.current_playlist:
            return
        
        self.current_index = (self.current_index + 1) % len(self.current_playlist)
        self.play_current_song()
    
    def play_prev_song(self):
        """Play the previous song in the playlist"""
        if not self.current_playlist:
            return
        
        self.current_index = (self.current_index - 1) % len(self.current_playlist)
        self.play_current_song()
    
    def toggle_pause(self):
        """Toggle between pause and play"""
        if self.playing:
            pygame.mixer.music.pause()
            self.playing = False
            self.paused = True
            print("Music paused")
            
            # Update UI
            if self.current_song_label:
                self.update_song_info()
                
        elif self.paused:
            pygame.mixer.music.unpause()
            self.playing = True
            self.paused = False
            print("Music resumed")
            
            # Update UI
            if self.current_song_label:
                self.update_song_info()
    
    def skip_forward(self, seconds=10):
        """Skip forward in current song"""
        if self.playing or self.paused:
            # Get current position
            current_pos = pygame.mixer.music.get_pos() / 1000.0  # in seconds
            # Calculate new position
            new_pos = current_pos + seconds
            
            # Since pygame doesn't support direct position setting, 
            # we need to reload and play from new position
            try:
                was_paused = self.paused
                song_path = self.current_playlist[self.current_index]
                pygame.mixer.music.stop()
                pygame.mixer.music.load(song_path)
                pygame.mixer.music.play(start=new_pos)
                
                if was_paused:
                    pygame.mixer.music.pause()
                    self.paused = True
                    self.playing = False
                else:
                    self.playing = True
                    self.paused = False
                    
                print(f"Skipped forward {seconds} seconds")
                return True
            except Exception as e:
                print(f"Error skipping forward: {e}")
        return False
    
    def skip_backward(self, seconds=10):
        """Skip backward in current song"""
        if self.playing or self.paused:
            # Get current position
            current_pos = pygame.mixer.music.get_pos() / 1000.0  # in seconds
            # Calculate new position (ensure not negative)
            new_pos = max(0, current_pos - seconds)
            
            # Reload and play from new position
            try:
                was_paused = self.paused
                song_path = self.current_playlist[self.current_index]
                pygame.mixer.music.stop()
                pygame.mixer.music.load(song_path)
                pygame.mixer.music.play(start=new_pos)
                
                if was_paused:
                    pygame.mixer.music.pause()
                    self.paused = True
                    self.playing = False
                else:
                    self.playing = True
                    self.paused = False
                    
                print(f"Skipped backward {seconds} seconds")
                return True
            except Exception as e:
                print(f"Error skipping backward: {e}")
        return False
    
    def set_volume(self, volume):
        """Set player volume (0.0 to 1.0)"""
        if 0.0 <= volume <= 1.0:
            pygame.mixer.music.set_volume(volume)
            self.settings.set_volume(volume)
            return True
        return False
    
    def get_current_song_info(self):
        """Get information about the current song"""
        if not self.current_playlist or self.current_index < 0:
            return None
        
        song_path = self.current_playlist[self.current_index]
        song_name = os.path.basename(song_path)
        
        # Try to get metadata
        try:
            audio = mutagen.File(song_path)
            if audio and hasattr(audio, 'tags'):
                title = audio.tags.get('title', [song_name])[0]
                artist = audio.tags.get('artist', ['Unknown'])[0]
                album = audio.tags.get('album', [''])[0]
            else:
                title = song_name
                artist = "Unknown"
                album = ""
        except:
            title = song_name
            artist = "Unknown"
            album = ""
        
        return {
            'file_name': song_name,
            'title': title,
            'artist': artist,
            'album': album,
            'path': song_path,
            'status': 'Playing' if self.playing else 'Paused' if self.paused else 'Stopped',
            'emotion': self.current_emotion,
            'index': self.current_index + 1,
            'total': len(self.current_playlist),
            'position': self.current_position,
            'length': self.song_length
        }
    
    def get_album_art(self, song_path):
        """Extract album art from audio file if available"""
        try:
            audio = mutagen.File(song_path)
            if audio and hasattr(audio, 'tags'):
                for tag in audio.tags.values():
                    if hasattr(tag, 'data') and tag.desc == 'Cover (front)':
                        return Image.open(io.BytesIO(tag.data))
            
            # If no album art found, use emotion icon
            if self.current_emotion:
                emotion_icon = os.path.join('emoji', f"{self.current_emotion}.png")
                if os.path.exists(emotion_icon):
                    return Image.open(emotion_icon)
        except:
            pass
        
        # Fallback to generic icon
        return None
    
    def detect_emotion(self, frame):
        """Detect faces and emotions using MediaPipe and the CNN model"""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            # Use first detected face
            detection = results.detections[0]
            
            # Get bounding box
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Ensure valid coordinates
            if x < 0: x = 0
            if y < 0: y = 0
            if x + width > w: width = w - x
            if y + height > h: height = h - y
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
            # Extract face for emotion detection
            face_roi = frame[y:y+height, x:x+width]
            # Convert to grayscale
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            # Resize to match model input size
            face_roi = cv2.resize(face_roi, (48, 48))
            # Normalize pixel values
            face_roi = face_roi / 255.0
            # Reshape for model
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)
            
            # Predict emotion
            emotion_probs = self.model.predict(face_roi, verbose=0)[0]
            emotion_idx = np.argmax(emotion_probs)
            emotion = self.emotions[emotion_idx]
            confidence = emotion_probs[emotion_idx]
            
            # Display emotion label with confidence
            text = f"{emotion} ({confidence:.2f})"
            cv2.putText(frame, text, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            return emotion, confidence
        
        # Fallback to Haar cascade if MediaPipe finds nothing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # Use first detected face
            x, y, w, h = faces[0]
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract face ROI and process
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)
            
            # Predict emotion
            emotion_probs = self.model.predict(face_roi, verbose=0)[0]
            emotion_idx = np.argmax(emotion_probs)
            emotion = self.emotions[emotion_idx]
            confidence = emotion_probs[emotion_idx]
            
            # Display emotion label with confidence
            text = f"{emotion} ({confidence:.2f})"
            cv2.putText(frame, text, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            return emotion, confidence
        
        return None, 0
    
    def update_song_info(self):
        """Update song information in the UI"""
        if not self.root:
            return
            
        song_info = self.get_current_song_info()
        if song_info:
            # Update song label
            if self.current_song_label:
                status = song_info['status']
                title = song_info['title']
                artist = song_info['artist']
                text = f"{status}: {title} - {artist}"
                self.current_song_label.config(text=text)
            
            # Update progress bar
            if self.progress_var and song_info['length'] > 0:
                position_percent = song_info['position'] / song_info['length']
                self.progress_var.set(position_percent)
            
            # Update album art
            if self.album_art:
                art_img = self.get_album_art(song_info['path'])
                if art_img:
                    art_img = art_img.resize((200, 200), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(art_img)
                    self.album_art.config(image=photo)
                    self.album_art.image = photo  # Keep reference to prevent garbage collection
        else:
            # No song playing
            if self.current_song_label:
                self.current_song_label.config(text="No music playing")
            if self.progress_var:
                self.progress_var.set(0)
    
    def update_mood_chart(self):
        """Update the mood tracking chart"""
        if not hasattr(self, 'mood_chart') or not self.mood_history:
            return
            
        # Clear previous plot
        self.mood_chart.clear()
        
        # Prepare data
        times = [t for t, _ in self.mood_history]
        emotions = [e for _, e in self.mood_history]
        
        # Map emotions to numeric values for plotting
        emotion_map = {e: i for i, e in enumerate(self.emotions)}
        y_values = [emotion_map.get(e, 0) for e in emotions]
        
        # Create plot
        self.mood_chart.plot(times, y_values, 'o-', markersize=8)
        self.mood_chart.set_yticks(range(len(self.emotions)))
        self.mood_chart.set_yticklabels(self.emotions)
        self.mood_chart.set_title("Your Mood History")
        self.mood_chart.set_xlabel("Time")
        
        # Format x-axis to show just hours:minutes
        self.mood_chart.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        
        # Rotate labels for better fit
        plt.setp(self.mood_chart.get_xticklabels(), rotation=45, ha='right')
        
        # Update canvas
        if hasattr(self, 'chart_canvas'):
            self.chart_canvas.draw()
    
    def show_settings(self):
        """Show settings dialog"""
        self.settings.show_settings_dialog()
        
        # Update player with new settings
        pygame.mixer.music.set_volume(self.settings.get_volume())
        if self.volume_var:
            self.volume_var.set(self.settings.get_volume())
    
    def build_gui(self):
        """Create the graphical user interface"""
        # Main window
        self.root = tk.Tk()
        self.root.title("Emotion Music Player")
        self.root.geometry("1000x700")
        self.root.configure(bg="#2b2b2b")
        
        # Configure grid layout
        self.root.columnconfigure(0, weight=3)  # Left side - webcam
        self.root.columnconfigure(1, weight=2)  # Right side - controls
        self.root.rowconfigure(0, weight=1)     # Main content
        self.root.rowconfigure(1, weight=0)     # Controls bar
        
        # Left panel - webcam feed
        left_frame = ttk.Frame(self.root, padding=10)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Webcam display
        self.cam_frame = ttk.Label(left_frame)
        self.cam_frame.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - music controls
        right_frame = ttk.Frame(self.root, padding=10)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Create notebook for tabbed interface
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Music player tab
        player_tab = ttk.Frame(notebook)
        notebook.add(player_tab, text="Music Player")
        
        # Current emotion display
        emotion_frame = ttk.Frame(player_tab)
        emotion_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(emotion_frame, text="Current Emotion:").pack(side=tk.LEFT)
        self.emotion_label = ttk.Label(emotion_frame, text="None")
        self.emotion_label.pack(side=tk.LEFT, padx=5)
        
        self.emotion_confidence = ttk.Progressbar(emotion_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.emotion_confidence.pack(side=tk.RIGHT, padx=5)
        
        # Album art
        art_frame = ttk.Frame(player_tab)
        art_frame.pack(pady=10)
        
        self.album_art = ttk.Label(art_frame)
        self.album_art.pack()
        
        # Default album art image
        empty_img = Image.new('RGB', (200, 200), color='darkgray')
        photo = ImageTk.PhotoImage(empty_img)
        self.album_art.config(image=photo)
        self.album_art.image = photo
        
        # Song info
        self.current_song_label = ttk.Label(player_tab, text="No music playing", wraplength=300)
        self.current_song_label.pack(pady=10)
        
        # Progress bar
        progress_frame = ttk.Frame(player_tab)
        progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_frame, 
            orient=tk.HORIZONTAL, 
            length=300, 
            mode='determinate',
            variable=self.progress_var
        )
        self.progress_bar.pack(fill=tk.X, padx=10)
        
        # Playback controls
        controls_frame = ttk.Frame(player_tab)
        controls_frame.pack(pady=10)
        
        prev_button = ttk.Button(controls_frame, text="⏮", width=3, command=self.play_prev_song)
        prev_button.pack(side=tk.LEFT, padx=5)
        
        rewind_button = ttk.Button(controls_frame, text="⏪", width=3, command=lambda: self.skip_backward(10))
        rewind_button.pack(side=tk.LEFT, padx=5)
        
        play_pause_button = ttk.Button(controls_frame, text="⏯", width=3, command=self.toggle_pause)
        play_pause_button.pack(side=tk.LEFT, padx=5)
        
        forward_button = ttk.Button(controls_frame, text="⏩", width=3, command=lambda: self.skip_forward(10))
        forward_button.pack(side=tk.LEFT, padx=5)
        
        next_button = ttk.Button(controls_frame, text="⏭", width=3, command=self.play_next_song)
        next_button.pack(side=tk.LEFT, padx=5)
        
        # Volume control
        volume_frame = ttk.Frame(player_tab)
        volume_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(volume_frame, text="Volume:").pack(side=tk.LEFT, padx=5)
        
        self.volume_var = tk.DoubleVar(value=self.settings.get_volume())
        volume_slider = ttk.Scale(
            volume_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.volume_var,
            command=lambda v: self.set_volume(float(v))
        )
        volume_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Settings button
        settings_button = ttk.Button(player_tab, text="Settings", command=self.show_settings)
        settings_button.pack(pady=10)
        
        # Mood tracking tab
        mood_tab = ttk.Frame(notebook)
        notebook.add(mood_tab, text="Mood Tracking")
        
        # Create figure for mood chart
        mood_frame = ttk.Frame(mood_tab)
        mood_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.mood_chart = fig.add_subplot(111)
        
        self.chart_canvas = FigureCanvasTkAgg(fig, master=mood_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Key bindings
        self.root.bind('<space>', lambda e: self.toggle_pause())
        self.root.bind('n', lambda e: self.play_next_song())
        self.root.bind('p', lambda e: self.play_prev_song())
        self.root.bind('f', lambda e: self.skip_forward(10))
        self.root.bind('b', lambda e: self.skip_backward(10))
        self.root.bind('s', lambda e: self.show_settings())
        self.root.bind('q', lambda e: self.root.quit())
        
        # Update timer for progress bar and song info
        def update_ui():
            if self.playing or self.paused:
                self.update_song_info()
            self.root.after(500, update_ui)
            
        # Start the UI update timer
        self.root.after(500, update_ui)
        
        # Also update mood chart periodically
        def update_mood_tracking():
            self.update_mood_chart()
            self.root.after(60000, update_mood_tracking)  # Update every minute
            
        self.root.after(1000, update_mood_tracking)
        
        return self.root
    
    def update_webcam_frame(self, frame):
        """Update the webcam frame in the GUI"""
        if self.cam_frame and self.root:
            # Convert to RGB for tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image=img)
            self.cam_frame.config(image=photo)
            self.cam_frame.image = photo  # Keep reference
    
    def run(self):
        """Main application loop"""
        # Initialize webcam in a separate thread
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open webcam. Please check your camera connection.")
        
        # Get frame dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # For stabilizing emotion detection
        emotion_buffer = []
        buffer_size = self.settings.get_stabilization_frames()
        confidence_threshold = self.settings.get_confidence_threshold()
        
        # Build the GUI
        gui = self.build_gui()
        
        # Function to process webcam frames
        def process_frame():
            nonlocal emotion_buffer
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                return
            
            # Detect emotion
            emotion, confidence = self.detect_emotion(frame)
            
            # Update emotion display in the UI
            if emotion and self.emotion_label:
                self.emotion_label.config(text=emotion.capitalize())
                self.emotion_confidence['value'] = confidence * 100
            
            # Only process emotions above confidence threshold
            if emotion and confidence >= confidence_threshold:
                emotion_buffer.append(emotion)
                # Keep buffer at fixed size
                if len(emotion_buffer) > buffer_size:
                    emotion_buffer.pop(0)
                
                # Get the most common emotion in the buffer
                if len(emotion_buffer) >= 3:  # Need at least 3 frames for stability
                    emotion_counts = {}
                    for e in emotion_buffer:
                        emotion_counts[e] = emotion_counts.get(e, 0) + 1
                    
                    # Find the emotion with the highest count
                    stable_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
                    
                    # Play music for the stable emotion
                    self.play_music_for_emotion(stable_emotion)
            
            # Update webcam display
            self.update_webcam_frame(frame)
            
            # Schedule next frame
            if gui.winfo_exists():
                gui.after(33, process_frame)  # ~30 FPS
            else:
                # GUI is closed, clean up
                cap.release()
        
        # Start frame processing
        gui.after(100, process_frame)
        
        # Run the GUI main loop
        gui.mainloop()
        
        # Clean up
        self.stop_music()
        cap.release()


if __name__ == "__main__":
    try:
        player = EmotionMusicPlayer()
        player.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()