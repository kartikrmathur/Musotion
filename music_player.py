import os
import random
import pygame
import threading
import time

class MusicPlayer:
    def __init__(self):
        """Initialize the music player"""
        pygame.mixer.init()
        self.current_emotion = None
        self.is_playing = False
        self.is_paused = False
        self.music_thread = None
        self.stop_event = threading.Event()
        
        # Check multiple possible music root locations
        music_paths = [
            "music",  # Current directory
            os.path.join("Musotion", "music"),  # Musotion subdirectory
            os.path.join(os.path.dirname(__file__), "music")  # Script directory
        ]
        
        self.music_root = None
        for path in music_paths:
            if os.path.exists(path) and os.path.isdir(path):
                self.music_root = path
                break
        
        # If no music folder found, use default and warn user
        if not self.music_root:
            self.music_root = os.path.join(os.path.dirname(__file__), "music")
            print(f"Warning: Music folder not found. Will use {self.music_root}")
            print("Please create this directory and add MP3 files for each emotion.")
            try:
                os.makedirs(self.music_root, exist_ok=True)
                # Create subdirectories for each emotion
                emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                for emotion in emotions:
                    emotion_dir = os.path.join(self.music_root, emotion)
                    os.makedirs(emotion_dir, exist_ok=True)
                print(f"Created music directory structure at {self.music_root}")
            except Exception as e:
                print(f"Error creating music directories: {e}")
        else:
            print(f"Using music directory: {self.music_root}")
        
        # Track playlist info
        self.current_playlist = []  # All songs for current emotion
        self.current_song_index = -1  # Index in playlist
        self.current_song_path = None
        
    def get_songs_for_emotion(self, emotion):
        """Get all songs for a specific emotion"""
        emotion_folder = os.path.join(self.music_root, emotion)
        
        # Check if folder exists
        if not os.path.exists(emotion_folder):
            print(f"Error: Music folder for {emotion} not found at {emotion_folder}")
            try:
                os.makedirs(emotion_folder, exist_ok=True)
                print(f"Created empty directory for {emotion} at {emotion_folder}")
                print(f"Please add MP3 files to {emotion_folder}")
            except Exception as e:
                print(f"Error creating {emotion} directory: {e}")
            return []
            
        # Get all mp3 files in the folder
        songs = [f for f in os.listdir(emotion_folder) 
                if f.lower().endswith('.mp3')]
        
        if not songs:
            print(f"No MP3 files found in {emotion_folder}")
            print(f"Please add MP3 files to {emotion_folder}")
        
        # Return list of full paths
        return [os.path.join(emotion_folder, song) for song in songs]
    
    def get_random_song(self, emotion):
        """Get a random song from the emotion folder"""
        songs = self.get_songs_for_emotion(emotion)
        
        # Check if there are any songs
        if not songs:
            print(f"No MP3 files found for {emotion}")
            return None
            
        # Return a random song with full path
        return random.choice(songs)
    
    def play_music(self, emotion):
        """Play music based on emotion"""
        # Stop any currently playing music
        self.stop_music()
        
        # Update current emotion
        self.current_emotion = emotion
        
        # Get all songs for this emotion
        self.current_playlist = self.get_songs_for_emotion(emotion)
        
        if self.current_playlist:
            # Start with a random song
            self.current_song_index = random.randint(0, len(self.current_playlist) - 1)
            self.current_song_path = self.current_playlist[self.current_song_index]
            
            # Start playing in a separate thread
            self._start_playback()
            return True
        else:
            print(f"No songs available for {emotion} emotion")
            return False
    
    def _start_playback(self):
        """Start playing the current song"""
        self.stop_event.clear()
        self.music_thread = threading.Thread(
            target=self._play_music_thread, 
            args=(self.current_song_path,)
        )
        self.music_thread.daemon = True
        self.music_thread.start()
    
    def _play_music_thread(self, song_path):
        """Thread function to play music"""
        try:
            print(f"Playing: {os.path.basename(song_path)}")
            pygame.mixer.music.load(song_path)
            pygame.mixer.music.play()
            self.is_playing = True
            self.is_paused = False
            
            # Wait for music to finish or stop_event to be set
            while (pygame.mixer.music.get_busy() or self.is_paused) and not self.stop_event.is_set():
                time.sleep(0.1)
                
            # If music ended naturally and not stopped/switched
            if not self.stop_event.is_set() and not self.is_paused:
                self.is_playing = False
                print("Music finished playing")
                # Auto-play next song
                self.next_song()
                
        except Exception as e:
            print(f"Error playing music: {e}")
            self.is_playing = False
            self.is_paused = False
    
    def stop_music(self):
        """Stop currently playing music"""
        if self.is_playing or self.is_paused:
            self.stop_event.set()
            pygame.mixer.music.stop()
            self.is_playing = False
            self.is_paused = False
            # Wait for thread to finish
            if self.music_thread and self.music_thread.is_alive():
                self.music_thread.join(1.0)
    
    def pause_music(self):
        """Pause currently playing music"""
        if self.is_playing and not self.is_paused:
            pygame.mixer.music.pause()
            self.is_paused = True
            self.is_playing = False
            print("Music paused")
            return True
        return False
    
    def resume_music(self):
        """Resume paused music"""
        if self.is_paused:
            pygame.mixer.music.unpause()
            self.is_paused = False
            self.is_playing = True
            print("Music resumed")
            return True
        return False
    
    def toggle_pause(self):
        """Toggle between pause and play"""
        if self.is_paused:
            return self.resume_music()
        elif self.is_playing:
            return self.pause_music()
        elif self.current_song_path:  # Stopped but song loaded
            self._start_playback()
            return True
        return False
    
    def skip_forward(self, seconds=10):
        """Skip forward by N seconds"""
        if self.is_playing or self.is_paused:
            # Get current position
            current_pos = pygame.mixer.music.get_pos() / 1000.0  # in seconds
            # Calculate new position
            new_pos = current_pos + seconds
            
            # Since pygame doesn't support direct position setting, 
            # we need to reload and play from new position
            try:
                was_paused = self.is_paused
                pygame.mixer.music.stop()
                pygame.mixer.music.load(self.current_song_path)
                pygame.mixer.music.play(start=new_pos)
                
                if was_paused:
                    pygame.mixer.music.pause()
                    self.is_paused = True
                    self.is_playing = False
                else:
                    self.is_playing = True
                    self.is_paused = False
                    
                print(f"Skipped forward {seconds} seconds")
                return True
            except Exception as e:
                print(f"Error skipping forward: {e}")
        return False
    
    def skip_backward(self, seconds=10):
        """Skip backward by N seconds"""
        if self.is_playing or self.is_paused:
            # Get current position
            current_pos = pygame.mixer.music.get_pos() / 1000.0  # in seconds
            # Calculate new position (ensure not negative)
            new_pos = max(0, current_pos - seconds)
            
            try:
                was_paused = self.is_paused
                pygame.mixer.music.stop()
                pygame.mixer.music.load(self.current_song_path)
                pygame.mixer.music.play(start=new_pos)
                
                if was_paused:
                    pygame.mixer.music.pause()
                    self.is_paused = True
                    self.is_playing = False
                else:
                    self.is_playing = True
                    self.is_paused = False
                    
                print(f"Skipped backward {seconds} seconds")
                return True
            except Exception as e:
                print(f"Error skipping backward: {e}")
        return False
    
    def next_song(self):
        """Play the next song in the playlist"""
        if not self.current_playlist:
            print("No playlist available")
            return False
            
        # Stop current playback
        self.stop_music()
        
        # Move to next song
        self.current_song_index = (self.current_song_index + 1) % len(self.current_playlist)
        self.current_song_path = self.current_playlist[self.current_song_index]
        
        # Start playing
        self._start_playback()
        return True
    
    def previous_song(self):
        """Play the previous song in the playlist"""
        if not self.current_playlist:
            print("No playlist available")
            return False
            
        # Stop current playback
        self.stop_music()
        
        # Move to previous song
        self.current_song_index = (self.current_song_index - 1) % len(self.current_playlist)
        self.current_song_path = self.current_playlist[self.current_song_index]
        
        # Start playing
        self._start_playback()
        return True
    
    def set_volume(self, volume):
        """Set music volume (0.0 to 1.0)"""
        pygame.mixer.music.set_volume(max(0.0, min(1.0, volume)))
    
    def get_current_emotion(self):
        """Return the current emotion being played"""
        return self.current_emotion
    
    def get_current_song_info(self):
        """Return information about the current song"""
        if self.current_song_path:
            file_name = os.path.basename(self.current_song_path)
            status = "Playing" if self.is_playing else ("Paused" if self.is_paused else "Stopped")
            position = pygame.mixer.music.get_pos() / 1000.0 if (self.is_playing or self.is_paused) else 0
            return {
                "file_name": file_name,
                "status": status,
                "position": position,
                "index": self.current_song_index + 1,
                "total": len(self.current_playlist)
            }
        return None


# Example usage
if __name__ == "__main__":
    player = MusicPlayer()
    
    # Print available emotion folders
    music_root = "music"
    if os.path.exists(music_root):
        print("Available emotion folders:")
        for folder in os.listdir(music_root):
            folder_path = os.path.join(music_root, folder)
            if os.path.isdir(folder_path):
                songs = [f for f in os.listdir(folder_path) if f.lower().endswith('.mp3')]
                print(f"  - {folder}: {len(songs)} songs")
    
        print("\nMusic player controls test:")
        print("  p: Play/Pause")
        print("  s: Stop")
        print("  n: Next song")
        print("  b: Previous song")
        print("  f: Skip forward 10 seconds")
        print("  r: Skip backward 10 seconds")
        print("  q: Quit")
        
        # Test with the first emotion that has songs
        test_emotion = None
        for emotion in ["happy", "sad", "angry", "neutral", "surprise", "fear", "disgust"]:
            songs = player.get_songs_for_emotion(emotion)
            if songs:
                test_emotion = emotion
                break
        
        if test_emotion:
            print(f"\nTesting with {test_emotion} music...")
            if player.play_music(test_emotion):
                # Interactive control loop
                while True:
                    cmd = input("\nEnter command (p/s/n/b/f/r/q): ").lower()
                    if cmd == 'p':
                        player.toggle_pause()
                    elif cmd == 's':
                        player.stop_music()
                    elif cmd == 'n':
                        player.next_song()
                    elif cmd == 'b':
                        player.previous_song()
                    elif cmd == 'f':
                        player.skip_forward()
                    elif cmd == 'r':
                        player.skip_backward()
                    elif cmd == 'q':
                        player.stop_music()
                        break
                    
                    # Display current song info
                    info = player.get_current_song_info()
                    if info:
                        print(f"{info['status']}: {info['file_name']} ({info['index']}/{info['total']})")
        else:
            print("No music files found in any emotion folder. Please add MP3 files.")
    
    print("\nMusic player test complete") 