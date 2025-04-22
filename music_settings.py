import os
import json
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pygame
import importlib

class MusicSettings:
    def __init__(self):
        """Initialize settings manager"""
        self.config_file = "music_settings.json"
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.music_root = "music"
        self.settings = self._load_settings()
        
        # Initialize pygame mixer if not already initialized
        if not pygame.get_init():
            pygame.init()
        if not pygame.mixer.get_init():
            pygame.mixer.init()
    
    def _load_settings(self):
        """Load settings from JSON file or create default settings"""
        default_settings = {
            "music_folders": {emotion: os.path.join(self.music_root, emotion) for emotion in self.emotions},
            "volume": 0.7,
            "confidence_threshold": 0.5,
            "stabilization_frames": 5,
            "supported_formats": [".mp3", ".wav", ".ogg"]
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    settings = json.load(f)
                # Ensure all required keys exist
                for key in default_settings:
                    if key not in settings:
                        settings[key] = default_settings[key]
                return settings
            except Exception as e:
                print(f"Error loading settings: {e}")
                return default_settings
        else:
            return default_settings
    
    def save_settings(self):
        """Save current settings to JSON file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False
    
    def get_folder_for_emotion(self, emotion):
        """Get the configured music folder for a specific emotion"""
        if emotion in self.settings["music_folders"]:
            return self.settings["music_folders"][emotion]
        # Fallback to default
        return os.path.join(self.music_root, emotion)
    
    def set_folder_for_emotion(self, emotion, folder_path):
        """Set a custom folder path for a specific emotion"""
        if emotion in self.emotions:
            self.settings["music_folders"][emotion] = folder_path
            return True
        return False
    
    def get_volume(self):
        """Get the current volume setting"""
        return self.settings["volume"]
    
    def set_volume(self, volume):
        """Set the volume (0.0 to 1.0)"""
        if 0.0 <= volume <= 1.0:
            self.settings["volume"] = volume
            pygame.mixer.music.set_volume(volume)
            return True
        return False
    
    def get_confidence_threshold(self):
        """Get the confidence threshold for emotion detection"""
        return self.settings["confidence_threshold"]
    
    def set_confidence_threshold(self, threshold):
        """Set the confidence threshold for emotion detection"""
        if 0.0 <= threshold <= 1.0:
            self.settings["confidence_threshold"] = threshold
            return True
        return False
    
    def get_stabilization_frames(self):
        """Get the number of frames used for stabilizing emotion detection"""
        return self.settings["stabilization_frames"]
    
    def set_stabilization_frames(self, frames):
        """Set the number of frames used for stabilizing emotion detection"""
        if frames >= 1:
            self.settings["stabilization_frames"] = int(frames)
            return True
        return False
    
    def get_supported_formats(self):
        """Get list of supported audio file formats"""
        return self.settings["supported_formats"]
    
    def open_music_scanner(self):
        """Open the music scanner utility"""
        try:
            # Import dynamically to avoid circular imports
            scanner_module = importlib.import_module('music_scanner')
            scanner = scanner_module.MusicScanner()
            scanner.show_scanner_dialog()
        except Exception as e:
            messagebox.showerror("Error", f"Could not open music scanner: {e}")
    
    def show_settings_dialog(self):
        """Display a GUI dialog for changing settings"""
        root = tk.Tk()
        root.title("Music Player Settings")
        root.geometry("600x500")
        root.resizable(True, True)
        
        # Style configuration
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        style.configure("TLabel", padding=6, font=('Helvetica', 10))
        style.configure("TFrame", padding=10)
        
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabbed interface
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Music Folders Tab
        folders_frame = ttk.Frame(notebook)
        notebook.add(folders_frame, text="Music Folders")
        
        # Music Scanner button at the top
        scanner_frame = ttk.Frame(folders_frame)
        scanner_frame.pack(fill=tk.X, pady=10)
        
        scanner_label = ttk.Label(scanner_frame, text="Need music files for emotions?")
        scanner_label.pack(side=tk.LEFT, padx=5)
        
        scanner_btn = ttk.Button(
            scanner_frame, 
            text="Open Music Scanner",
            command=self.open_music_scanner
        )
        scanner_btn.pack(side=tk.RIGHT, padx=5)
        
        # Add separator
        ttk.Separator(folders_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Dictionary to store folder entry widgets
        folder_entries = {}
        
        # Create UI elements for each emotion
        for i, emotion in enumerate(self.emotions):
            row_frame = ttk.Frame(folders_frame)
            row_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(row_frame, text=f"{emotion.capitalize()}:").pack(side=tk.LEFT, padx=5)
            
            entry = ttk.Entry(row_frame, width=40)
            entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            entry.insert(0, self.settings["music_folders"][emotion])
            folder_entries[emotion] = entry
            
            # Browse button
            def create_browse_callback(emotion_name, entry_widget):
                def browse_callback():
                    folder = filedialog.askdirectory(
                        title=f"Select Music Folder for {emotion_name.capitalize()}"
                    )
                    if folder:
                        entry_widget.delete(0, tk.END)
                        entry_widget.insert(0, folder)
                return browse_callback
            
            browse_btn = ttk.Button(
                row_frame, 
                text="Browse...",
                command=create_browse_callback(emotion, entry)
            )
            browse_btn.pack(side=tk.RIGHT, padx=5)
        
        # Playback Settings Tab
        playback_frame = ttk.Frame(notebook)
        notebook.add(playback_frame, text="Playback")
        
        # Volume control
        volume_frame = ttk.Frame(playback_frame)
        volume_frame.pack(fill=tk.X, pady=10)
        ttk.Label(volume_frame, text="Volume:").pack(side=tk.LEFT, padx=5)
        
        volume_var = tk.DoubleVar(value=self.settings["volume"])
        volume_scale = ttk.Scale(
            volume_frame, 
            from_=0.0, 
            to=1.0, 
            orient=tk.HORIZONTAL,
            variable=volume_var,
            length=300
        )
        volume_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        volume_label = ttk.Label(volume_frame, text=f"{int(self.settings['volume']*100)}%")
        volume_label.pack(side=tk.RIGHT, padx=5)
        
        def update_volume_label(event):
            volume_label.config(text=f"{int(volume_var.get()*100)}%")
            # Preview volume change
            pygame.mixer.music.set_volume(volume_var.get())
            
        volume_scale.bind("<Motion>", update_volume_label)
        
        # Format support section
        format_frame = ttk.Frame(playback_frame)
        format_frame.pack(fill=tk.X, pady=10)
        ttk.Label(format_frame, text="Supported Formats:").pack(anchor=tk.W, pady=5)
        
        # Format checkboxes
        format_vars = {}
        all_formats = [".mp3", ".wav", ".ogg", ".flac", ".m4a"]
        for fmt in all_formats:
            var = tk.BooleanVar(value=fmt in self.settings["supported_formats"])
            format_vars[fmt] = var
            cb = ttk.Checkbutton(format_frame, text=fmt, variable=var)
            cb.pack(anchor=tk.W, padx=20)
        
        # Detection Settings Tab
        detection_frame = ttk.Frame(notebook)
        notebook.add(detection_frame, text="Detection")
        
        # Confidence threshold
        conf_frame = ttk.Frame(detection_frame)
        conf_frame.pack(fill=tk.X, pady=10)
        ttk.Label(conf_frame, text="Confidence Threshold:").pack(anchor=tk.W, pady=5)
        
        threshold_var = tk.DoubleVar(value=self.settings["confidence_threshold"])
        threshold_scale = ttk.Scale(
            conf_frame, 
            from_=0.0, 
            to=1.0, 
            orient=tk.HORIZONTAL,
            variable=threshold_var,
            length=300
        )
        threshold_scale.pack(anchor=tk.W, padx=20, fill=tk.X)
        
        threshold_label = ttk.Label(conf_frame, text=f"{int(self.settings['confidence_threshold']*100)}%")
        threshold_label.pack(anchor=tk.W, padx=20)
        
        def update_threshold_label(event):
            threshold_label.config(text=f"{int(threshold_var.get()*100)}%")
            
        threshold_scale.bind("<Motion>", update_threshold_label)
        
        # Stabilization frames
        stab_frame = ttk.Frame(detection_frame)
        stab_frame.pack(fill=tk.X, pady=10)
        ttk.Label(stab_frame, text="Stabilization Frames:").pack(anchor=tk.W, pady=5)
        
        frames_var = tk.IntVar(value=self.settings["stabilization_frames"])
        stab_spinbox = ttk.Spinbox(
            stab_frame, 
            from_=1, 
            to=20, 
            textvariable=frames_var,
            width=5
        )
        stab_spinbox.pack(anchor=tk.W, padx=20)
        ttk.Label(stab_frame, text="(Higher values = more stable emotion detection, but slower response)").pack(anchor=tk.W, padx=20)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        def save_callback():
            try:
                # Save folder settings
                for emotion, entry in folder_entries.items():
                    folder_path = entry.get().strip()
                    if folder_path:
                        self.settings["music_folders"][emotion] = folder_path
                
                # Save playback settings
                self.settings["volume"] = volume_var.get()
                
                # Save format settings
                self.settings["supported_formats"] = [fmt for fmt, var in format_vars.items() if var.get()]
                
                # Save detection settings
                self.settings["confidence_threshold"] = threshold_var.get()
                self.settings["stabilization_frames"] = frames_var.get()
                
                # Write to file
                if self.save_settings():
                    messagebox.showinfo("Settings Saved", "Your settings have been saved successfully.")
                    root.destroy()
                else:
                    messagebox.showerror("Error", "Failed to save settings file.")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")
        
        save_btn = ttk.Button(button_frame, text="Save", command=save_callback)
        save_btn.pack(side=tk.RIGHT, padx=5)
        
        cancel_btn = ttk.Button(button_frame, text="Cancel", command=root.destroy)
        cancel_btn.pack(side=tk.RIGHT, padx=5)
        
        # Center window on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')
        
        root.mainloop()


if __name__ == "__main__":
    # Run settings dialog when script is executed directly
    settings = MusicSettings()
    settings.show_settings_dialog() 