import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import shutil
import mutagen
import random
from music_settings import MusicSettings
import re

class MusicScanner:
    def __init__(self):
        """Initialize the music scanner"""
        self.settings = MusicSettings()
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.music_root = self.settings.music_root
        
        # Keywords that might indicate emotional content
        self.emotion_keywords = {
            'happy': ['happy', 'joy', 'fun', 'upbeat', 'cheerful', 'bright', 'sunny', 'positive', 'exciting'],
            'sad': ['sad', 'melancholy', 'heartbreak', 'sorrow', 'depressed', 'blue', 'gloomy', 'tearful'],
            'angry': ['angry', 'rage', 'fury', 'intense', 'aggressive', 'mad', 'hate', 'furious'],
            'fear': ['fear', 'scary', 'horror', 'tense', 'frightening', 'terrifying', 'spooky', 'creepy'],
            'neutral': ['calm', 'chill', 'relaxing', 'ambient', 'background', 'neutral', 'moderate'],
            'surprise': ['surprise', 'unexpected', 'shocking', 'exciting', 'wow', 'astonishing'],
            'disgust': ['disgust', 'gross', 'revolting', 'nasty', 'repulsive']
        }
        
        # Track found files
        self.found_files = []
        
    def scan_directory(self, directory):
        """Scan a directory for music files"""
        self.found_files = []
        
        # Get supported formats
        formats = self.settings.get_supported_formats()
        
        # Walk through directory and find all music files
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(fmt) for fmt in formats):
                    full_path = os.path.join(root, file)
                    self.found_files.append(full_path)
        
        return len(self.found_files)
    
    def suggest_emotion(self, file_path):
        """Suggest an emotion for a music file based on metadata"""
        try:
            # Get tags from the file
            audio = mutagen.File(file_path)
            
            # Default to neutral if we can't determine
            best_match = 'neutral'
            
            if audio and hasattr(audio, 'tags'):
                # Extract relevant metadata
                title = ""
                artist = ""
                album = ""
                genre = ""
                comment = ""
                
                # Get metadata fields (handle different tag formats)
                if hasattr(audio, 'get'):
                    title = " ".join(audio.get('title', ['']))
                    artist = " ".join(audio.get('artist', ['']))
                    album = " ".join(audio.get('album', ['']))
                    genre = " ".join(audio.get('genre', ['']))
                    comment = " ".join(audio.get('comment', ['']))
                else:
                    # Try direct tag access for other formats
                    for tag in audio.tags:
                        tag_name = str(tag).lower()
                        tag_value = str(audio.tags[tag])
                        
                        if 'title' in tag_name:
                            title = tag_value
                        elif 'artist' in tag_name:
                            artist = tag_value
                        elif 'album' in tag_name:
                            album = tag_value
                        elif 'genre' in tag_name:
                            genre = tag_value
                        elif 'comment' in tag_name:
                            comment = tag_value
                
                # Combine all metadata into one string for analysis
                all_text = f"{title} {artist} {album} {genre} {comment}".lower()
                
                # Look for emotional keywords
                emotion_scores = {emotion: 0 for emotion in self.emotions}
                
                for emotion, keywords in self.emotion_keywords.items():
                    for keyword in keywords:
                        if re.search(r'\b' + keyword + r'\b', all_text):
                            emotion_scores[emotion] += 1
                
                # Find emotion with highest score
                max_score = max(emotion_scores.values())
                if max_score > 0:
                    # Get all emotions with the max score
                    best_emotions = [e for e, s in emotion_scores.items() if s == max_score]
                    best_match = random.choice(best_emotions)
                else:
                    # If no keywords found, try BPM-based classification if available
                    try:
                        if hasattr(audio.info, 'bpm') and audio.info.bpm:
                            bpm = float(audio.info.bpm)
                            if bpm < 70:
                                best_match = 'sad'
                            elif 70 <= bpm < 100:
                                best_match = 'neutral'
                            elif 100 <= bpm < 120:
                                best_match = 'happy'
                            else:
                                best_match = 'surprise'
                    except:
                        pass
        except:
            pass
            
        return best_match
    
    def copy_file_to_emotion_folder(self, file_path, emotion):
        """Copy a file to the specified emotion folder"""
        target_folder = self.settings.get_folder_for_emotion(emotion)
        
        # Ensure the target folder exists
        os.makedirs(target_folder, exist_ok=True)
        
        # Get the filename from the path
        filename = os.path.basename(file_path)
        
        # Check if a file with this name already exists
        target_path = os.path.join(target_folder, filename)
        if os.path.exists(target_path):
            # Add a number to make it unique
            base, ext = os.path.splitext(filename)
            count = 1
            while os.path.exists(os.path.join(target_folder, f"{base}_{count}{ext}")):
                count += 1
            target_path = os.path.join(target_folder, f"{base}_{count}{ext}")
        
        try:
            shutil.copy2(file_path, target_path)
            return True
        except Exception as e:
            print(f"Error copying file: {e}")
            return False
    
    def show_scanner_dialog(self):
        """Show the scanner dialog UI"""
        root = tk.Tk()
        root.title("Music Scanner")
        root.geometry("800x600")
        root.resizable(True, True)
        
        # Configure style
        style = ttk.Style()
        style.configure("TButton", padding=6)
        style.configure("TLabel", padding=4)
        
        # Main frame
        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top controls
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(controls_frame, text="Source Directory:").pack(side=tk.LEFT, padx=5)
        
        dir_entry = ttk.Entry(controls_frame, width=50)
        dir_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        def browse_directory():
            directory = filedialog.askdirectory(title="Select Music Source Directory")
            if directory:
                dir_entry.delete(0, tk.END)
                dir_entry.insert(0, directory)
                # Scan directory
                num_files = self.scan_directory(directory)
                status_label.config(text=f"Found {num_files} music files")
                # Update list
                update_file_list()
        
        browse_btn = ttk.Button(controls_frame, text="Browse...", command=browse_directory)
        browse_btn.pack(side=tk.LEFT, padx=5)
        
        scan_btn = ttk.Button(controls_frame, text="Scan", command=lambda: self.scan_directory(dir_entry.get()))
        scan_btn.pack(side=tk.LEFT, padx=5)
        
        # Status label
        status_label = ttk.Label(main_frame, text="No files scanned yet")
        status_label.pack(anchor=tk.W, pady=5)
        
        # Create table for files
        columns = ('File', 'Suggested', 'Selected')
        tree = ttk.Treeview(main_frame, columns=columns, show='headings')
        
        # Configure columns
        tree.heading('File', text='File')
        tree.heading('Suggested', text='Suggested Emotion')
        tree.heading('Selected', text='Selected Emotion')
        
        tree.column('File', width=400)
        tree.column('Suggested', width=150)
        tree.column('Selected', width=150)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack tree and scrollbar
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Dictionary to keep track of selected emotions for each file
        file_emotions = {}
        
        def update_file_list():
            # Clear current items
            for item in tree.get_children():
                tree.delete(item)
            
            # Add files to the list
            for file_path in self.found_files:
                filename = os.path.basename(file_path)
                suggested = self.suggest_emotion(file_path)
                
                # Use suggested emotion as the default selected
                if file_path not in file_emotions:
                    file_emotions[file_path] = suggested
                
                tree.insert('', tk.END, values=(filename, suggested, file_emotions[file_path]))
        
        # Popup menu for selecting emotion
        popup_menu = tk.Menu(root, tearoff=0)
        
        def show_popup(event):
            # Select row under mouse
            item = tree.identify_row(event.y)
            if item:
                tree.selection_set(item)
                popup_menu.post(event.x_root, event.y_root)
        
        tree.bind("<Button-3>", show_popup)
        
        def set_emotion(emotion):
            # Get selected item
            selection = tree.selection()
            if selection:
                item = selection[0]
                file_idx = tree.index(item)
                file_path = self.found_files[file_idx]
                
                # Update selected emotion
                file_emotions[file_path] = emotion
                
                # Update display
                tree.item(item, values=(os.path.basename(file_path), 
                                       tree.item(item)['values'][1], 
                                       emotion))
        
        # Add emotion options to popup menu
        for emotion in self.emotions:
            popup_menu.add_command(
                label=f"Set as {emotion.capitalize()}", 
                command=lambda e=emotion: set_emotion(e)
            )
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10, side=tk.BOTTOM)
        
        def process_files():
            # Count processing results
            copied = 0
            failed = 0
            
            for file_path, emotion in file_emotions.items():
                if self.copy_file_to_emotion_folder(file_path, emotion):
                    copied += 1
                else:
                    failed += 1
            
            # Show results
            messagebox.showinfo(
                "Processing Complete", 
                f"Processed {len(file_emotions)} files.\n"
                f"Successfully copied: {copied}\n"
                f"Failed: {failed}"
            )
            
            if copied > 0:
                status_label.config(text=f"Copied {copied} files to emotion folders")
        
        process_btn = ttk.Button(button_frame, text="Copy to Emotion Folders", command=process_files)
        process_btn.pack(side=tk.RIGHT, padx=5)
        
        cancel_btn = ttk.Button(button_frame, text="Close", command=root.destroy)
        cancel_btn.pack(side=tk.RIGHT, padx=5)
        
        # Auto suggest button
        auto_btn = ttk.Button(button_frame, text="Auto-Suggest All", 
                           command=lambda: [update_file_list(), 
                                          status_label.config(text="Auto-suggestions applied to all files")])
        auto_btn.pack(side=tk.LEFT, padx=5)
        
        # Center window on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')
        
        root.mainloop()


if __name__ == "__main__":
    scanner = MusicScanner()
    scanner.show_scanner_dialog() 