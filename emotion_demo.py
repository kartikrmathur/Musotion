import sys
import os

def main():
    print("Python version:", sys.version)
    
    try:
        import tensorflow as tf
        print("TensorFlow version:", tf.__version__)
    except ImportError:
        print("TensorFlow is not installed or not compatible with this Python version (3.13).")
        print("TensorFlow currently supports Python versions up to 3.12.")
        print("To fix this issue, you need to use a compatible Python version:")
        print("  1. Create a virtual environment with Python 3.12 or earlier")
        print("  2. Install TensorFlow in that environment")
        
    # Show that we can load the sample dataset
    try:
        import pandas as pd
        import numpy as np
        print("\nChecking for sample dataset...")
        
        file_path = 'sample_fer2013.csv'
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            print(f"Successfully loaded the sample dataset: {file_path}")
            print(f"Dataset contains {len(data)} samples")
            print("Sample emotions:", data['emotion'].values[:5])
            
            # Demonstrate that we can process the data
            # This shows the preprocessing works, even if we can't train the model
            print("\nDemonstrating data preprocessing...")
            
            # Process one sample to show the pixel conversion
            sample_pixels = data['pixels'][0]
            pixel_data = sample_pixels.split()
            sample_array = np.array(pixel_data, dtype=np.uint8).reshape(48, 48)
            print(f"First few pixel values: {sample_array.flatten()[:10]}")
            print(f"Image shape: {sample_array.shape}")
            
            # This would work if TensorFlow were installed
            print("\nTraining would work if TensorFlow were installed in Python 3.12 or earlier")
        else:
            print(f"Could not find {file_path} in the current directory")
            print(f"Current directory: {os.getcwd()}")
            files = os.listdir()
            print(f"Files in current directory: {files}")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
    
    print("\nRecommended solution:")
    print("1. Install Python 3.12 (or earlier version), which is compatible with TensorFlow")
    print("2. Create a virtual environment using that Python version")
    print("3. Install TensorFlow in the virtual environment")
    print("4. Run the emotionRecTrain.py script in that environment")

if __name__ == "__main__":
    main() 