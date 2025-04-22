import pandas as pd
import numpy as np
import os
import sys

def convert_dataset(input_file, output_file):
    # Load the input CSV file
    print(f"Loading dataset from {input_file}...")
    try:
        data = pd.read_csv(input_file)
        print(f"Loaded dataset with {len(data)} samples")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return False
    
    print(f"Original columns: {data.columns.tolist()}")
    
    # Check if the dataset is already in the expected format
    if 'emotion' in data.columns and 'pixels' in data.columns:
        print("Dataset is already in the expected format")
        if input_file != output_file:
            data.to_csv(output_file, index=False)
            print(f"Copied to {output_file}")
        return True
    
    # Create a new DataFrame for the converted data
    converted_data = pd.DataFrame()
    
    # Emotion mapping: map column names to emotion indices
    # Standard FER2013 emotion mapping: 
    # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    emotion_cols = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    
    # Alternative column names that might be present
    alt_emotion_cols = {
        'angry': 'anger',
        'happy': 'happiness',
        'sad': 'sadness'
    }
    
    # Normalize column names
    for alt, std in alt_emotion_cols.items():
        if alt in data.columns and std not in data.columns:
            data[std] = data[alt]
    
    # Check which emotion columns are present
    present_cols = [col for col in emotion_cols if col in data.columns]
    
    if len(present_cols) < 3:  # Need at least 3 emotion columns
        print(f"Error: Not enough emotion columns found. Present: {present_cols}")
        return False
    
    print(f"Found emotion columns: {present_cols}")
    
    # For each row, find the emotion with the highest value
    # and create a new 'emotion' column with the corresponding index
    emotion_indices = {
        'anger': 0,
        'disgust': 1,
        'fear': 2,
        'happiness': 3,
        'sadness': 4,
        'surprise': 5,
        'neutral': 6
    }
    
    # Function to find the emotion with highest value and return its index
    def get_emotion_index(row):
        max_emotion = None
        max_value = -1
        
        for emotion in present_cols:
            if row[emotion] > max_value:
                max_value = row[emotion]
                max_emotion = emotion
        
        return emotion_indices.get(max_emotion, 6)  # Default to neutral if not found
    
    # Create the emotion column
    converted_data['emotion'] = data.apply(get_emotion_index, axis=1)
    
    # Create a dummy pixels column (48x48=2304 pixels of value 128)
    # In a real implementation, you would extract the actual pixel data from images
    pixel_dummy = ' '.join(['128'] * 2304)  # 48x48 = 2304 pixels
    converted_data['pixels'] = [pixel_dummy] * len(data)
    
    # Save the converted dataset
    converted_data.to_csv(output_file, index=False)
    print(f"Converted dataset saved to {output_file}")
    print(f"Created {len(converted_data)} samples with columns: {converted_data.columns.tolist()}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_dataset.py input_file.csv [output_file.csv]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # Generate output filename based on input filename
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_converted{ext}"
    
    success = convert_dataset(input_file, output_file)
    
    if success:
        print("\nNext steps:")
        print(f"1. Run the training script on the converted dataset:")
        print(f"   python emotionRecTrain.py --csv_file={output_file} --export_path=output_model --debug")
    else:
        print("\nConversion failed. Please check the format of your input file.")
        sys.exit(1) 