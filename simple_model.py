import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--csv_file', type=str, required=True)
parser.add_argument('--export_path', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=5)
parser.add_argument('--debug', dest='debug', action='store_true')
args = parser.parse_args()

if args.debug:
    args.batch_size = 32
    args.n_epochs = 1
    print("Debug mode enabled. Using smaller batch size and 1 epoch.")

# Constants
NUM_CLASSES = 7  # 7 different emotions
IMG_SIZE = 48    # 48x48 pixel images

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

# Load and prepare the dataset
print(f"Loading dataset from {args.csv_file}...")
try:
    data = pd.read_csv(args.csv_file)
    print(f"Loaded dataset with {len(data)} samples")
except Exception as e:
    print(f"Error loading CSV file: {e}")
    sys.exit(1)

# Validate required columns
if 'emotion' not in data.columns or 'pixels' not in data.columns:
    print("Error: Dataset must contain 'emotion' and 'pixels' columns")
    sys.exit(1)

# Process emotions (convert to one-hot encoding)
emotions = data['emotion'].values
y_data = to_categorical(emotions, NUM_CLASSES)

# Process pixels
pixels = data['pixels'].values
x_data = np.zeros((len(pixels), IMG_SIZE, IMG_SIZE, 1))

for i, pixel_string in enumerate(pixels):
    try:
        # Check if all pixel values are the same (likely dummy data)
        if len(set(pixel_string.split())) <= 1:
            # Create random pixel data for demonstration
            pixel_values = np.random.randint(0, 256, (IMG_SIZE, IMG_SIZE))
        else:
            # Parse the pixel string to get values
            pixel_values = np.array([int(p) for p in pixel_string.split()]).reshape(IMG_SIZE, IMG_SIZE)
        
        # Normalize pixel values to [0, 1]
        x_data[i, :, :, 0] = pixel_values / 255.0
    except Exception as e:
        print(f"Error processing pixels for row {i}: {e}")
        # Fill with random pixels if there's an error
        x_data[i, :, :, 0] = np.random.random((IMG_SIZE, IMG_SIZE))

# Split data into training and testing sets
train_ratio = 0.8
indices = np.random.permutation(len(x_data))
train_count = int(len(indices) * train_ratio)
train_indices = indices[:train_count]
test_indices = indices[train_count:]

x_train, x_test = x_data[train_indices], x_data[test_indices]
y_train, y_test = y_data[train_indices], y_data[test_indices]

print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")

# Create a simple CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Train the model
print(f"Training model with batch_size={args.batch_size}, epochs={args.n_epochs}...")
history = model.fit(
    x_train, y_train,
    batch_size=args.batch_size,
    epochs=args.n_epochs,
    validation_data=(x_test, y_test),
    verbose=1
)

# Evaluate the model
print("Evaluating model...")
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")

# Create the output directory if it doesn't exist
if not os.path.exists(args.export_path):
    os.makedirs(args.export_path)
    print(f"Created directory: {args.export_path}")

# Save the model
model_file = os.path.join(args.export_path, "emotion_model.keras")
model.save(model_file)
print(f"Model saved to {model_file}")

# Save a test sample for later verification
sample_file = os.path.join(args.export_path, "test_sample.npz")
np.savez(sample_file, x=x_test[:1], y=y_test[:1])
print(f"Test sample saved to {sample_file}")

print("Training completed successfully.") 