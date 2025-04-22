import os
import sys
import tensorflow as tf
from tensorflow import keras

print("==== TENSORFLOW/KERAS TEST ====")
# Print version information
print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")
print(f"TensorFlow path: {tf.__file__}")
print(f"Keras path: {keras.__file__}")

print("\n==== TESTING MODEL CREATION ====")
# Try to create a simple model
try:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    print("✓ Successfully created and compiled a simple model.")
    
    # Try saving and loading the model
    temp_model_path = "test_model.h5"
    print(f"Saving model to {temp_model_path}...")
    model.save(temp_model_path)
    print(f"✓ Model saved to {temp_model_path}")
    
    print(f"Loading model from {temp_model_path}...")
    loaded_model = tf.keras.models.load_model(temp_model_path)
    print("✓ Successfully loaded model.")
    
    # Clean up
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)
        print(f"✓ Removed temporary model file: {temp_model_path}")
        
    print("\n==== TEST SUCCESSFUL ====")
    print("TensorFlow and Keras are working correctly.")
        
except Exception as e:
    print(f"\nERROR: {e}")
    print("TensorFlow or Keras may not be installed correctly.") 