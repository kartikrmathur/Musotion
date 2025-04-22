import argparse
<<<<<<< HEAD
import os
import sys
=======
>>>>>>> e845925b6297e856d3840686af965c6579551143

import numpy as np
import pandas as pd
import tensorflow as tf
<<<<<<< HEAD
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Print TensorFlow version info
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

=======
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adamax
from keras.utils import np_utils
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter
from tensorflow.keras.utils import to_categorical

>>>>>>> e845925b6297e856d3840686af965c6579551143
parser = argparse.ArgumentParser()
parser.add_argument('--csv_file', type=str, required=True)
parser.add_argument('--export_path', type=str, required=True)
# OPTIONAL
<<<<<<< HEAD
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=20)
=======
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--n_epochs', type=int, default=1)
>>>>>>> e845925b6297e856d3840686af965c6579551143
parser.add_argument('--debug', dest='debug', action='store_true')

FLAGS = parser.parse_args()

if (FLAGS.debug):
    FLAGS.batch_size = 10
    FLAGS.n_epochs = 1
<<<<<<< HEAD
    print("Debug mode enabled. Using batch_size=10 and n_epochs=1")
=======
>>>>>>> e845925b6297e856d3840686af965c6579551143

NUM_CLASSES = 7
IMG_SIZE = 48

<<<<<<< HEAD
# For small sample dataset, adjust train/test split
# These values can be modified based on the size of your dataset
TRAIN_SPLIT = 0.8  # Use 80% for training, 20% for testing by default


def split_for_test(data_array, split_ratio=TRAIN_SPLIT):
    """Split data into training and testing sets based on specified ratio"""
    if not isinstance(data_array, np.ndarray):
        data_array = np.array(data_array)
    
    n_samples = len(data_array)
    
    if n_samples <= 1:
        print("Error: Not enough data to split for training and testing")
        return data_array, np.array([])
    
    n_train = max(1, int(n_samples * split_ratio))
    
    # Ensure we have at least one sample for testing if there's more than one sample
    if n_samples > 1 and n_train >= n_samples:
        n_train = n_samples - 1
    
    train = data_array[:n_train]
    test = data_array[n_train:] if n_train < n_samples else np.array([])
    
    print(f"Split dataset: {len(train)} training samples, {len(test)} testing samples")
=======
# TODO: Use the 'Usage' field to separate based on training/testing
TRAIN_END = 28708
TEST_START = TRAIN_END + 1


def split_for_test(list):
    train = list[0:TRAIN_END]
    test = list[TEST_START:]
>>>>>>> e845925b6297e856d3840686af965c6579551143
    return train, test


def pandas_vector_to_list(pandas_df):
<<<<<<< HEAD
    """Convert a pandas column to a Python list"""
    try:
        py_list = [item[0] for item in pandas_df.values.tolist()]
        return py_list
    except (IndexError, TypeError) as e:
        print(f"Error converting pandas column to list: {e}")
        print("Attempting alternative conversion...")
        return pandas_df.values.flatten().tolist()
=======
    py_list = [item[0] for item in pandas_df.values.tolist()]
    return py_list
>>>>>>> e845925b6297e856d3840686af965c6579551143


def process_emotion(emotion):
    """
    Takes in a vector of emotions and outputs a list of emotions as one-hot vectors.
<<<<<<< HEAD
    :param emotion: vector of ints (0-6)
    :return: list of one-hot vectors (array of 7)
    """
    try:
        emotion_as_list = pandas_vector_to_list(emotion)
        y_data = []
        for index in range(len(emotion_as_list)):
            emotion_value = emotion_as_list[index]
            # Ensure emotion value is valid (0-6 for 7 classes)
            if not (0 <= emotion_value < NUM_CLASSES):
                print(f"Warning: Invalid emotion value {emotion_value} at index {index}. Using 0 instead.")
                emotion_value = 0
            y_data.append(emotion_value)

        # Y data
        y_data_categorical = to_categorical(y_data, NUM_CLASSES)
        return y_data_categorical
    except Exception as e:
        print(f"Error processing emotions: {e}")
        sys.exit(1)
=======
    :param emotion: vector of ints (0-7)
    :return: list of one-hot vectors (array of 7)
    """
    emotion_as_list = pandas_vector_to_list(emotion)
    y_data = []
    for index in range(len(emotion_as_list)):
        y_data.append(emotion_as_list[index])

    # Y data
    y_data_categorical = to_categorical(y_data, NUM_CLASSES)
    return y_data_categorical
>>>>>>> e845925b6297e856d3840686af965c6579551143


def process_pixels(pixels, img_size=IMG_SIZE):
    """
    Takes in a string (pixels) that has space separated ints. Will transform the ints
    to a 48x48 matrix of floats(/255).
    :param pixels: string with space separated ints
    :param img_size: image size
    :return: array of 48x48 matrices
    """
<<<<<<< HEAD
    try:
        pixels_as_list = pandas_vector_to_list(pixels)

        np_image_array = []
        for index, item in enumerate(pixels_as_list):
            # 48x48
            data = np.zeros((img_size, img_size), dtype=np.uint8)
            
            # Handle non-string pixel data
            if not isinstance(item, str):
                print(f"Warning: Non-string pixel data at index {index}. Skipping.")
                continue
                
            # split space separated ints
            pixel_data = item.split()
            
            # Check if we have the correct number of pixels
            expected_pixels = img_size * img_size
            if len(pixel_data) < expected_pixels:
                # Pad with zeros if needed
                print(f"Warning: Insufficient pixel data at index {index}. Padding with zeros.")
                pixel_data = pixel_data + ['0'] * (expected_pixels - len(pixel_data))
            elif len(pixel_data) > expected_pixels:
                # Truncate if needed
                print(f"Warning: Excess pixel data at index {index}. Truncating.")
                pixel_data = pixel_data[:expected_pixels]

            try:
                # Convert to integers
                pixel_data = [int(p) for p in pixel_data]
            except ValueError as e:
                print(f"Error converting pixel data to integers at index {index}: {e}")
                # Fill with zeros as a fallback
                pixel_data = [0] * expected_pixels

            # 0 -> 47, loop through the rows
            for i in range(0, img_size):
                # (0 = 0), (1 = 48), (2 = 96), ...
                pixel_index = i * img_size
                # (0 = [0:47]), (1 = [48:95]), (2 = [96:143]), ...
                data[i] = pixel_data[pixel_index:pixel_index + img_size]

            np_image_array.append(np.array(data))

        np_image_array = np.array(np_image_array)
        if len(np_image_array) == 0:
            print("Error: No valid image data processed.")
            sys.exit(1)
            
        # convert to float and divide by 255
        np_image_array = np_image_array.astype('float32') / 255.0
        return np_image_array
    except Exception as e:
        print(f"Error processing pixel data: {e}")
        sys.exit(1)


def duplicate_input_layer(array_input, size):
    """
    Convert grayscale images to RGB format for VGG16 input
    """
    try:
        vg_input = np.empty([size, 48, 48, 3])
        for index, item in enumerate(array_input):
            vg_input[index, :, :, 0] = item
            vg_input[index, :, :, 1] = item
            vg_input[index, :, :, 2] = item
        return vg_input
    except Exception as e:
        print(f"Error duplicating input layer: {e}")
        sys.exit(1)


def get_vgg16_output(vgg16, array_input, n_feature_maps):
    """
    Generate feature maps using VGG16
    """
    try:
        vg_input = duplicate_input_layer(array_input, n_feature_maps)

        print(f"Generating features for {n_feature_maps} images...")
        
        # Use predict in batches to save memory
        batch_size = 32
        feature_map = np.empty([n_feature_maps, 512])
        
        for i in range(0, n_feature_maps, batch_size):
            end = min(i + batch_size, n_feature_maps)
            batch = vg_input[i:end]
            batch_features = vgg16.predict(batch, verbose=0)
            feature_map[i:end] = batch_features
            
            # Print progress
            if (i + batch_size) % 100 == 0 or end == n_feature_maps:
                print(f"Processed {end}/{n_feature_maps} images")

        return feature_map
    except Exception as e:
        print(f"Error generating VGG16 features: {e}")
        sys.exit(1)


def main():
    try:
        # Get the data in a Pandas dataframe
        if not os.path.exists(FLAGS.csv_file):
            print(f"Error: CSV file not found at {FLAGS.csv_file}")
            sys.exit(1)
            
        print(f"Loading dataset from {FLAGS.csv_file}...")
        try:
            raw_data = pd.read_csv(FLAGS.csv_file)
            print(f"Loaded dataset with {len(raw_data)} samples")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            sys.exit(1)

        # Validate dataset structure
        required_columns = ['emotion', 'pixels']
        for col in required_columns:
            if col not in raw_data.columns:
                print(f"Error: Required column '{col}' not found in dataset")
                sys.exit(1)

        # Create export directory if it doesn't exist
        if not os.path.exists(FLAGS.export_path):
            try:
                os.makedirs(FLAGS.export_path)
                print(f"Created directory: {FLAGS.export_path}")
            except Exception as e:
                print(f"Error creating export directory: {e}")
                sys.exit(1)

        # Data processing
        print("Processing emotions...")
        emotion_array = process_emotion(raw_data[['emotion']])
        
        print("Processing image data...")
        pixel_array = process_pixels(raw_data[['pixels']])

        # Data validation
        if len(emotion_array) != len(pixel_array):
            print(f"Error: Mismatch between number of emotions ({len(emotion_array)}) and images ({len(pixel_array)})")
            sys.exit(1)

        # split for test/train
        print("Splitting dataset into training and testing sets...")
        y_train, y_test = split_for_test(emotion_array)
        x_train_matrix, x_test_matrix = split_for_test(pixel_array)

        n_train = len(x_train_matrix)
        n_test = len(x_test_matrix)
        
        if n_train == 0:
            print("Error: No training samples available")
            sys.exit(1)
            
        if n_test == 0:
            print("Warning: No testing samples available. Using training data for evaluation.")
            n_test = n_train
            x_test_matrix = x_train_matrix
            y_test = y_train
            
        print(f"Training samples: {n_train}, Test samples: {n_test}")
        
        # Prepare input for VGG16
        print("Preparing input for VGG16...")
        x_train_input = duplicate_input_layer(x_train_matrix, n_train)
        x_test_input = duplicate_input_layer(x_test_matrix, n_test)

        # Load VGG16 model
        print("Loading VGG16 model...")
        try:
            vgg16 = VGG16(include_top=False, input_shape=(48, 48, 3), pooling='avg', weights='imagenet')
            print("VGG16 model loaded")
        except Exception as e:
            print(f"Error loading VGG16 model: {e}")
            print("Please check your internet connection and TensorFlow installation")
            sys.exit(1)

        # Generate feature maps
        print("Generating feature maps...")
        x_train_feature_map = get_vgg16_output(vgg16, x_train_matrix, n_train)
        x_test_feature_map = get_vgg16_output(vgg16, x_test_matrix, n_test)
        print("Feature maps generated")

        # Build the model
        print("Building model...")
        top_layer_model = Sequential()
        top_layer_model.add(Dense(256, input_shape=(512,), activation='relu'))
        top_layer_model.add(Dense(256, input_shape=(256,), activation='relu'))
        top_layer_model.add(Dropout(0.5))
        top_layer_model.add(Dense(128, input_shape=(256,)))
        top_layer_model.add(Dense(NUM_CLASSES, activation='softmax'))

        # Use legacy Adam optimizer for compatibility
        try:
            adamax = Adam(learning_rate=0.001)
        except TypeError:
            # Fallback for older TensorFlow versions
            adamax = Adam(lr=0.001)

        top_layer_model.compile(loss='categorical_crossentropy',
                              optimizer=adamax, metrics=['accuracy'])
        print("Model compiled")

        # Train the model
        print(f"Training model with batch_size={FLAGS.batch_size}, epochs={FLAGS.n_epochs}...")
        top_layer_model.fit(x_train_feature_map, y_train,
                          validation_data=(x_test_feature_map, y_test),
                          epochs=FLAGS.n_epochs, batch_size=FLAGS.batch_size,
                          verbose=1)
                          
        # Evaluate top layer model
        print("Evaluating model...")
        score = top_layer_model.evaluate(x_test_feature_map,
                                       y_test, batch_size=FLAGS.batch_size)

        print(f"Top layer model evaluation (test set): Loss: {score[0]:.4f}, Accuracy: {score[1]:.4f}")

        # Create final model
        print("Creating final model...")
        inputs = Input(shape=(48, 48, 3))
        vg_output = vgg16(inputs)
        model_predictions = top_layer_model(vg_output)
        final_model = Model(inputs=inputs, outputs=model_predictions)
        final_model.compile(loss='categorical_crossentropy',
                          optimizer=adamax, metrics=['accuracy'])
        
        # Evaluate final model
        print("Evaluating final model...")
        final_model_score = final_model.evaluate(x_train_input,
                                               y_train, batch_size=FLAGS.batch_size)
        print(f"Final model evaluation (train set): Loss: {final_model_score[0]:.4f}, Accuracy: {final_model_score[1]:.4f}")

        final_model_score = final_model.evaluate(x_test_input,
                                               y_test, batch_size=FLAGS.batch_size)
        print(f"Final model evaluation (test set): Loss: {final_model_score[0]:.4f}, Accuracy: {final_model_score[1]:.4f}")
        
        # Save the model
        print(f"Saving model to {FLAGS.export_path}...")
        try:
            final_model.save(FLAGS.export_path)
            print(f"Model saved successfully to {FLAGS.export_path}")
            
            # Save model summary
            summary_path = os.path.join(FLAGS.export_path, "model_summary.txt")
            with open(summary_path, 'w') as f:
                # Redirect summary to file
                old_stdout = sys.stdout
                sys.stdout = f
                final_model.summary()
                sys.stdout = old_stdout
            print(f"Model summary saved to {summary_path}")
            
            print(f"Model training completed. Accuracy on test set: {final_model_score[1]:.4f}")
        except Exception as e:
            print(f"Error saving model: {e}")
            sys.exit(1)
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
=======
    pixels_as_list = pandas_vector_to_list(pixels)

    np_image_array = []
    for index, item in enumerate(pixels_as_list):
        # 48x48
        data = np.zeros((img_size, img_size), dtype=np.uint8)
        # split space separated ints
        pixel_data = item.split()

        # 0 -> 47, loop through the rows
        for i in range(0, img_size):
            # (0 = 0), (1 = 47), (2 = 94), ...
            pixel_index = i * img_size
            # (0 = [0:47]), (1 = [47: 94]), (2 = [94, 141]), ...
            data[i] = pixel_data[pixel_index:pixel_index + img_size]

        np_image_array.append(np.array(data))

    np_image_array = np.array(np_image_array)
    # convert to float and divide by 255
    np_image_array = np_image_array.astype('float32') / 255.0
    return np_image_array


def get_vgg16_output(vgg16, array_input, n_feature_maps):
    vg_input = duplicate_input_layer(array_input, n_feature_maps)

    picture_train_features = vgg16.predict(vg_input)
    del (vg_input)

    feature_map = np.empty([n_feature_maps, 512])
    for idx_pic, picture in enumerate(picture_train_features):
        feature_map[idx_pic] = picture
    return feature_map


def duplicate_input_layer(array_input, size):
    vg_input = np.empty([size, 48, 48, 3])
    for index, item in enumerate(vg_input):
        item[:, :, 0] = array_input[index]
        item[:, :, 1] = array_input[index]
        item[:, :, 2] = array_input[index]
    return vg_input


def main():
    # used to get the session/graph data from keras
    K.set_learning_phase(0)
    # get the data in a Pandas dataframe
    raw_data = pd.read_csv(FLAGS.csv_file)

    # convert to one hot vectors
    emotion_array = process_emotion(raw_data[['emotion']])
    # convert to a 48x48 float matrix
    pixel_array = process_pixels(raw_data[['pixels']])

    # split for test/train
    y_train, y_test = split_for_test(emotion_array)
    x_train_matrix, x_test_matrix = split_for_test(pixel_array)

    n_train = int(len(x_train_matrix))
    n_test = int(len(x_test_matrix))

    x_train_input = duplicate_input_layer(x_train_matrix, n_train)
    x_test_input = duplicate_input_layer(x_test_matrix, n_test)

    # vgg 16. include_top=False so the output is the 512 and use the learned weights
    vgg16 = VGG16(include_top=False, input_shape=(48, 48, 3), pooling='avg', weights='imagenet')

    # get vgg16 outputs
    x_train_feature_map = get_vgg16_output(vgg16, x_train_matrix, n_train)
    x_test_feature_map = get_vgg16_output(vgg16, x_test_matrix, n_test)

    # build and train model
    top_layer_model = Sequential()
    top_layer_model.add(Dense(256, input_shape=(512,), activation='relu'))
    top_layer_model.add(Dense(256, input_shape=(256,), activation='relu'))
    top_layer_model.add(Dropout(0.5))
    top_layer_model.add(Dense(128, input_shape=(256,)))
    top_layer_model.add(Dense(NUM_CLASSES, activation='softmax'))

    adamax = Adamax()

    top_layer_model.compile(loss='categorical_crossentropy',
                            optimizer=adamax, metrics=['accuracy'])

    # train
    top_layer_model.fit(x_train_feature_map, y_train,
                        validation_data=(x_train_feature_map, y_train),
                        nb_epoch=FLAGS.n_epochs, batch_size=FLAGS.batch_size)
    # Evaluate
    score = top_layer_model.evaluate(x_test_feature_map,
                                     y_test, batch_size=FLAGS.batch_size)

    print("After top_layer_model training (test set): {}".format(score))

    # Merge two models and create the final_model_final_final
    inputs = Input(shape=(48, 48, 3))
    vg_output = vgg16(inputs)
    print("vg_output: {}".format(vg_output.shape))
    # TODO: the 'pooling' argument of the VGG16 model is important for this to work otherwise you will have to  squash
    # output from (?, 1, 1, 512) to (?, 512)
    model_predictions = top_layer_model(vg_output)
    final_model = Model(input=inputs, output=model_predictions)
    final_model.compile(loss='categorical_crossentropy',
                        optimizer=adamax, metrics=['accuracy'])
    final_model_score = final_model.evaluate(x_train_input,
                                             y_train, batch_size=FLAGS.batch_size)
    print("Sanity check - final_model (train score): {}".format(final_model_score))

    final_model_score = final_model.evaluate(x_test_input,
                                             y_test, batch_size=FLAGS.batch_size)
    print("Sanity check - final_model (test score): {}".format(final_model_score))
    # config = final_model.get_config()
    # weights = final_model.get_weights()

    # probably don't need to create a new model
    # model_to_save = Model.from_config(config)
    # model_to_save.set_weights(weights)
    model_to_save = final_model

    print("Model input name: {}".format(model_to_save.input))
    print("Model output name: {}".format(model_to_save.output))

    # Save Model
    builder = saved_model_builder.SavedModelBuilder(FLAGS.export_path)
    signature = predict_signature_def(inputs={'images': model_to_save.input},
                                      outputs={'scores': model_to_save.output})
    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(sess=sess,
                                             tags=[tag_constants.SERVING],
                                             signature_def_map={'predict': signature})
        builder.save()
>>>>>>> e845925b6297e856d3840686af965c6579551143


if __name__ == "__main__":
    main()
