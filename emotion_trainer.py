import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adamax
from tensorflow.keras.utils import to_categorical

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--csv_file', type=str, required=True)
parser.add_argument('--export_path', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--n_epochs', type=int, default=1)
parser.add_argument('--debug', dest='debug', action='store_true')

args = parser.parse_args()

if args.debug:
    args.batch_size = 10
    args.n_epochs = 1

NUM_CLASSES = 7
IMG_SIZE = 48

# For small sample dataset, adjust train/test split
TRAIN_END = 5  # Use first 5 samples for training
TEST_START = 5  # Use remaining samples for testing


def split_for_test(data_list):
    train = data_list[0:TRAIN_END]
    test = data_list[TEST_START:]
    return train, test


def pandas_vector_to_list(pandas_df):
    py_list = [item[0] for item in pandas_df.values.tolist()]
    return py_list


def process_emotion(emotion):
    """
    Takes in a vector of emotions and outputs a list of emotions as one-hot vectors.
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


def process_pixels(pixels, img_size=IMG_SIZE):
    """
    Takes in a string (pixels) that has space separated ints. Will transform the ints
    to a 48x48 matrix of floats(/255).
    :param pixels: string with space separated ints
    :param img_size: image size
    :return: array of 48x48 matrices
    """
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


def duplicate_input_layer(array_input, size):
    vg_input = np.empty([size, 48, 48, 3])
    for index, item in enumerate(vg_input):
        item[:, :, 0] = array_input[index]
        item[:, :, 1] = array_input[index]
        item[:, :, 2] = array_input[index]
    return vg_input


def get_vgg16_output(vgg16, array_input, n_feature_maps):
    vg_input = duplicate_input_layer(array_input, n_feature_maps)

    picture_train_features = vgg16.predict(vg_input, verbose=0)
    del vg_input

    feature_map = np.empty([n_feature_maps, 512])
    for idx_pic, picture in enumerate(picture_train_features):
        feature_map[idx_pic] = picture
    return feature_map


def main():
    try:
        # Make sure the export path exists
        os.makedirs(args.export_path, exist_ok=True)
        
        # Get the data in a Pandas dataframe
        print(f"Loading dataset from {args.csv_file}...")
        raw_data = pd.read_csv(args.csv_file)
        print(f"Loaded dataset with {len(raw_data)} samples")

        # Convert to one hot vectors
        print("Processing emotion data...")
        emotion_array = process_emotion(raw_data[['emotion']])
        
        # Convert to a 48x48 float matrix
        print("Processing pixel data...")
        pixel_array = process_pixels(raw_data[['pixels']])

        # Split for test/train
        y_train, y_test = split_for_test(emotion_array)
        x_train_matrix, x_test_matrix = split_for_test(pixel_array)

        n_train = int(len(x_train_matrix))
        n_test = int(len(x_test_matrix))
        
        print(f"Training samples: {n_train}, Test samples: {n_test}")

        x_train_input = duplicate_input_layer(x_train_matrix, n_train)
        x_test_input = duplicate_input_layer(x_test_matrix, n_test)

        # Load VGG16 model
        print("Loading VGG16 model...")
        vgg16 = VGG16(include_top=False, input_shape=(48, 48, 3), pooling='avg', weights='imagenet')
        print("VGG16 model loaded")

        # Get VGG16 outputs
        print("Generating feature maps...")
        x_train_feature_map = get_vgg16_output(vgg16, x_train_matrix, n_train)
        x_test_feature_map = get_vgg16_output(vgg16, x_test_matrix, n_test)
        print("Feature maps generated")

        # Build and train model
        print("Building model...")
        top_layer_model = Sequential()
        top_layer_model.add(Dense(256, input_shape=(512,), activation='relu'))
        top_layer_model.add(Dense(256, activation='relu'))
        top_layer_model.add(Dropout(0.5))
        top_layer_model.add(Dense(128))
        top_layer_model.add(Dense(NUM_CLASSES, activation='softmax'))

        adamax = Adamax()

        top_layer_model.compile(loss='categorical_crossentropy',
                                optimizer=adamax, metrics=['accuracy'])
        print("Model compiled")

        # Train
        print("Training model...")
        top_layer_model.fit(
            x_train_feature_map, y_train,
            validation_data=(x_train_feature_map, y_train),
            epochs=args.n_epochs, 
            batch_size=args.batch_size,
            verbose=1
        )
        
        # Evaluate
        print("Evaluating model...")
        score = top_layer_model.evaluate(x_test_feature_map, y_test, batch_size=args.batch_size)
        print(f"Top layer model evaluation (test set): {score}")

        # Merge models and create the final model
        print("Creating final model...")
        inputs = Input(shape=(48, 48, 3))
        vg_output = vgg16(inputs)
        print(f"VGG output shape: {vg_output.shape}")
        
        model_predictions = top_layer_model(vg_output)
        final_model = Model(inputs=inputs, outputs=model_predictions)
        final_model.compile(loss='categorical_crossentropy',
                            optimizer=adamax, metrics=['accuracy'])
        
        # Evaluate final model
        print("Evaluating final model...")
        final_model_score = final_model.evaluate(x_train_input, y_train, batch_size=args.batch_size)
        print(f"Final model evaluation (train set): {final_model_score}")

        final_model_score = final_model.evaluate(x_test_input, y_test, batch_size=args.batch_size)
        print(f"Final model evaluation (test set): {final_model_score}")
        
        # Save Model using modern TensorFlow SavedModel API
        print(f"Saving model to {args.export_path}...")
        
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, 48, 48, 3], dtype=tf.float32, name='images')])
        def serving_fn(images):
            return {'scores': final_model(images, training=False)}
        
        tf.saved_model.save(
            final_model, 
            args.export_path,
            signatures={
                'serving_default': serving_fn
            }
        )
        print(f"Model saved to {args.export_path}")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 