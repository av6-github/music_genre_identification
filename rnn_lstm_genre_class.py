import json
import numpy as np
from sklearn.model_selection import train_test_split
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import librosa
keras = tf.keras
import math


DATASET_PATH = "data.json"

def load_data(dataset_path): # loads data.json, converts mfcc and labels to numpy arrays

    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # convert lists in json file to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    
    return X, y

def prepare_datasets(test_size, validation_size): # splits data into training validation and testing

    # load data
    X, y = load_data(DATASET_PATH)

    # create train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # create train validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    return X_train, X_validation, X_test , y_train, y_validation, y_test

def build_model(input_shape):
    # create model
    model = keras.Sequential()

    # lstm layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    # dense layers
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model

def predict(model, X, y):

    with open("data.json", "r") as fp:
        data = json.load(fp)

    genres = np.array(data['mapping'])

    X = X[np.newaxis, ...]

    # prediction = [[0.1, 0.2, ....]]
    prediction = model.predict(X) # X -> (1, 130, 13, 1)

    #extract index with max value
    predicted_index = np.argmax(prediction, axis=1) # [ind]
    print(f"Expected genre: {genres[y]}, Predicted genre: {genres[predicted_index]}")

def predict_from_file(model, file_path, data_path="data.json", n_mfcc=13, n_fft=2048, hop_length=512):
    with open(data_path, "r") as fp:
        data = json.load(fp)
    genres = np.array(data["mapping"])

    SAMPLE_RATE = 22050
    DURATION = 30
    num_segments = 10
    SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_per_segment = math.ceil(num_samples_per_segment / hop_length)

    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    predictions = []

    for s in range(num_segments):
        start_sample = s * num_samples_per_segment
        finish_sample = start_sample + num_samples_per_segment

        mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                    sr=sr,
                                    n_fft=n_fft,
                                    n_mfcc=n_mfcc,
                                    hop_length=hop_length)
        mfcc = mfcc.T

        if len(mfcc) == expected_num_mfcc_per_segment:
            mfcc = mfcc[np.newaxis, ...]
            prediction = model.predict(mfcc, verbose=0)
            predicted_index = np.argmax(prediction, axis=1)[0]
            predictions.append(predicted_index)

    if predictions:
        # Majority vote
        final_prediction = max(set(predictions), key=predictions.count)
        print(f"Predicted genre: {genres[final_prediction]}")
    else:
        print("Could not make a prediction (audio too short or invalid format).")


if __name__ == "__main__":

    # create train, validation and test sets

    X_train, X_validation, X_test , y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # build rnn network

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)

    # compile network
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    # train cnn
    model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

    # evaluate cnn on test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"accuracy on test set is: {test_accuracy}")


    user_file = input("Enter the path to the audio file: ")
    predict_from_file(model, user_file)

