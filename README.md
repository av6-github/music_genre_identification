# Music Genre Classification with RNN/LSTM

This project is a deep learning pipeline that classifies music genres from audio files using Mel Frequency Cepstral Coefficients (MFCC) and a Recurrent Neural Network (RNN) with LSTM layers.

---

## 📁 Folder Structure
```
dataset_compr/
├── genres/
│   ├── blues/
│   ├── classical/
│   ├── country/
│   ├── disco/
│   ├── hiphop/
│   ├── jazz/
│   ├── metal/
│   ├── pop/
│   ├── reggae/
│   └── rock/
data.json
preprocess.py
rnn_lstm_genre_classif.py
```

---

## 🎯 Goal
To classify an audio file into one of 10 music genres using a deep learning model trained on MFCC features.

---

## 📊 Data Preparation
The dataset consists of audio files organized into genre-specific folders. Each file is:

- 30 seconds long
- Sampled at 22050 Hz

### Feature Extraction (`preprocess.py`):
- Loads each audio file
- Splits it into 10 segments
- Extracts MFCC features for each segment
- Saves the MFCC data and corresponding labels into `data.json`

Run:
```bash
python preprocess.py
```

---

## 🧠 Model Architecture
Defined in `rnn_lstm_genre_classif.py`:

- **Input**: Shape `(segments, MFCCs)` = `(130, 13)`
- **LSTM Layer 1**: 64 units, returns sequences
- **LSTM Layer 2**: 64 units
- **Dense Layer**: 64 units, ReLU activation
- **Dropout Layer**: 0.3 dropout rate
- **Output Layer**: 10 units, softmax activation

```python
model = keras.Sequential()
model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
model.add(keras.layers.LSTM(64))
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(10, activation="softmax"))
```

---

## 🏋️ Training the Model
When you run `rnn_lstm_genre_classif.py`, it:

1. Loads data from `data.json`
2. Splits it into train, validation, and test sets
3. Compiles and trains the model for 30 epochs

```bash
python rnn_lstm_genre_classif.py
```

### Output:
```
Epoch 1/30
...
accuracy on test set is: 0.835
```

---

## ✅ Evaluation and Prediction
After training:

- The model is evaluated on the test set.
- You are prompted to input a new file path for prediction.

### Evaluation:
Automatically prints accuracy after training.

### Prediction:
- Enter a WAV file path (30 sec, 22050 Hz).
- MFCCs are extracted.
- The model predicts the genre.

Example:
```
Enter the file path for the audio file: path/to/test.wav
Predicted genre: rock
```

---

## 🚀 Running the Code
### 1. Preprocess Audio Files
```bash
python preprocess.py
```

### 2. Train and Test the Model
```bash
python rnn_lstm_genre_classif.py
```

You’ll be prompted for a file path to test the model.

---

## 📦 Requirements
- Python 3.x
- librosa
- numpy
- tensorflow
- scikit-learn

Install via pip:
```bash
pip install librosa numpy tensorflow scikit-learn
```
---

## 📌 Notes
- Input files must be 30 seconds long, sampled at 22050 Hz.
- Useful for understanding how RNNs and MFCCs can be combined for audio classification.

---
📄 License

This project is licensed under the MIT License. You are free to use, modify, and distribute this code for personal and commercial purposes with proper attribution.


