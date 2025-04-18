# Music Genre Classification Using LSTM and MFCC

This project classifies music genres using **Long Short-Term Memory (LSTM)** networks and **Mel-frequency cepstral coefficients (MFCC)** as features. The model is trained on a dataset of music tracks, with each genre represented by audio files. The goal of the project is to predict the genre of a given music track by extracting features from the audio and passing them through a neural network.

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation and Prediction](#evaluation-and-prediction)
- [Running the Code](#running-the-code)
- [License](#license)

---

## Project Overview

This project uses **MFCC** to extract audio features and **LSTM** (Long Short-Term Memory) networks to classify music tracks into one of ten genres. The genres included are:
- Blues
- Classical
- Country
- Disco
- Hip-Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

The workflow consists of the following steps:
1. **Data Preprocessing**: Extract MFCC features from audio files and store them with their corresponding genre labels in a JSON file.
2. **Model Building**: Use an LSTM network to learn temporal dependencies in the MFCC features and classify the music genres.
3. **Model Training**: Train the model on the preprocessed data, and evaluate its performance on a test set.
4. **Prediction**: Allow users to input an audio file, and predict the genre of the track using the trained model.

---

## Requirements

To run this project, you'll need the following Python libraries:

- `librosa` for audio processing and MFCC extraction.
- `numpy` for numerical operations.
- `tensorflow` for building and training the LSTM model.
- `scikit-learn` for data splitting and evaluation.

Install them using `pip`:

```bash
pip install librosa numpy tensorflow scikit-learn
