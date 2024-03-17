import os
import librosa as lr
import numpy as np
import pandas as pd

import pickle
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import layers, models


def extract_features(file_path):
    # Load audio file
    y, sr = lr.load(file_path)

    # Extract features
    mfccs = lr.feature.mfcc(y=y, sr=sr)
    chroma_stft = lr.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = lr.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = lr.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = lr.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = lr.feature.zero_crossing_rate(y)
    rms = lr.feature.rms(y=y)
    poly_features = lr.feature.poly_features(y=y, sr=sr)
    tonnetz = lr.feature.tonnetz(y=y, sr=sr)
    mel_spectrogram = lr.feature.melspectrogram(y=y, sr=sr)
    spectral_contrast = lr.feature.spectral_contrast(y=y, sr=sr)

    # Aggregate features
    return {
        'mfccs': np.mean(np.mean(mfccs, axis=1)),
        'chroma_stft': np.mean(np.mean(chroma_stft, axis=1)),
        'spectral_centroid': np.mean(spectral_centroid),
        'spectral_bandwidth': np.mean(spectral_bandwidth),
        'spectral_rolloff': np.mean(spectral_rolloff),
        'zero_crossing_rate': np.mean(zero_crossing_rate),
        'rms': np.mean(rms),
        'poly_features': np.mean(np.mean(poly_features, axis=1)),
        'tonnetz': np.mean(np.mean(tonnetz, axis=1)),
        'mel_spectrogram': np.mean(np.mean(mel_spectrogram, axis=1)),
        'spectral_contrast': np.mean(np.mean(spectral_contrast, axis=1))
    }

def cnn_extract_features(audio_path, max_length=345):
    audio_data, _ = lr.load(audio_path, sr=None)
    mfccs = lr.feature.mfcc(y=audio_data, sr=44100, n_mfcc=40)
    if mfccs.shape[1] < max_length:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
    elif mfccs.shape[1] > max_length:
        mfccs = mfccs[:, :max_length]
    return mfccs

print('load bgnoise')
cnn_model = None

# because it is deployed in a linux envirnoment and gcp will only let us write to /tmp locally
if os.name == 'posix':
    with open('model/bgnoise/bgnoise_cnn_model.keras', 'rb') as file:
        with open('/tmp/model.keras', 'wb') as file2:
            file2.write(file.read())

    cnn_model = models.load_model('/tmp/model.keras')
else:
    cnn_model = models.load_model('model/bgnoise/bgnoise_cnn_model.keras')


def detectBgNoise(audio_file):
    print(f'run bgnoise')
    features = cnn_extract_features(audio_file)
    features = np.expand_dims(features, axis=0)
    preds = cnn_model.predict(features)

    index_to_preds = {
        0: 'low',
        1: 'medium',
        2: 'high'
    }
    return index_to_preds[np.argmax(preds)]
