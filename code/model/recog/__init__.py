import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.image import resize

model = load_model("model/recog/audio_classification_model_v2.h5")
print('load recog')

def ai_human_recog(audio_file):
    print('run recog')
    target_shape = (128, 128)
    classes = ["AI", "Human"]

    audio_data, sample_rate = librosa.load(audio_file, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
    mel_spectrogram = tf.reshape(mel_spectrogram, (1,) + target_shape + (1,))

    predictions = model.predict(mel_spectrogram, verbose=0)
    class_probabilities = predictions[0]
    predicted_class_index = np.argmax(class_probabilities)
    predicted_class = classes[predicted_class_index]

    # class, ai probability, human probability

    return str(predicted_class), class_probabilities[0], class_probabilities[1]