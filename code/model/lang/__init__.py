import os
import speech_recognition as sr
from langdetect import detect
from pydub import AudioSegment
import io

# Initialize the recognizer
print('load lang')
recognizer = sr.Recognizer()

# Function to transcribe audio file to text and detect language
def transcribe_and_detect_language(audio_file: io.BytesIO):
    print('run lang')
    try:
        # Load the WAV audio file
        with sr.AudioFile(io.BufferedReader(audio_file)) as source:
            # Record the audio data
            audio_data = recognizer.record(source)

            # Use the recognizer to transcribe the audio to text
            text = recognizer.recognize_google(audio_data)

            # Detect the language of the transcribed text
            language = detect(text)

            return text, language, True

    except Exception as e:
        print(f"Music File processing {audio_file}: {e}")
        return '', '', False

if __name__ == "__main__":
    # Directory containing WAV files
    wav_folder = "D:/wellshack/xx/dataset/wav"

    # Iterate through WAV files in the directory
    for filename in os.listdir(wav_folder):
        if filename.endswith('.wav'):
            audio_file = os.path.join(wav_folder, filename)
            transcribe_and_detect_language(audio_file)
