from model.bgnoise import detectBgNoise
from model.emotion import emotional_analysis
from model.lang import transcribe_and_detect_language

import copy

def predict(audio_file):
    # this dict could have more parameters which the frontend could show
    # right now this returns a random bool
    transcript, language, is_voice = transcribe_and_detect_language(copy.deepcopy(audio_file))

    if not is_voice:
        return {
            "analysis": {
                "detectedVoice": is_voice
            }
        }

    emotion = emotional_analysis(transcript)
    bg_noise_level = detectBgNoise(copy.deepcopy(audio_file))
    return {
        "analysis": {
            "detectedVoice": is_voice,
            "voiceType": "human",
            "confidenceScore": {
                "apiProbability": 12,
                "humanProbability": 12
            },
            "additionalInfo": {
                "emotionalTone": emotion,
                "backgroundNoiseLevel": bg_noise_level,
                "language": language,
                "transcript": transcript
                # "accent": "indian" | "american" | ...,
            }
        }
    }