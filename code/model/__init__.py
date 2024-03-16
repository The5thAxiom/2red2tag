import random
from model.bgnoise import detectBgNoise
from model.emotion import emotional_analysis

def predict(audio_file):
    # this dict could have more parameters which the frontend could show
    # right now this returns a random bool
    transcript = 'i hate google cloud'
    return {
        "analysis": {
            "detectedVoice": True,
            "voiceType": "human",
            "confidenceScore": {
                "apiProbability": 12,
                "humanProbability": 12
            },
            "additionalInfo": {
                "emotionalTone": emotional_analysis(transcript),
                "backgroundNoiseLevel": detectBgNoise(audio_file),
                # "language": "hindi" | "english" | ...,
                # "accent": "indian" | "american" | ...,
            }
        }
    }