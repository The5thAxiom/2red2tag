import random
from model.bgnoise.bgnoise import detectBgNoise

def predict(audio_file):
    # this dict could have more parameters which the frontend could show
    # right now this returns a random bool
    return {
        "analysis": {
            "detectedVoice": True,
            "voiceType": "human",
            "confidenceScore": {
                "apiProbability": 12,
                "humanProbability": 12
            },
            "additionalInfo": {
                "emotionalTone": "neutral",
                "backgroundNoiseLevel": detectBgNoise(audio_file),
                # "language": "hindi" | "english" | ...,
                # "accent": "indian" | "american" | ...,
            }
        }
    }