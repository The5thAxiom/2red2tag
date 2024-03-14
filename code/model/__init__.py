import random

def predict(audio_file):
    
    # preprocessing
    # feature extraction
    # prediction
    # preparing the response
    
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
                "backgroundNoiseLevel": "low",
                # "laguage": "hindi" | "english" | ...,
                # "accent": "indian" | "american" | ...,
            }
        }
    }