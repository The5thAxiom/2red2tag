from model.bgnoise import detectBgNoise
from model.emotion import emotional_analysis
from model.lang import transcribe_and_detect_language
from model.recog import ai_human_recog

import copy

def get_voice_type(ai_prob,human_prob,voice_type):

    if ai_prob<0.75 and ai_prob>0.35:
        return "combo"

    return voice_type

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

    voice_type, ai_prob, human_prob = ai_human_recog(copy.deepcopy(audio_file))
    emotion = emotional_analysis(transcript)
    bg_noise_level = detectBgNoise(copy.deepcopy(audio_file))
    return {
        "analysis": {
            "detectedVoice": is_voice,
            "voiceType": get_voice_type(ai_prob,human_prob,voice_type),
            "confidenceScore": {
                "aiProbability": ai_prob,
                "humanProbability": human_prob
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