from threading import Thread

from model.bgnoise import detectBgNoise
from model.emotion import emotional_analysis
from model.lang import transcribe_and_detect_language
from model.ai_recog import ai_recog

import copy

class Predictor:

    def __init__(self, audio_file):
        self.audio_file = audio_file

    def get_voice_type(self):
        if self.ai_prob<0.75 and self.ai_prob>0.35:
            return "combo"
        return self.voice_type
    
    def _lang(self):
        self.transcript, self.language, self.is_voice = transcribe_and_detect_language(copy.deepcopy(self.audio_file))
    
    def _bgnoise(self):
        self.bg_noise_level = detectBgNoise(copy.deepcopy(self.audio_file))

    def _emotion(self):
        self.emotion_type = emotional_analysis(self.transcript)

    def _ai_recog(self):
        self.voice_type, self.ai_prob, self.human_prob = ai_recog(copy.deepcopy(self.audio_file))

    def predict(self):
        self._lang()
        
        if not self.is_voice:
            return {
                "analysis": {
                    "detectedVoice": self.is_voice
                }
            }

        ## multithreaded code
        ai_recog_thread = Thread(target=self._ai_recog, name='ai_recog_thread')
        emotion_thread = Thread(target=self._emotion, name='emotion_thread')
        bgnoise_thread = Thread(target=self._bgnoise, name='bgnoise_thread')

        ai_recog_thread.start()
        emotion_thread.start()
        bgnoise_thread.start()

        ai_recog_thread.join()
        emotion_thread.join()
        bgnoise_thread.join()

        ## single threaded code
        # self._ai_recog()
        # self._emotion()
        # self._bgnoise()

        return {
            "analysis": {
                "detectedVoice": self.is_voice,
                "voiceType": self.get_voice_type(),
                "confidenceScore": {
                    "aiProbability": self.ai_prob,
                    "humanProbability": self.human_prob
                },
                "additionalInfo": {
                    "emotionalTone": self.emotion_type,
                    "backgroundNoiseLevel": self.bg_noise_level,
                    "language": self.language,
                    "transcript": self.transcript
                }
            }
        }