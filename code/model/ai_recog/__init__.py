import copy

from model.recog import ai_human_recog
from model.recog_rawnet import ai_human_recog_rawnet

def ai_recog(audio_file):
    _, m_ai, m_hum = ai_human_recog_rawnet(copy.deepcopy(audio_file))
    _, s_ai, s_hum = ai_human_recog(copy.deepcopy(audio_file))

    print(m_ai)
    print(s_ai)

    ai = (m_ai * .8) + (s_ai * .2)
    print(ai)
    hum = (m_hum * .8) + (s_hum * .2)
    pred = ''

    if ai >= 0.55:
        pred = 'AI'
    elif ai <= 0.45:
        pred = 'Human'
    else:
        pred = 'combo'
    
    return pred, ai, hum