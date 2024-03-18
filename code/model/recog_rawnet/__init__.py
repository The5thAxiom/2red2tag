import torch
from model.recog_rawnet.model import RawNet
import librosa
import numpy as np

print('load recog_rawnet')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d_args = {'nb_samp': 64600,
  'first_conv': 1024 ,  # no. of filter coefficients
  'in_channels': 1,
  'filts': [20, [20, 20], [20, 128], [128, 128]], # no. of filters channel in residual blocks
  'blocks': [2, 4],
  'nb_fc_node': 1024,
  'gru_node': 1024,
  'nb_gru_layer': 3}
model = RawNet(d_args,device)

model.load_state_dict(torch.load('model/recog_rawnet/best_model_6600.pth'))

model.eval()

def ai_human_recog_rawnet(audio_file):
    print('run recog_rawnet')
    path = audio_file
    sample_rate = 16000
    duration =4
    data, sr = librosa.load(path, sr=sample_rate)
    # Pad or trim the audio signal
    if len(data) > sample_rate*duration:
        data = data[:sample_rate*duration]
    else:
        data = np.pad(data, (0, max(0, sample_rate*duration - len(data))), "constant")

    input_tensor = torch.from_numpy(data)
    input_tensor = input_tensor.to(device)
    input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output, _ = model(input_tensor)

    _, predicted = torch.max(output.data, 1)

    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_class_indices = torch.argmax(probabilities, dim=1)
    class_names = ['AI', 'Human']
    predicted_classes = [class_names[i] for i in predicted_class_indices]

    return predicted_classes[0], probabilities[0][0].item(), probabilities[0][1].item()
