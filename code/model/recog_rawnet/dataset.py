# dataset.py
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class VoiceDataset(Dataset):
    def __init__(self, ai_folder, human_folder):
        self.files = []
        self.labels = []
        
        # AI files (label 0)
        for file in os.listdir(ai_folder):
            if file.endswith('.npy'):
                self.files.append(os.path.join(ai_folder, file))
                self.labels.append(0)
        
        # Human files (label 1)
        for file in os.listdir(human_folder):
            if file.endswith('.npy'):
                self.files.append(os.path.join(human_folder, file))
                self.labels.append(1)
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load the preprocessed numpy array
        audio_data = np.load(self.files[idx])
        label = self.labels[idx]
        return audio_data, label
