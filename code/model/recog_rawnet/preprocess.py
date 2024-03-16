# import os
# import librosa
# import numpy as np

# class Preprocessor:
#     def __init__(self, sample_rate=16000, duration=4, save_path='processed_data'):
#         self.sample_rate = sample_rate
#         self.duration = duration
#         self.save_path = save_path

#     def preprocess_dataset(self, folder_path):
#         os.makedirs(self.save_path, exist_ok=True)
#         for file in os.listdir(folder_path):
#             try:
#                 # Load the audio file
#                 path = os.path.join(folder_path, file)
#                 data, sr = librosa.load(path, sr=self.sample_rate)
                
#                 # Pad or trim the audio signal
#                 if len(data) > self.sample_rate * self.duration:
#                     data = data[:self.sample_rate * self.duration]
#                 else:
#                     data = np.pad(data, (0, max(0, self.sample_rate * self.duration - len(data))), "constant")
                
#                 # Save processed data
#                 filename = os.path.splitext(file)[0]
#                 np.save(os.path.join(self.save_path, filename), data)
                
#                 print(f"Processed {file}")
#             except Exception as e:
#                 print(f"Could not process {file} due to {e}.")

#     def preprocess(self, folder_path):
#         self.preprocess_dataset(folder_path)


# if __name__ == '__main__':
#     import argparse

#     parser = argparse.ArgumentParser(description='Preprocess audio dataset.')
#     parser.add_argument('--folder_path', type=str, required=True, help='Directory containing audio files.')
#     parser.add_argument('--save_path', type=str, default='processed_data', help='Directory to save processed files.')
#     args = parser.parse_args()

#     preprocessor = Preprocessor(save_path=args.save_path)
#     preprocessor.preprocess(folder_path=args.folder_path)



# preprocess.py
import librosa
import numpy as np
import os
import argparse

def preprocess_dataset(folder_path, sample_rate=16000, duration=4, save_path='processed_data'):
    os.makedirs(save_path, exist_ok=True)
    for file in os.listdir(folder_path):
        try:
            # Load the audio file
            path = os.path.join(folder_path, file)
            data, sr = librosa.load(path, sr=sample_rate)
            
            # Pad or trim the audio signal
            if len(data) > sample_rate*duration:
                data = data[:sample_rate*duration]
            else:
                data = np.pad(data, (0, max(0, sample_rate*duration - len(data))), "constant")
            
            # Save processed data
            filename = os.path.splitext(file)[0]
            np.save(os.path.join(save_path, filename), data)
            
            print(f"Processed {file}")
        except Exception as e:
            print(f"Could not process {file} due to {e}.")

def main():
    parser = argparse.ArgumentParser(description='Preprocess audio dataset.')
    parser.add_argument('--folder_path', type=str, required=True, help='Directory containing audio files.')
    parser.add_argument('--save_path', type=str, default='processed_data', help='Directory to save processed files.')
    args = parser.parse_args()

    preprocess_dataset(folder_path=args.folder_path, save_path=args.save_path)

if __name__ == '__main__':
    main()

#python preprocess.py --folder_path "D:\wellshack\LibriSeVoc\LibriSeVoc\zz_large_ai" --save_path "processed_ai"
#python preprocess.py --folder_path "D:\wellshack\LibriSeVoc\LibriSeVoc\zz_large_human" --save_path "processed_human"