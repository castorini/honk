import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

class SpeechDataset(data.Dataset):
    def __init__(self, directory, wanted_words=["silence", "unknown", "command", "random"]):
        super().__init__()
        self.directory = directory

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

def main():
    pass

if __name__ == "__main__":
    main()