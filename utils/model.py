import os
import random

import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

class SpeechModel(nn.Module):
    def __init__(self, **config):
        super().__init__()
        config = SpeechModel.init_default_config(config)
        n_labels = config["n_labels"]
        n_featmaps1 = config["n_feature_maps1"]
        n_featmaps2 = config["n_feature_maps2"]
        conv1_size = config["conv1_size"] # (time, frequency)
        conv2_size = config["conv2_size"]
        conv1_pool = config["conv1_pool"]
        conv2_pool = config["conv2_pool"]
        conv1_stride = config["conv1_stride"]
        conv2_stride = config["conv2_stride"]
        linear_size = config["linear_size"]
        dnn_size = config["dnn_size"]
        width = config["width"]
        height = config["height"]
        self.conv1 = nn.Conv2d(1, n_featmaps1, conv1_size, stride=conv1_stride)
        self.pool1 = nn.MaxPool2d(conv1_pool)
        self.conv2 = nn.Conv2d(n_featmaps1, n_featmaps2, conv2_size, stride=conv2_stride)
        self.pool2 = nn.MaxPool2d(conv2_pool)
        m1, m2 = conv2_size
        s, v = conv2_stride
        p, q = conv2_pool
        conv_net_size = ((width - m1 + 1) // (s * p)) * ((height - m2 + 1) // (v * q))
        self.linear = nn.Linear(n_featmaps2 * conv_net_size, linear_size)
        self.dnn = nn.Linear(linear_size, dnn_size)
        self.output = nn.Linear(dnn_size, n_labels)

    def forward(self, x):
        x = F.relu(self.conv1(x.unsqueeze(1))) # shape: (batch, channels, i1, o1)
        x = self.pool1(x)
        x = F.relu(self.conv2(x)) # shape: (batch, o1, i2, o2)
        x = self.pool2(x)
        x = self.linear(x.view(x.size(0), -1)) # shape: (batch, o3)
        x = F.relu(self.dnn(x))
        return self.output(x)

    @staticmethod
    def init_default_config(config): 
        # Full arch (~9.7M params)
        # TODO: initialize width and height
        config["n_feature_maps1"] = config.get("n_feature_maps1", 64)
        config["n_feature_maps2"] = config.get("n_feature_maps2", 64)
        config["conv1_size"] = config.get("conv1_size", (20, 8))
        config["conv2_size"] = config.get("conv2_size", (10, 4))
        config["conv1_pool"] = config.get("conv1_pool", (1, 3))
        config["conv1_stride"] = config.get("conv1_stride", (1, 1))
        config["conv2_stride"] = config.get("conv2_stride", (1, 1))
        config["conv2_pool"] = config.get("conv2_pool", (1, 1)) # no-op
        config["linear_size"] = config.get("linear_size", 32)
        config["dnn_size"] = config.get("dnn_size", 128)

class SpeechDataset(data.Dataset):
    LABEL_SILENCE = "__silence__"
    LABEL_UNKNOWN = "__unknown__"
    def __init__(self, data, bg_noise_files=[], silence_prob=0.1, noise_prob=0.8):
        super().__init__()
        self.audio_files = list(data.keys())
        self.audio_labels = list(data.values())
        self.bg_noise_audio = [librosa.core.load(file, sr=16000) for file in bg_noise_files]
        self.unknown_prob = unknown_prob
        self.silence_prob = silence_prob
        self.noise_prob = noise_prob
        self.filters = librosa.filters.dct(13, 40)

    def preprocess(self, example):
        bg_noise = random.choice(self.bg_noise_audio)
        if bg_noise:
            a = random.randint(0, len(bg_noise) - 16000 - 1)
            bg_noise = bg_noise[a:a + 16000]
        else:
            bg_noise = np.zeros(16000)

        if random.random() < self.silence_prob:
            data = bg_noise[a:a + 16000]
        else:
            data = librosa.core.load(example, sr=16000)
            if random.random() < self.noise_prob:
                a = random.random() * 0.3
                data = a * bg_noise + (1 - a) * data
        data = np.log(librosa.feature.melspectrogram(data[0], sr=data[1], n_mels=40, hop_length=160, n_fft=400))
        data = data.transpose()
        data = np.array([np.matmul(self.filters, x) for x in np.split(data)])
        data = [data[a:a + 32] for a in range(len(data) - 32)]
        return data

    @classmethod
    def splits(cls, folder, wanted_words=["command", "random"], unknown_prob=0.1, train_pct=80, dev_pct=10, test_pct=10):
        words = {word: i + 2 for i, word in enumerate(wanted_words)}
        words.update(dict(cls.LABEL_SILENCE=0, cls.LABEL_UNKNOWN=1))
        files = {0: {}, 1: {}, 2: {}}
        bg_noise_files = []

        for folder_name in os.listdir(folder):
            path_name = os.path.join(folder, folder_name)
            is_bg_noise = False
            if os.path.isfile(path_name):
                continue
            if folder_name in words:
                label = words[folder_name]
            elif folder_name == "_background_noise_":
                is_bg_noise = True
            else:
                label = words[cls.LABEL_UNKNOWN]

            for filename in os.listdir(path_name):
                wav_name = os.path.join(path_name, filename)
                if is_bg_noise and os.path.isfile(wav_name):
                    bg_noise_files.append(wav_name)
                    continue
                if label == words[cls.LABEL_UNKNOWN] and random.random() > unknown_prob:
                    continue
                bucket = hash(filename) % 100
                if bucket < train_pct:
                    tag = 0
                elif bucket < train_pct + dev_pct:
                    tag = 1
                else:
                    tag = 2
                files[tag][wav_name] = words[label]
        return (cls(files[0], bg_noise_files), cls(files[1], noise_prob=0), cls(files[2], noise_prob=0))

    def __getitem__(self, index):
        return self.preprocess(self.audio_files[index]), self.audio_labels[index]

    def __len__(self):
        return len(self.audio_labels)

def main():
    pass

if __name__ == "__main__":
    main()