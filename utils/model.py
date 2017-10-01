from collections import ChainMap
import argparse
import os
import random

from torch.autograd import Variable
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

class ConfigBuilder(object):
    def __init__(self, *default_configs):
        self.default_config = ChainMap(*default_configs)

    def build_argparse(self):
        parser = argparse.ArgumentParser()
        for key, value in self.default_config.items():
            key = "--{}".format(key)
            if isinstance(value, tuple):
                parser.add_argument(key, default=list(value), nargs=len(value), type=type(value[0]))
            elif isinstance(value, list):
                parser.add_argument(key, default=value, nargs="+", type=type(value[0]))
            elif isinstance(value, bool) and not value:
                parser.add_argument(key, action="store_true")
            else:
                parser.add_argument(key, default=value, type=type(value))
        return parser

    def config_from_argparse(self, parser=None):
        if not parser:
            parser = self.build_argparse()
        args = vars(parser.parse_args())
        return ChainMap(args, self.default_config)

class SimpleCache(dict):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit
        self.n_keys = 0

    def __setitem__(self, key, value):
        if key in self.keys():
            super().__setitem__(key, value)
        elif self.n_keys < self.limit:
            self.n_keys += 1
            super().__setitem__(key, value)
        return value

class SpeechModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_labels = config["n_labels"]
        n_featmaps1 = config["n_feature_maps1"]
        n_featmaps2 = config["n_feature_maps2"]
        conv1_size = config["conv1_size"] # (time, frequency)
        conv2_size = config["conv2_size"]
        conv1_pool = config["conv1_pool"]
        conv2_pool = config["conv2_pool"]
        dropout_prob = config["dropout_prob"]
        conv1_stride = tuple(config["conv1_stride"])
        conv2_stride = tuple(config["conv2_stride"])
        linear_size = config["linear_size"]
        dnn_size = config["dnn_size"]
        width = config["width"]
        height = config["height"]
        self.conv1 = nn.Conv2d(1, n_featmaps1, conv1_size, stride=conv1_stride)
        self.pool1 = nn.MaxPool2d(conv1_pool)
        self.conv2 = nn.Conv2d(n_featmaps1, n_featmaps2, conv2_size, stride=conv2_stride)
        self.pool2 = nn.MaxPool2d(conv2_pool)

        m1, m2 = conv1_size
        s, v = conv1_stride
        p, q = conv1_pool
        h = ((height - m1 + 1) // (s * p)) 
        w = ((width - m2 + 1) // (v * q))

        m1, m2 = conv2_size
        s, v = conv2_stride
        p, q = conv2_pool
        conv_net_size = ((h - m1 + 1) // (s * p)) * ((w - m2 + 1) // (v * q))

        self.linear = nn.Linear(n_featmaps2 * conv_net_size, linear_size)
        self.dnn = nn.Linear(linear_size, dnn_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.output = nn.Linear(dnn_size, n_labels)

    @staticmethod
    def default_config():
        # Full arch (~9.7M params)
        config = {}
        config["dropout_prob"] = 0.5
        config["height"] = 101
        config["width"] = 40
        config["n_labels"] = 5
        config["n_feature_maps1"] = 64
        config["n_feature_maps2"] = 64
        config["conv1_size"] = (20, 8)
        config["conv2_size"] = (10, 4)
        config["conv1_pool"] = (1, 3)
        config["conv1_stride"] = (1, 1)
        config["conv2_stride"] = (1, 1)
        config["conv2_pool"] = (1, 1)
        config["linear_size"] = 32
        config["dnn_size"] = 128
        return config

    def forward(self, x):
        x = F.relu(self.conv1(x.unsqueeze(1))) # shape: (batch, channels, i1, o1)
        x = self.dropout(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x)) # shape: (batch, o1, i2, o2)
        x = self.dropout(x)
        x = self.pool2(x)
        x = self.linear(x.view(x.size(0), -1)) # shape: (batch, o3)
        x = F.relu(self.dnn(x))
        x = self.dropout(x)
        return self.output(x)

class SpeechDataset(data.Dataset):
    LABEL_SILENCE = "__silence__"
    LABEL_UNKNOWN = "__unknown__"
    def __init__(self, data, config):
        super().__init__()
        self.audio_files = list(data.keys())
        self.audio_labels = list(data.values())
        config["bg_noise_files"] = list(filter(lambda x: x.endswith("wav"), config.get("bg_noise_files", [])))
        self.bg_noise_audio = [librosa.core.load(file, sr=16000)[0] for file in config["bg_noise_files"]]
        self.unknown_prob = config["unknown_prob"]
        self.silence_prob = config["silence_prob"]
        self.noise_prob = config["noise_prob"]
        self.n_dct = config["n_dct_filters"]
        self.filters = librosa.filters.dct(config["n_dct_filters"], config["n_mels"])
        self.n_mels = config["n_mels"]
        self._audio_cache = SimpleCache(16384) # ~600 MB

    @staticmethod
    def default_config():
        config = {}
        config["silence_prob"] = 0.1
        config["noise_prob"] = 0.8
        config["n_dct_filters"] = 40
        config["input_length"] = 16000
        config["n_mels"] = 40
        config["unknown_prob"] = 0.01
        config["train_pct"] = 80
        config["dev_pct"] = 10
        config["test_pct"] = 10
        config["wanted_words"] = ["left", "bed"]
        config["data_folder"] = "/data/speech_dataset"
        return config

    def preprocess(self, example):
        try:
            return self._audio_cache[example]
        except KeyError:
            pass
        if self.bg_noise_audio:
            bg_noise = random.choice(self.bg_noise_audio)
            a = random.randint(0, len(bg_noise) - 16000 - 1)
            bg_noise = bg_noise[a:a + 16000]
        else:
            bg_noise = np.zeros(16000)

        is_silence = False
        if random.random() < self.silence_prob:
            is_silence = True
            data = bg_noise
        else:
            data = librosa.core.load(example, sr=16000)[0]
            data = np.pad(data, (0, max(0, 16000 - len(data))), "constant")
            if random.random() < self.noise_prob:
                a = random.random() * 0.1
                data = a * bg_noise + (1 - a) * data
        data = librosa.feature.melspectrogram(data, sr=16000, n_mels=self.n_mels, hop_length=160, n_fft=400)
        data[data > 0] = np.log(data[data > 0])
        data = [np.matmul(self.filters, x) for x in np.split(data, data.shape[1], axis=1)]
        data = np.array(data, order="F").squeeze(2).astype(np.float32)
        data = torch.from_numpy(data) # shape: (frames, dct_coeffs)
        self._audio_cache[example] = (data, is_silence)
        return data, is_silence

    @classmethod
    def splits(cls, config):
        folder = config["data_folder"]
        wanted_words = config["wanted_words"]
        unknown_prob = config["unknown_prob"]
        train_pct = config["train_pct"]
        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]

        words = {word: i + 2 for i, word in enumerate(wanted_words)}
        words.update({cls.LABEL_SILENCE:0, cls.LABEL_UNKNOWN:1})
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
                files[tag][wav_name] = label
        train_cfg = ChainMap(dict(bg_noise_files=bg_noise_files), config)
        test_cfg = ChainMap(dict(noise_prob=0), config)
        return (cls(files[0], train_cfg), cls(files[1], test_cfg), cls(files[2], test_cfg))

    def __getitem__(self, index):
        data, is_silence = self.preprocess(self.audio_files[index])
        label = 0 if is_silence else self.audio_labels[index]
        return data, label

    def __len__(self):
        return len(self.audio_labels)

def print_eval(name, scores, labels, loss, end="\n"):
    batch_size = labels.size(0)
    accuracy = (torch.max(scores, 1)[1].view(batch_size).data == labels.data).sum() / batch_size
    loss = loss.cpu().data.numpy()[0]
    print("{} accuracy: {:>5}, loss: {:>15}".format(name, accuracy, loss), end=end)

def train(config):
    train_set, dev_set, test_set = SpeechDataset.splits(config)
    model = SpeechModel(config)
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    
    train_loader = data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    dev_loader = data.DataLoader(dev_set, batch_size=len(dev_set))
    test_loader = data.DataLoader(test_set, batch_size=len(test_set))

    for epoch_idx in range(config["n_epochs"]):
        for batch_idx, (model_in, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            if not config["no_cuda"]:
                model_in = model_in.cuda()
                labels = labels.cuda()
            model_in = Variable(model_in, requires_grad=False)
            scores = model(model_in)
            labels = Variable(labels, requires_grad=False)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print_eval("train", scores, labels, loss, end="\r")
        print()

        if epoch_idx % 100 == 99:
            model.eval()
            for model_in, labels in dev_loader:
                model_in = Variable(model_in, requires_grad=False)
                if not config["no_cuda"]:
                    model_in = model_in.cuda()
                    labels = labels.cuda()
                scores = model(model_in)
                labels = Variable(labels, requires_grad=False)
                loss = criterion(scores, labels)
                print_eval("dev", scores, labels, loss)

    for model_in, labels in test_loader:
        model_in = Variable(model_in, requires_grad=False)
        if not config["no_cuda"]:
            model_in = model_in.cuda()
            labels = labels.cuda()
        scores = model(model_in)
        labels = Variable(labels, requires_grad=False)
        loss = criterion(scores, labels)
        print_eval("test", scores, labels, loss)

def main():
    global_config = dict(no_cuda=False, n_epochs=1200, lr=0.001, mode="train", batch_size=100,
        input_file="", output_file="", gpu_no=1)
    config = ConfigBuilder(
        SpeechModel.default_config(),
        SpeechDataset.default_config(),
        global_config).config_from_argparse()
    if config["mode"] == "train":
        train(config)
    elif config["mode"] == "eval":
        evaluate(config)

if __name__ == "__main__":
    main()
