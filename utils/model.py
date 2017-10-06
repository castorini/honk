from collections import ChainMap
import argparse
import hashlib
import os
import random
import sys

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

        self.output = nn.Linear(n_featmaps2 * conv_net_size, n_labels)
        self.dropout = nn.Dropout(dropout_prob)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

    @staticmethod
    def default_config():
        # Tensorflow arch
        config = {}
        config["dropout_prob"] = 0.5
        config["height"] = 101
        config["width"] = 40
        config["n_labels"] = 4
        config["n_feature_maps1"] = 64
        config["n_feature_maps2"] = 64
        config["conv1_size"] = (20, 8)
        config["conv2_size"] = (10, 4)
        config["conv1_pool"] = (2, 2)
        config["conv1_stride"] = (1, 1)
        config["conv2_stride"] = (1, 1)
        config["conv2_pool"] = (1, 1)
        return config

    def forward(self, x):
        x = F.relu(self.conv1(x.unsqueeze(1))) # shape: (batch, channels, i1, o1)
        x = self.dropout(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x)) # shape: (batch, o1, i2, o2)
        x = self.dropout(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1) # shape: (batch, o3)
        return self.output(x)

def preprocess_audio(data, n_mels, dct_filters):
    data = librosa.feature.melspectrogram(data, sr=16000, n_mels=n_mels, hop_length=160, n_fft=480, fmin=20, fmax=4000)
    data[data > 0] = np.log(data[data > 0])
    data = [np.matmul(dct_filters, x) for x in np.split(data, data.shape[1], axis=1)]
    data = np.array(data, order="F").squeeze(2).astype(np.float32)
    return torch.from_numpy(data) # shape: (frames, dct_coeffs)

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
        self.input_length = config["input_length"]
        self.timeshift_ms = config["timeshift_ms"]
        self.filters = librosa.filters.dct(config["n_dct_filters"], config["n_mels"])
        self.n_mels = config["n_mels"]
        self._audio_cache = SimpleCache(config["cache_size"])
        n_unk = len(list(filter(lambda x: x == 1, self.audio_labels)))
        self.n_silence = int(self.silence_prob * (len(self.audio_labels) - n_unk))

    @staticmethod
    def default_config():
        config = {}
        config["silence_prob"] = 0.1
        config["noise_prob"] = 0.8
        config["n_dct_filters"] = 40
        config["input_length"] = 16000
        config["n_mels"] = 40
        config["timeshift_ms"] = 100
        config["unknown_prob"] = 0.1
        config["train_pct"] = 80
        config["dev_pct"] = 10
        config["test_pct"] = 10
        config["wanted_words"] = ["command", "random"]
        config["data_folder"] = "/data/speech_dataset"
        return config

    def _timeshift_audio(self, data):
        shift = (16000 * self.timeshift_ms) // 1000
        shift = random.randint(-shift, shift)
        a = -min(0, shift)
        b = max(0, shift)
        data = np.pad(data, (a, b), "constant")
        return data[:len(data) - a] if a else data[b:]

    def preprocess(self, example, silence=False):
        if silence:
            example = "__silence__"
        if random.random() < 0.75:
            try:
                return self._audio_cache[example]
            except KeyError:
                pass
        in_len = self.input_length
        if self.bg_noise_audio:
            bg_noise = random.choice(self.bg_noise_audio)
            a = random.randint(0, len(bg_noise) - in_len - 1)
            bg_noise = bg_noise[a:a + in_len]
        else:
            bg_noise = np.zeros(in_len)

        data = np.zeros(in_len, dtype=np.float32) if silence else librosa.core.load(example, sr=16000)[0]
        data = np.pad(data, (0, max(0, in_len - len(data))), "constant")
        data = self._timeshift_audio(data)

        if random.random() < self.noise_prob or silence:
            a = 0.1
            data = np.clip(a * bg_noise + data, -1, 1)
        data = preprocess_audio(data, self.n_mels, self.filters)
        self._audio_cache[example] = data
        return data

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
        sets = [{}, {}, {}]
        unknowns = [0] * 3
        bg_noise_files = []
        unknown_files = []
        
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
                elif label == words[cls.LABEL_UNKNOWN]:
                    unknown_files.append(wav_name)
                    continue
                bucket = int(hashlib.sha1(filename.encode()).hexdigest(), 16) % 100
                if bucket < train_pct:
                    tag = 0
                elif bucket < train_pct + dev_pct:
                    tag = 1
                else:
                    tag = 2
                sets[tag][wav_name] = label

        for tag in range(len(sets)):
            unknowns[tag] = int(unknown_prob * len(sets[tag]))
        unk_files = np.random.choice(unknown_files, sum(unknowns), replace=False)
        a = 0
        for i, dataset in enumerate(sets):
            b = a + unknowns[i]
            unk_dict = {u: words[cls.LABEL_UNKNOWN] for u in unk_files[a:b]}
            dataset.update(unk_dict)
            a = b

        train_cfg = ChainMap(dict(bg_noise_files=bg_noise_files), config)
        test_cfg = ChainMap(dict(noise_prob=0), config)
        datasets = (cls(sets[0], train_cfg), cls(sets[1], test_cfg), cls(sets[2], test_cfg))
        return datasets

    def __getitem__(self, index):
        if index >= len(self.audio_labels):
            return self.preprocess(None, silence=True), 0
        return self.preprocess(self.audio_files[index]), self.audio_labels[index]

    def __len__(self):
        return len(self.audio_labels) + self.n_silence

def print_eval(name, scores, labels, loss, end="\n"):
    batch_size = labels.size(0)
    accuracy = (torch.max(scores, 1)[1].view(batch_size).data == labels.data).sum() / batch_size
    loss = loss.cpu().data.numpy()[0]
    print("{} accuracy: {:>5}, loss: {:<25}".format(name, accuracy, loss), end=end)

def set_seed(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not config["no_cuda"]:
        torch.cuda.manual_seed(seed)
    random.seed(seed)

def evaluate(config, model=None, test_loader=None):
    if not test_loader:
        _, _, test_set = SpeechDataset.splits(config)
        test_loader = data.DataLoader(test_set, batch_size=len(test_set))
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
    if not model:
        model = SpeechModel(config)
        model.load(config["input_file"])
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    model.eval()
    criterion = nn.CrossEntropyLoss()
    for model_in, labels in test_loader:
        model_in = Variable(model_in, requires_grad=False)
        if not config["no_cuda"]:
            model_in = model_in.cuda()
            labels = labels.cuda()
        scores = model(model_in)
        labels = Variable(labels, requires_grad=False)
        loss = criterion(scores, labels)
        print_eval("test", scores, labels, loss)

def train(config):
    train_set, dev_set, test_set = SpeechDataset.splits(config)
    if config["input_file"]:
        model = SpeechModel(config)
        model.load(config["input_file"])
    else:
        model = SpeechModel(config)
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    min_loss = sys.float_info.max

    train_loader = data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    dev_loader = data.DataLoader(dev_set, batch_size=min(len(dev_set), 500))
    test_loader = data.DataLoader(test_set, batch_size=min(len(test_set), 500))
    step_no = 0

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
            step_no += 1
            print_eval("train step #{}".format(step_no), scores, labels, loss)

        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
            model.eval()
            for model_in, labels in dev_loader:
                model_in = Variable(model_in, requires_grad=False)
                if not config["no_cuda"]:
                    model_in = model_in.cuda()
                    labels = labels.cuda()
                scores = model(model_in)
                labels = Variable(labels, requires_grad=False)
                loss = criterion(scores, labels)
                loss_numeric = loss.cpu().data.numpy()[0]
                if loss_numeric < min_loss:
                    min_loss = loss_numeric
                    model.save(config["output_file"])
                print_eval("dev", scores, labels, loss)
    evaluate(config, model, test_loader)

def main():
    output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model", "model.pt")
    global_config = dict(no_cuda=False, n_epochs=500, lr=0.001, batch_size=100, dev_every=10, seed=0,
        input_file="", output_file=output_file, gpu_no=1, cache_size=32768)
    builder = ConfigBuilder(
        SpeechModel.default_config(),
        SpeechDataset.default_config(),
        global_config)
    parser = builder.build_argparse()
    parser.add_argument("--mode", choices=["train", "eval"], default="train", type=str)
    config = builder.config_from_argparse(parser)
    set_seed(config)
    if config["mode"] == "train":
        train(config)
    elif config["mode"] == "eval":
        evaluate(config)

if __name__ == "__main__":
    main()
