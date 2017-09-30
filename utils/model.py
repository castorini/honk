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