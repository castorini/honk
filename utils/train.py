from collections import ChainMap
import argparse
import datetime
import os
import random
import re
import json

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable

from . import model as mod

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
        args = vars(parser.parse_known_args()[0])
        return ChainMap(args, self.default_config)

def print_eval(name, scores, labels, loss, end="\n"):
    batch_size = labels.size(0)
    accuracy = (torch.max(scores, 1)[1].view(batch_size).data == labels.data).float().sum() / batch_size
    loss = loss.item()
    # print("{} accuracy: {:>5}, loss: {:<25}".format(name, accuracy, loss), end=end)
    return accuracy

def set_seed(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not config["no_cuda"]:
        torch.cuda.manual_seed(seed)
    random.seed(seed)

def evaluate(config, model=None, test_loader=None):
    if not test_loader:
        _, _, test_set = mod.SpeechDataset.splits(config)
        test_loader = data.DataLoader(test_set, batch_size=len(test_set))
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
    if not model:
        model = config["model_class"](config)
        model.load(config["input_file"])
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    model.eval()
    if config["type"] == "eval":
        print(f"{sum(p.numel() for p in model.parameters())} parameters")
        if config["prune_pct"]:
            model.prune(config["prune_pct"], freeze=True)
            print(f"{sum(p.numel() for p in model.parameters())} parameters after slimming")
    criterion = nn.CrossEntropyLoss()
    results = []
    total = 0
    for model_in, labels in test_loader:
        model_in = Variable(model_in, requires_grad=False)
        if not config["no_cuda"]:
            model_in = model_in.cuda()
            labels = labels.cuda()
        scores = model(model_in)
        labels = Variable(labels, requires_grad=False)
        loss = criterion(scores, labels)
        results.append(print_eval("test", scores, labels, loss) * model_in.size(0))
        total += model_in.size(0)
    print("final test accuracy: {}".format(sum(results) / total))

def train(config):
    train_set, dev_set, test_set = mod.SpeechDataset.splits(config)
    model = config["model_class"](config)
    if config["input_file"]:
        model.load(config["input_file"])
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    if config["network_slimming"]:
        model.prune(config["prune_pct"])
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"][0],
        nesterov=config["use_nesterov"],
        weight_decay=config["weight_decay"],
        momentum=config["momentum"])
    schedule_steps = config["schedule"]
    schedule_steps.append(np.inf)
    sched_idx = 0
    criterion = nn.CrossEntropyLoss()
    max_acc = 0

    train_loader = data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    dev_loader = data.DataLoader(dev_set, batch_size=min(len(dev_set), 16), shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=min(len(test_set), 16), shuffle=True)
    step_no = 0

    for epoch_idx in range(config["n_epochs"]):
        for _, (model_in, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            if not config["no_cuda"]:
                model_in = model_in.cuda()
                labels = labels.cuda()
            model_in = Variable(model_in, requires_grad=False)
            scores = model(model_in)
            labels = Variable(labels, requires_grad=False)
            loss = criterion(scores, labels)
            if config["slimming_lambda"]:
                loss += model.regularization()
            loss.backward()
            optimizer.step()
            step_no += 1
            if step_no > schedule_steps[sched_idx]:
                sched_idx += 1
                print("changing learning rate to {}".format(config["lr"][sched_idx]))
                optimizer.param_groups[0]["lr"] = config["lr"][sched_idx]
            # print_eval("train step #{}".format(step_no), scores, labels, loss)

        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
            print("progress - ", epoch_idx, "/", config["n_epochs"])
            model.eval()
            accs = []
            for model_in, labels in dev_loader:
                model_in = Variable(model_in, requires_grad=False)
                if not config["no_cuda"]:
                    model_in = model_in.cuda()
                    labels = labels.cuda()
                scores = model(model_in)
                labels = Variable(labels, requires_grad=False)
                loss = criterion(scores, labels)
                accs.append(print_eval("dev", scores, labels, loss))
            avg_acc = np.mean(accs)
            print("final dev accuracy: {}".format(avg_acc))
            if avg_acc > max_acc:
                print("saving best model...")
                max_acc = avg_acc
                model.save(config["output_file"])

    generate_json(config, model)

    evaluate(config, model, test_loader)

def generate_json(config, model):
    weight_json = {}
    for key, val in model.state_dict().items():
        weight_json[key] = val.tolist()
    file_name = config['model_name'] + '_' + datetime.datetime.now().strftime("%m-%d_%H:%M") + '.js'
    file_path = 'model/' + file_name
    with open(file_path, 'w') as file:
        json.dump(weight_json, file, indent=4, sort_keys=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[x.value for x in list(mod.ConfigType)], default="cnn-trad-pool2", type=str)    
    config, _ = parser.parse_known_args()

    model_dir_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model")

    global_config = dict(
        no_cuda=False,
        n_epochs=500,
        lr=[0.001],
        schedule=[np.inf],
        batch_size=64,
        dev_every=10,
        seed=0,
        use_nesterov=False,
        input_file="",
        output_file=os.path.join(model_dir_name, config.model+".pt"),
        gpu_no=1,
        cache_size=32768,
        momentum=0.9,
        weight_decay=0.00001,
        network_slimming=False,
        prune_pct=0.,
        slimming_lambda=0.,
        model_name=config.model)
    model_name = config.model
    mod_cls = mod.find_model(config.model)
    builder = ConfigBuilder(
        mod.find_config(config.model),
        mod.SpeechDataset.default_config(),
        global_config)
    parser = builder.build_argparse()
    parser.add_argument("--type", choices=["train", "eval", "json"], default="train", type=str)
    config = builder.config_from_argparse(parser)
    config["model_class"] = mod_cls

    curr_time = datetime.datetime.now().strftime("%H_%M")
    model_file_name = model_name + "_" + curr_time+".pt"
    if config["network_slimming"]:
        model_file_name = model_name + "_" + str(round(config["prune_pct"] * 100)) + "_" + curr_time + ".pt"
    print(model_file_name)
    config["output_file"] = os.path.join(model_dir_name, model_file_name)

    set_seed(config)
    if config["type"] == "train":
        train(config)
    elif config["type"] == "eval":
        evaluate(config)
    elif config["type"] == "json":
        print("generating json from " + config["input_file"])
        model = config["model_class"](config)
        model.load(config["input_file"])
        generate_json(config, model)

if __name__ == "__main__":
    main()
