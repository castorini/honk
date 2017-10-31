import io
import os
import shutil
import subprocess
import uuid
import threading
import wave

import librosa
import numpy as np
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    pass
try:
    import onnx
    import onnx_caffe2.backend
except ImportError:
    pass

from utils.manage_audio import AudioSnippet, preprocess_audio
try:
    import utils.model as model
except ImportError:
    pass

def _softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

class LabelService(object):
    def evaluate(self, speech_dirs, indices=[]):
        dir_labels = {}
        if indices:
            real_labels = [self.labels[i] for i in indices]
        else:
            real_labels = [os.dirname(d) for d in speech_dirs]
        for i, label in enumerate(real_labels):
            if label not in self.labels:
                real_labels[i] = "_unknown_"
            dir_labels[speech_dirs[i]] = real_labels[i]
        accuracy = []
        for folder in speech_dirs:
            for filename in os.listdir(folder):
                fp = os.path.join(folder, filename)
                with wave.open(fp) as f:
                    b_data = f.readframes(16000)
                label, _ = self.label(b_data)
                accuracy.append(int(label == dir_labels[folder]))
        return sum(accuracy) / len(accuracy)                

    def label(self, wav_data):
        raise NotImplementedError

class Caffe2LabelService(LabelService):
    def __init__(self, onnx_filename, labels):
        self.labels = labels
        self.model_filename = onnx_filename
        self.filters = librosa.filters.dct(40, 40)
        self._graph = onnx.load(onnx_filename)
        self._in_name = self._graph.graph.input[0].name
        self.model = onnx_caffe2.backend.prepare(self._graph)

    def label(self, wav_data):
        wav_data = np.frombuffer(wav_data, dtype=np.int16) / 32768.
        model_in = np.expand_dims(preprocess_audio(wav_data, 40, self.filters), 0)
        model_in = np.expand_dims(model_in, 0)
        model_in = model_in.astype(np.float32)
        predictions = _softmax(self.model.run({self._in_name: model_in})[0])
        return (self.labels[np.argmax(predictions)], np.max(predictions))

class TorchLabelService(LabelService):
    def __init__(self, model_filename, no_cuda=False, labels=["_silence_", "_unknown_", "command", "random"]):
        self.labels = labels
        self.model_filename = model_filename
        self.no_cuda = no_cuda
        self.filters = librosa.filters.dct(40, 40)
        self.reload()

    def reload(self):
        config = model.find_config(model.ConfigType.CNN_TRAD_POOL2)
        config["n_labels"] = len(self.labels)
        self.model = model.SpeechModel(config)
        if not self.no_cuda:
            self.model.cuda()
        self.model.load(self.model_filename)
        self.model.eval()

    def label(self, wav_data):
        """Labels audio data as one of the specified trained labels

        Args:
            wav_data: The WAVE to label

        Returns:
            A (most likely label, probability) tuple
        """
        wav_data = np.frombuffer(wav_data, dtype=np.int16) / 32768.
        model_in = torch.from_numpy(preprocess_audio(wav_data, 40, self.filters)).unsqueeze(0)
        model_in = torch.autograd.Variable(model_in, requires_grad=False)
        if not self.no_cuda:
            model_in = model_in.cuda()
        predictions = F.softmax(self.model(model_in).squeeze(0).cpu()).data.numpy()
        return (self.labels[np.argmax(predictions)], np.max(predictions))

def stride(array, stride_size, window_size):
    i = 0
    while i + window_size <= len(array):
        yield array[i:i + window_size]
        i += stride_size

class TrainingService(object):
    def __init__(self, train_script, speech_dataset_path, options):
        self.train_script = train_script
        self.neg_directory = os.path.join(speech_dataset_path, "random")
        self.pos_directory = os.path.join(speech_dataset_path, "command")
        self.options = options
        self._run_lck = threading.Lock()
        self.script_running = False
        self._create_dirs()

    def _create_dirs(self):
        if not os.path.exists(self.neg_directory):
            os.makedirs(self.neg_directory)
        if not os.path.exists(self.pos_directory):
            os.makedirs(self.pos_directory)

    def generate_contrastive(self, data):
        snippet = AudioSnippet(data)
        phoneme_chunks = AudioSnippet(data).chunk_phonemes()
        phoneme_chunks2 = AudioSnippet(data).chunk_phonemes(factor=0.8, group_threshold=500)
        joined_chunks = []
        for i in range(len(phoneme_chunks) - 1):
            joined_chunks.append(AudioSnippet.join([phoneme_chunks[i], phoneme_chunks[i + 1]]))
        if len(joined_chunks) == 1:
            joined_chunks = []
        if len(phoneme_chunks) == 1:
            phoneme_chunks = []
        if len(phoneme_chunks2) == 1:
            phoneme_chunks2 = []
        chunks = [c.copy() for c in phoneme_chunks2]
        for chunk_list in (phoneme_chunks, joined_chunks, phoneme_chunks2):
            for chunk in chunk_list:
                chunk.rand_pad(32000)
        for chunk in chunks:
            chunk.repeat_fill(32000)
            chunk.rand_pad(32000)
        chunks.extend(phoneme_chunks)
        chunks.extend(phoneme_chunks2)
        chunks.extend(joined_chunks)
        return chunks

    def clear_examples(self, positive=True, tag=""):
        directory = self.pos_directory if positive else self.neg_directory
        if not tag:
            shutil.rmtree(directory)
            self._create_dirs()
        else:
            for name in os.listdir(directory):
                if name.startswith("{}-".format(tag)):
                    os.unlink(os.path.join(directory, name))

    def write_example(self, wav_data, positive=True, filename=None, tag=""):
        if tag:
            tag = "{}-".format(tag)
        if not filename:
            filename = "{}{}.wav".format(tag, str(uuid.uuid4()))
        directory = self.pos_directory if positive else self.neg_directory
        filename = os.path.join(directory, filename)
        AudioSnippet(wav_data).save(filename)

    def _run_script(self, script, options):
        cmd_strs = ["python", script]
        for option, value in options.items():
            cmd_strs.append("--{}={}".format(option, value))
        subprocess.run(cmd_strs)

    def _run_training_script(self, callback):
        with self._run_lck:
            self.script_running = True
        self._run_script(self.train_script, self.options)
        if callback:
            callback()
        self.script_running = False

    def run_train_script(self, callback=None):
        if self.script_running:
            return False
        threading.Thread(target=self._run_training_script, args=(callback,)).start()
        return True
