import os
import shutil
import subprocess
import uuid
import threading
import wave

from tensorflow.contrib.framework.python.ops import audio_ops
import numpy as np
import tensorflow as tf

from utils.manage_audio import AudioSnippet

class LabelService(object):
    def __init__(self, graph_filename, labels=["_silence_", "_unknown_", "command", "random"], max_memory_pct=0.01):
        with tf.gfile.FastGFile(graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
        self.labels = labels
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=max_memory_pct)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def label(self, wav_data):
        """Labels audio data as one of the specified trained labels

        Args:
            wav_data: The WAVE to label

        Returns:
            A (most likely label, probability) tuple
        """
        output = self.sess.graph.get_tensor_by_name("labels_softmax:0")
        predictions, = self.sess.run(output, {"wav_data:0": wav_data})
        return (self.labels[np.argmax(predictions)], max(predictions))


def encode_audio(self, wav_data):
    """Encodes raw audio data in WAVE format

    Args:
        wav_data: The raw amplitude data in standard speech command dataset format

    Returns:
        Encoded WAVE audio
    """
    buf = io.BytesIO()
    with wave.open(buf, "w") as f:
        set_speech_format(f)
        f.writeframes(wav_data)
    return buf.getvalue()

def set_speech_format(f):
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(16000)

def stride(array, stride_size, window_size):
    i = 0
    while i < len(array):
        yield array[i:i + window_size]
        i += stride_size

class TrainingService(object):
    def __init__(self, scripts_path, speech_dataset_path, options):
        self.freeze_script = os.path.join(scripts_path, "freeze.py")
        self.train_script = os.path.join(scripts_path, "train.py")
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

    def generate_constrastive(self, data):
        snippet = AudioSnippet(data)
        chunks = snippet.trim().chunk()
        for chunk in chunks:
            chunk.rand_pad(32000)
        return chunks

    def clear_examples(self, positive=True):
        shutil.rmtree(self.pos_directory if positive else self.neg_directory)
        self._create_dirs()

    def write_example(self, wav_data, positive=True, filename=None):
        if not filename:
            filename = "{}.wav".format(str(uuid.uuid4()))
        directory = self.pos_directory if positive else self.neg_directory
        filename = os.path.join(directory, filename)
        with wave.open(filename, "wb") as f:
            set_speech_format(f)
            f.writeframes(wav_data)
        return buf.getvalue()

    def _run_script(self, script, options):
        cmd_strs = ["python", script]
        for option, value in options.items():
            cmd_strs.append("--{}={}".format(option, value))
        subprocess.run(cmd_strs)
        self.script_running = False

    def run_train_script(self):
        if self.script_running:
            return False
        with self._run_lck:
            self.script_running = True
        threading.Thread(target=self._run_script, args=(self.train_script, self.options["train"])).start()
        return True
