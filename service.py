import io
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
        self.sess = None
        self.labels = labels
        self.graph_filename = graph_filename
        self.max_memory_pct = max_memory_pct
        self.reload()

    def reload(self):
        if self.sess:
            self.sess.close()
        with tf.gfile.FastGFile(self.graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.max_memory_pct)
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


def encode_audio(wav_data):
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

    def generate_contrastive(self, data):
        snippet = AudioSnippet(data)
        chunks = snippet.trim().chunk(3000, 1000)
        if len(chunks) == 1:
            return []
        long_chunks = snippet.chunk(5000, 1000)
        if len(long_chunks) > 1:
            chunks.extend(long_chunks)
        long_chunks = snippet.chunk(8000, 1000)
        if len(long_chunks) > 1:
            chunks.extend(long_chunks)
        chunks2 = snippet.chunk(5000, 1000)
        for chunk in chunks:
            chunk.rand_pad(32000)
        for chunk in chunks2:
            chunk.repeat_fill(32000)
            chunk.rand_pad(32000)
        chunks = chunks * 2
        chunks.extend(chunks2)
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
        with wave.open(filename, "wb") as f:
            set_speech_format(f)
            f.writeframes(wav_data)

    def _run_script(self, script, options):
        cmd_strs = ["python", script]
        for option, value in options.items():
            cmd_strs.append("--{}={}".format(option, value))
        subprocess.run(cmd_strs)

    def _run_training_script(self, callback):
        with self._run_lck:
            self.script_running = True
        self._run_script(self.train_script, self.options["train"])
        self._run_script(self.freeze_script, self.options["freeze"])
        if callback:
            callback()
        self.script_running = False

    def run_train_script(self, callback=None):
        if self.script_running:
            return False
        threading.Thread(target=self._run_training_script, args=(callback,)).start()
        return True
