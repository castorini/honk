import base64
import json
import io
import os
import re
import shutil
import subprocess
import uuid
import threading
import wave
import zlib

from tensorflow.contrib.framework.python.ops import audio_ops
import cherrypy
import numpy as np
import tensorflow as tf

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

def set_speech_format(f):
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(16000)

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

    def run_script(self):
        if self.script_running:
            return False
        with self._run_lck:
            self.script_running = True
        threading.Thread(target=self._run_script, args=(self.train_script, self.options["train"])).start()
        return True

def json_in(f):
    def merge_dicts(x, y):
        z = x.copy()
        z.update(y)
        return z
    def wrapper(*args, **kwargs):
        cl = cherrypy.request.headers["Content-Length"]
        data = json.loads(cherrypy.request.body.read(int(cl)).decode("utf-8"))
        kwargs = merge_dicts(kwargs, data)
        return f(*args, **kwargs)
    return wrapper

def stride(array, stride_size, window_size):
    i = 0
    while i < len(array):
        yield array[i:i + window_size]
        i += stride_size

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

class TrainEndpoint(object):
    exposed = True
    def __init__(self, train_service):
        self.train_service = train_service

    @cherrypy.tools.json_out()
    def POST(self):
        return dict(success=self.train_service.run_script())

    @cherrypy.tools.json_out()
    def GET(self):
        return dict(in_progress=self.train_service.script_running)

class DataEndpoint(object):
    exposed = True
    def __init__(self, train_service):
        self.train_service = train_service

    @cherrypy.tools.json_out()
    @json_in
    def POST(self, **kwargs):
        wav_data = zlib.decompress(base64.b64.decode(kwargs["wav_data"]))
        positive = kwargs["positive"]
        self.train_service.write_example(wav_data, positive=positive)
        return dict(success=True)

    @cherrypy.tools.json_out()
    @json_in
    def DELETE(self, **kwargs):
        self.train_service.clear_examples(positive=kwargs["positive"])
        return dict(success=True)

class ListenEndpoint(object):
    exposed = True
    def __init__(self, label_service, stride_size=500, min_keyword_prob=0., keyword="command"):
        """The REST API endpoint that determines if audio contains the keyword.

        Args:
            label_service: The labelling service to use
            stride_size: The stride in milliseconds of the 1-second window to use. It should divide 1000 ms.
            min_keyword_prob: The minimum probability the keyword must take in order to be classified as such
            keyword: The keyword
        """
        self.label_service = label_service
        self.stride_size = stride_size
        self.min_keyword_prob = min_keyword_prob
        self.keyword = keyword

    @cherrypy.tools.json_out()
    @json_in
    def POST(self, **kwargs):
        wav_data = zlib.decompress(base64.b64decode(kwargs["wav_data"]))
        for data in stride(wav_data, int(2 * 16000 * self.stride_size / 1000), 2 * 16000):
            label, prob = self.label_service.label(encode_audio(data))
            if label == "command" and prob >= self.min_keyword_prob:
                return dict(contains_command=True)
        return dict(contains_command=False)

def make_abspath(rel_path):
    if not os.path.isabs(rel_path):
        rel_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), rel_path)
    return rel_path

def start(config):
    cherrypy.config.update({
        "environment": "production",
        "log.screen": True
    })
    cherrypy.config.update(config["server"])
    rest_config = {"/": {
        "request.dispatch": cherrypy.dispatch.MethodDispatcher()
    }}
    model_path = make_abspath(config["model_path"])
    scripts_path = make_abspath(config["scripts_path"])
    speech_dataset_path = make_abspath(config["speech_dataset_path"])

    lbl_service = LabelService(model_path)
    train_service = TrainingService(scripts_path, speech_dataset_path, config["model_options"])
    cherrypy.tree.mount(ListenEndpoint(lbl_service), "/listen", rest_config)
    cherrypy.tree.mount(DataEndpoint(train_service), "/data", rest_config)
    cherrypy.tree.mount(TrainEndpoint(train_service), "/train", rest_config)
    cherrypy.engine.start()
    cherrypy.engine.block()