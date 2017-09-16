from tensorflow.contrib.framework.python.ops import audio_ops
import base64
import cherrypy
import json
import io
import numpy as np
import os
import tensorflow as tf
import wave
import zlib

class LabelService(object):
    def __init__(self, graph_filename, labels=["_silence_", "_unknown_", "anserini", "random"]):
        with tf.gfile.FastGFile(graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
        self.labels = labels
        self.sess = tf.Session()

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

class ListenEndpoint(object):
    exposed = True
    def __init__(self, label_service, stride_size=500, min_keyword_prob=0.8, keyword="anserini"):
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

    def _encode_audio(self, wav_data):
        """Encodes raw audio data in WAVE format

        Args:
            wav_data: The raw amplitude data in standard speech command dataset format

        Returns:
            Encoded WAVE audio
        """
        buf = io.BytesIO()
        with wave.open(buf, "w") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(wav_data)
        return buf.getvalue()

    @cherrypy.tools.json_out()
    @json_in
    def POST(self, **kwargs):
        wav_data = zlib.decompress(base64.b64decode(kwargs["wav_data"]))
        labels = {label: 0. for label in self.label_service.labels}
        for data in stride(wav_data, int(2 * 16000 * self.stride_size / 1000), 2 * 16000):
            label, prob = self.label_service.label(self._encode_audio(data))
            labels[label] = max(labels[label], prob)
        return dict(contains_command=bool(labels[self.keyword] > self.min_keyword_prob))

def start(config):
    cherrypy.config.update({
        "environment": "production",
        "log.screen": True
    })
    cherrypy.config.update(config["server"])
    rest_config = {"/": {
        "request.dispatch": cherrypy.dispatch.MethodDispatcher()
    }}
    model_path = config["model_path"]
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_path)
    service = LabelService(model_path)
    cherrypy.quickstart(ListenEndpoint(service), "/listen", rest_config)