import numpy as np

class BaseAudioExtractor():
    def __init__(self, target_audios, threshold):
        self.threshold = threshold
        self.target_audios = target_audios

    def extract_keywords(self, data, window_ms=1000, hop_ms=250):
        raise NotImplementedError
