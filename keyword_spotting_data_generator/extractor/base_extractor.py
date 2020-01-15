import inflect
import numpy as np

class BaseAudioExtractor():
    def __init__(self, keyword, threshold):
        self.keyword = keyword
        self.threshold = threshold

    def extract_keywords(self, file_name, sample_rate=16000, window_ms=1000, hop_ms=250):
        raise NotImplementedError
