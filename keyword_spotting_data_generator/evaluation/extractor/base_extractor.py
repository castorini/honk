import numpy as np

class BaseAudioExtractor():
    def __init__(self, target, threshold):
        self.threshold = threshold
        self.raw_target = np.array(target)

    def extract_keywords(self, data, window_ms=1000, hop_ms=250):
        raise NotImplementedError
