import os

from pocketsphinx import get_model_path, AudioFile, Pocketsphinx
from .base_extractor import BaseAudioExtractor

class SphinxSTTExtractor(BaseAudioExtractor):
    def __init__(self, keyword, threshold=1e-20):
        super().__init__(keyword, threshold)
        print(self.__class__.__name__, " is initialized with threshold ", threshold)

        self.kws_config = {
            'verbose': False,
            'keyphrase': self.keyword,
            'kws_threshold':threshold,
            'lm': False,
        }


    def extract_keywords(self, file_name, sample_rate=16000, window_ms=1000, hop_ms=500):

        kws_results = []

        self.kws_config['audio_file'] = file_name
        audio = AudioFile(**self.kws_config)

        for phrase in audio:
            result = phrase.segments(detailed=True)
            if len(result) == 0:
                continue
            start_time = result[0][2] * 10
            end_time = result[0][3] * 10
            # print('%4sms ~ %4sms' % (start_time, end_time))

            if len(result) > 1:
                print(result)
                raise ValueError('Result has more than one entry')

            kws_results.append((start_time, end_time))

        return kws_results
