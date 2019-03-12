import numpy as np
import librosa

from .base_extractor import BaseAudioExtractor

class EditDistanceExtractor(BaseAudioExtractor):
    def __init__(self, target_audios, threshold, sr=16000, n_dct_filters=40, n_mels=40, f_max=4000, f_min=20, n_fft=480, hop_ms=10):
        super().__init__(target_audios, threshold)
        self.n_mels = n_mels
        self.dct_filters = librosa.filters.dct(n_dct_filters, n_mels)
        self.sr = sr
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min
        self.n_fft = n_fft
        self.hop_length = sr // 1000 * hop_ms

        # TODO :: process the target audios

        # self.processed_target = ...

        # for target in target_audios:
        #     mfcc_target = self.compute_mfccs(target)
        #     quantized_target = self.vector_quantization(mfcc_target)
        # 
        #     TODO :: aggregate and update processed_target

    def compute_mfccs(self, data):
        data = librosa.feature.melspectrogram(
            data,
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            fmin=self.f_min,
            fmax=self.f_max)
        data[data > 0] = np.log(data[data > 0])
        data = [np.matmul(self.dct_filters, x) for x in np.split(data, data.shape[1], axis=1)]
        return data

    def vector_quantization(self, data):
        raise NotImplementedError

    def compute_edit_distance(self, data):
        if len(data) == 0:
            return len(self.processed_target)
        if len(self.processed_target) == 0:
            return len(data)
        M = [[i for i in range(len(self.processed_target) + 1)],
             [0 for i in range(len(self.processed_target) + 1)]]
        for i in range(1, len(data) + 1):
            idx = i % 2
            M[idx][0] = i
            for j in range(1, len(self.processed_target) + 1):
                temp = M[1 - idx][j - 1] if self.processed_target[j - 1] == data[i - 1] else M[1 - idx][j - 1] + 1
                M[idx][j] = min((M[1 - idx][j] + 1), (M[idx][j - 1] + 1), temp)
        return M[len(data) % 2][-1]


    def extract_keywords(self, data, window_ms=1000, hop_ms=250):
        selected_window = []

        current_start = 0
        while current_start + window_ms < len(data):
            window = data[current_start:current_start+window_ms]

            # TODO :: process current window
            # mfcc_window = self.compute_mfccs(window)
            # vq_window = self.vector_quantization(mfcc_window)

            # TODO :: measure the distance
            # distance = self.compute_edit_distance(vq_window)

            # if distance < self.threshold:
            #     selected_window.append(current_start)

            current_start += hop_ms

        return selected_window