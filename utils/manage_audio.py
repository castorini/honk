import argparse
import os
import random
import sys
import wave

import numpy as np
import pyaudio

class AudioSnippet(object):
    def __init__(self, byte_data=b"", dtype=np.int16):
        self.byte_data = byte_data
        self.dtype = dtype
        self._compute_amps()

    @staticmethod
    def join(snippets):
        snippet = AudioSnippet(dtype=snippets[0].dtype)
        for s in snippets:
            snippet.append(s)
        return snippet

    def copy(self):
        return AudioSnippet(self.byte_data)

    def chunk(self, size=32000 // 5, stride=4000):
        chunks = []
        i = 0
        while i + size < len(self.byte_data):
            chunks.append(AudioSnippet(self.byte_data[i:i + size]))
            i += stride
        return chunks

    def rand_pad(self, total_length, noise_level=0.001):
        space = total_length - len(self.byte_data)
        len_a = random.randint(0, space)
        len_b = space - len_a
        rand_a = (noise_level * np.random.random(size=len_a).astype(self.dtype)).astype(self.dtype)
        rand_b = (noise_level * np.random.random(size=len_b).astype(self.dtype)).astype(self.dtype)
        self.byte_data = b"".join([rand_a, self.byte_data, rand_b])
        self._compute_amps()

    def trim_window(self, window_size):
        nbytes = len(self.byte_data) // len(self.amplitudes)
        window_size //= nbytes
        cc_window = np.ones(window_size)
        clip_energy = np.correlate(np.abs(self.amplitudes), cc_window)
        smooth_window_size = 1000
        smooth_window = np.ones(smooth_window_size)
        scale = len(self.amplitudes) / (len(self.amplitudes) - smooth_window_size + 1)
        clip_energy2 = np.correlate(clip_energy, smooth_window)
        window_i = int(np.argmax(clip_energy2) * scale)
        window_i = max(0, window_i - window_i % nbytes)
        self.amplitudes = self.amplitudes[window_i:window_i + window_size]
        window_i *= nbytes
        self.byte_data = self.byte_data[window_i:window_i + window_size * nbytes]

    def ltrim(self, limit=0.01):
        if not self.byte_data:
            return
        i = 0
        for i in range(len(self.amplitudes)):
            if self.amplitudes[i] > limit:
                break
        nbytes = len(self.byte_data) // len(self.amplitudes)
        i = max(0, i - i % nbytes)
        self.amplitudes = self.amplitudes[i:]
        self.byte_data = self.byte_data[i * nbytes:]
        return self

    def trim(self, limit=0.01):
        self.ltrim(limit)
        self.rtrim(limit)
        return self

    def rtrim(self, limit=0.01):
        if not self.byte_data:
            return
        i = len(self.amplitudes)
        for i in range(len(self.amplitudes) - 1, -1, -1):
            if self.amplitudes[i] > limit:
                break
        nbytes = len(self.byte_data) // len(self.amplitudes)
        i = min(len(self.amplitudes), i + (nbytes - i % nbytes))
        self.amplitudes = self.amplitudes[:i]
        self.byte_data = self.byte_data[:i * nbytes]
        return self

    def _compute_amps(self):
        self.amplitudes = np.frombuffer(self.byte_data, self.dtype).astype(float) / np.iinfo(self.dtype).max

    def append(self, snippet):
        self.byte_data = b''.join([self.byte_data, snippet.byte_data])
        self._compute_amps()
        return self

    def amplitude_rms(self, start=0, end=-1):
        return np.sqrt(np.mean([a * a for a in self.amplitudes[start:end]]))

class AudioSnippetGenerator(object):
    def __init__(self, sr=16000, fmt=pyaudio.paInt16, chunk_size=1024, channels=1):
        self.sr = sr
        self.fmt = fmt
        self.channels = channels
        self.chunk_size = chunk_size
        self.stream = None

    def __enter__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.fmt, channels=self.channels, rate=self.sr, input=True, 
          frames_per_buffer=self.chunk_size)
        return self

    def __exit__(self, *args):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        self.stream = None

    def __iter__(self):
        if not self.stream:
            raise ValueError("Audio stream isn't open")
        return self

    def __next__(self):
        return AudioSnippet(self.stream.read(self.chunk_size))

def print_sound_level():
    with AudioSnippetGenerator() as generator:
        for audio in generator:
            print("Sound level: {}".format(audio.amplitude_rms()), end="\r")

def record_speech_sequentially(file_name_prefix="output", min_sound_lvl=0.01, speech_timeout_secs=1., i=0):
    """Records audio in sequential audio files.

    Args:
        file_name_prefix: The prefix of the output filenames
        min_sound_lvl: The minimum sound level as measured by root mean square
        speech_timeout_secs: Timeout of audio after that duration of silence as measured by min_sound_lvl
        i: The beginning index of sequence
    """
    while True:
        input("Input any key to record: ")
        with AudioSnippetGenerator() as generator:
            timeout_len = int(speech_timeout_secs * generator.sr / generator.chunk_size)
            active_count = timeout_len
            curr_snippet = None
            for audio in generator:
                if curr_snippet:
                    curr_snippet.append(audio)
                else:
                    curr_snippet = audio
                if audio.amplitude_rms() < min_sound_lvl:
                    active_count -= 1
                else:
                    active_count = timeout_len
                print("Time left: {:<10}".format(active_count), end="\r")
                if active_count == 0:
                    output_name = "{}.{}.wav".format(file_name_prefix, i)
                    i += 1
                    with wave.open(output_name, "w") as f:
                        f.setnchannels(1)
                        f.setsampwidth(generator.audio.get_sample_size(generator.fmt))
                        f.setframerate(generator.sr)
                        f.writeframes(curr_snippet.byte_data)
                    print("Saved to {}".format(output_name))
                    break

def clean_dir(directory=".", cutoff_ms=1000):
    """Trims all audio in directory to the loudest window of length cutoff_ms. 1 second is consistent 
    with the speech command dataset.

    Args:
        directory: The directory containing all the .wav files. Should have nothing but .wav
        cutoff_ms: The length of time to trim audio to in milliseconds
    """
    for filename in os.listdir(directory):
        fullpath = os.path.join(directory, filename)
        try:
            with wave.open(fullpath) as f:
                n_channels = f.getnchannels()
                width = f.getsampwidth()
                rate = f.getframerate()
                n_samples = int((cutoff_ms / 1000) * rate)
                snippet = AudioSnippet(f.readframes(10 * n_samples))
            snippet.trim_window(n_samples * width)
            with wave.open(fullpath, "w") as f:
                f.setnchannels(n_channels)
                f.setsampwidth(width)
                f.setframerate(rate)
                f.writeframes(snippet.byte_data)
            print("Trimmed {} to {} ms".format(filename, cutoff_ms))
        except (wave.Error, IsADirectoryError, PermissionError) as e:
            pass

def main():
    parser = argparse.ArgumentParser()
    commands = dict(record=record_speech_sequentially, trim=clean_dir, listen=print_sound_level)
    parser.add_argument("subcommand")
    def print_sub_commands():
        print("Subcommands: {}".format(", ".join(commands.keys())))
    if len(sys.argv) <= 1:
        print_sub_commands()
        return
    subcommand = sys.argv[1]
    if subcommand == "record":
        parser.add_argument(
            "--output-prefix",
            type=str,
            default="output",
            help="Prefix of the output audio sequence")
        parser.add_argument(
            "--min-sound-lvl",
            type=float,
            default=0.01,
            help="Minimum sound level at which audio is not considered silent")
        parser.add_argument(
            "--timeout-seconds",
            type=float,
            default=1.,
            help="Duration of silence after which recording halts")
        parser.add_argument(
            "--output-begin-index",
            type=int,
            default=0,
            help="Starting sequence number")
        flags, _ = parser.parse_known_args()
        record_speech_sequentially(file_name_prefix=flags.output_prefix, i=flags.output_begin_index, 
            min_sound_lvl=flags.min_sound_lvl, speech_timeout_secs=flags.timeout_seconds)
    elif subcommand == "trim":
        parser.add_argument(
            "directory",
            type=str,
            nargs="?",
            default=".",
            help="Trim the directory's audio files")
        flags, _ = parser.parse_known_args()
        clean_dir(flags.directory)
    elif subcommand == "listen":
        print_sound_level()
    else:
        print_sub_commands()

if __name__ == "__main__":
    main()