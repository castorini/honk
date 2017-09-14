import argparse
import os
import pyaudio
import numpy as np
import wave

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

  def trim(self, limit=0.01):
    self.ltrim(limit)
    self.rtrim(limit)

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
    self.stream = self.audio.open(format=self.fmt, channels=self.channels, rate=self.sr, input=True, frames_per_buffer=self.chunk_size)
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
      print(audio.amplitude_rms(), end="\r")

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
          #curr_snippet.trim()
          i += 1
          with wave.open(output_name, "w") as f:
            f.setnchannels(1)
            f.setsampwidth(generator.audio.get_sample_size(generator.fmt))
            f.setframerate(generator.sr)
            f.writeframes(curr_snippet.byte_data)
          print("Saved to {}".format(output_name))
          break

def clean_dir(directory=".", cutoff_ms=1000):
  """Trims all audio in directory to cutoff_ms. 1 second is consistent with speech command dataset.

  Args:
    directory: The directory containing all the .wav files. Should have nothing but .wav
    cutoff_ms: The length of time to trim audio to in milliseconds
  """
  for filename in os.listdir(directory):
    filename = os.path.join(directory, filename)
    with wave.open(filename) as f:
      n_channels = f.getnchannels()
      width = f.getsampwidth()
      rate = f.getframerate()
      data = f.readframes(int((cutoff_ms / 1000) * rate))
    with wave.open(filename, "w") as f:
      f.setnchannels(n_channels)
      f.setsampwidth(width)
      f.setframerate(rate)
      f.writeframes(data)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--record-audio",
    type=bool,
    default=False,
    help="Do audio recording")
  parser.add_argument(
    "--clean-dir",
    type=str,
    default="",
    help="Clean the directory's audio files")
  parser.add_argument(
    "--print-sound-level",
    type=bool,
    default=False,
    help="Print the sound level")
  flags, unparsed = parser.parse_known_args()
  if flags.record_audio:
    record_speech_sequentially()
  if flags.clean_dir:
    clean_dir(flags.clean_dir)
  if flags.print_sound_level:
    print_sound_level()
