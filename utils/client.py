from manage_audio import AudioSnippet
import argparse
import base64
import json
import os
import pyaudio
import requests
import speech_recognition as sr
import zlib

class Client(object):
    def __init__(self, listen_endpoint, qa_endpoint):
        self.listen_endpoint = listen_endpoint
        self.qa_endpoint = qa_endpoint
        self.chunk_size = 16000
        self.recognizer = sr.Recognizer()

    def contains_command(self, data):
        data = base64.b64encode(zlib.compress(data))
        response = requests.post("{}/listen".format(self.listen_endpoint), json=dict(wav_data=data.decode()))
        return json.loads(response.content.decode())["contains_command"]

    def query_qa(self, question):
        response = requests.post("{}/answer".format(self.qa_endpoint), json=dict(question=question, num_hits=1))
        print(response.content.decode())

    def _recognize_speech(self):
        self._stop_listening()
        try:
            with sr.Microphone() as source:
                audio = self.recognizer.listen(source)
            return self.recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return None
        finally:
            self._start_listening()

    def _start_listening(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=self.chunk_size)

    def _stop_listening(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def start_live_qa(self):
        self._start_listening()
        print("Speak Anserini when ready!")
        buf = [self.stream.read(self.chunk_size), self.stream.read(self.chunk_size)]
        while True:
            if self.contains_command(b''.join(buf)):
                print("Detected Anserini! Ask away...")
                question = self._recognize_speech()
                print("You asked, \"{}\"".format(question))
                print("Answer: {}".format(self.query_qa(question)))
                buf = [self.stream.read(self.chunk_size), self.stream.read(self.chunk_size)]
                continue
            buf[0] = buf[1]
            buf[1] = self.stream.read(self.chunk_size)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--listen-endpoint",
        type=str,
        default="http://127.0.0.1:16888",
        help="The endpoint to use")
    parser.add_argument(
        "--qa-endpoint",
        type=str,
        default="http://dragon00.cs.uwaterloo.ca:80")
    flags, _ = parser.parse_known_args()
    client = Client(flags.listen_endpoint, flags.qa_endpoint)
    client.start_live_qa()

if __name__ == "__main__":
    main()