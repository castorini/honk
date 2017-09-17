from manage_audio import AudioSnippet
from threading import Lock, Thread
from tkinter import *
from tkinter.ttk import *
import argparse
import base64
import io
import json
import os
import PIL.ImageTk as itk
import pyaudio
import pyttsx3
import re
import requests
import speech_recognition as sr
import time
import wave
import zlib

class GooseWindow(Thread):
    def __init__(self, assets_dir):
        Thread.__init__(self)
        self.assets_dir = assets_dir
        self._switch = False
        self._speaking_lck = Lock()
        self.start()

    def run(self):
        window = Tk()
        window.title("Anserini")
        window.resizable(0, 0)
        img_width, img_height = (213, 365)
        self.canvas = Canvas(width=img_width + 64, height=img_height + 64, bg="white")
        self.canvas.pack()
        self.init_images()
        self.draw_goose("inactive")
        window.mainloop()

    def init_images(self, names=["awake", "inactive", "open"]):
        self.images = {}
        for name in names:
            file = os.path.join(self.assets_dir, "anserini-{}.png".format(name))
            self.images[name] = itk.PhotoImage(file=file)

    def _open_mouth(self, length_secs):
        with self._speaking_lck:
            self.draw_goose("open")
            time.sleep(length_secs)
            self.draw_goose("awake")

    def open_mouth(self, length_secs):
        Thread(target=self._open_mouth, args=(length_secs,)).start()

    def draw_goose(self, name):
        self.canvas.create_image(32, 32, image=self.images[name], anchor=NW, tags="image1" if self._switch else "image2")
        self.canvas.delete("image2" if self._switch else "image1")
        self._switch = not self._switch

def clean_text(text):
    pattern = re.compile(r"^([A-z0-9,\.]+(\-*[A-z0-9,\.]+)*)+$")
    words = []
    for tok in text.split():
        if re.match(pattern, tok):
            words.append(tok)
    return " ".join(words).replace(" .", ".")

class WatsonApi(object):
    def __init__(self, username, password):
        self.auth = requests.auth.HTTPBasicAuth(username, password)

    def fetch_tts(self, text):
        ep = "https://stream.watsonplatform.net/text-to-speech/api/v1/synthesize"
        response = requests.get(ep, params=dict(accept="audio/wav", text=text, voice="en-US_AllisonVoice"), stream=True, auth=self.auth)
        buf = io.BytesIO()
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                buf.write(chunk)
        buf.seek(0)
        return buf

def play_audio(data, amplitude_cb=None):
    with wave.open(data) as f:
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=audio.get_format_from_width(f.getsampwidth()),
            channels=f.getnchannels(),
            rate=f.getframerate(),
            output=True)
        data = f.readframes(512)
        while data:
            if amplitude_cb:
                amplitude = AudioSnippet(data).amplitude_rms()
                amplitude_cb(amplitude)
            stream.write(data)
            data = f.readframes(512)
        stream.stop_stream()
        stream.close()
        audio.terminate()

class Client(object):
    def __init__(self, listen_endpoint, qa_endpoint, goose_window, watson_api=None):
        self.watson_api = watson_api
        self.listen_endpoint = listen_endpoint
        self.qa_endpoint = qa_endpoint
        self.chunk_size = 16000
        self.recognizer = sr.Recognizer()
        self.goose_window = goose_window
        if not watson_api:
            self._tts = pyttsx3.init()
            self._tts.connect("started-word", self._make_tts_cb())

    def say_text(self, text):
        if self.watson_api:
            data = self.watson_api.fetch_tts(text)
            play_audio(data, amplitude_cb=self._make_tts_cb())
        else:
            self._tts.say(text)
            self._tts.runAndWait()

    def _make_tts_cb(self):
        def on_start(name, location, length):
            self.goose_window.open_mouth(length / 40)
        def on_amplitude(amplitude):
            if amplitude > 0.05:
                self.goose_window.draw_goose("open")
            else:
                self.goose_window.draw_goose("awake")
        return on_amplitude if self.watson_api else on_start

    def contains_command(self, data):
        data = base64.b64encode(zlib.compress(data))
        response = requests.post("{}/listen".format(self.listen_endpoint), json=dict(wav_data=data.decode()))
        return json.loads(response.content.decode())["contains_command"]

    def query_qa(self, question):
        response = requests.post("{}/answer".format(self.qa_endpoint), json=dict(question=question, num_hits=1))
        response = json.loads(response.content.decode())
        try:
            return response["answers"][0]["passage"]
        except KeyError:
            return None

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
                self.goose_window.draw_goose("awake")
                print("Detected Anserini! Ask away...")

                question = self._recognize_speech()
                print("You asked, \"{}\"".format(question))
                answer = clean_text(self.query_qa(question))
                print("Answer: {}".format(answer))
                self.say_text(answer)

                buf = [self.stream.read(self.chunk_size), self.stream.read(self.chunk_size)]
                self.goose_window.draw_goose("inactive")
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
    parser.add_argument(
        "--watson-username",
        type=str,
        default="",
        help="If supplied, uses Watson's TTS")
    parser.add_argument(
        "--watson-password",
        type=str,
        default="",
        help="If supplied, uses Watson's TTS")
    flags, _ = parser.parse_known_args()
    file_dir = os.path.dirname(os.path.realpath(__file__))
    goose_window = GooseWindow(file_dir)
    watson_api = None
    if flags.watson_username and flags.watson_password:
        watson_api = WatsonApi(flags.watson_username, flags.watson_password)
    client = Client(flags.listen_endpoint, flags.qa_endpoint, goose_window, watson_api)
    client.start_live_qa()

if __name__ == "__main__":
    main()
    