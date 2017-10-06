from threading import Lock, Thread
from tkinter import *
from tkinter.ttk import *
import argparse
import base64
import io
import json
import os
import re
import requests
import time
import wave
import zlib

import PIL.ImageTk as itk
import pyaudio
import pyttsx3
import speech_recognition as sr

from manage_audio import AudioSnippet

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
        self._init_resources()
        self.draw_goose("inactive")
        window.mainloop()

    def _init_resources(self, names=["awake", "inactive", "open"]):
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
    if not text:
        return ""
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
    def __init__(self, server_endpoint, qa_endpoint, goose_window, watson_api=None):
        self.watson_api = watson_api
        self.server_endpoint = server_endpoint
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
        response = requests.post("{}/listen".format(self.server_endpoint), json=dict(wav_data=data.decode(), method="command_tagging"))
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

    def send_retarget_data(self, data, positive=True):
        data = base64.b64encode(zlib.compress(data))
        requests.post("{}/data".format(self.server_endpoint), json=dict(wav_data=data.decode(), positive=positive))

    def _retarget_negative(self, n_minutes=1):
        if n_minutes == 1:
            self.say_text("Please speak random words for the next minute.")
        else:
            self.say_text("Please speak random words for the next {} minutes.".format(n_minutes))
        t0 = 0
        snippet = AudioSnippet()
        while t0 < n_minutes * 60:
            snippet.append(AudioSnippet(self.stream.read(self.chunk_size)))
            t0 += self.chunk_size / 16000
        for chunk in snippet.chunk(32000, 16000):
            self.send_retarget_data(chunk.byte_data, positive=False)

    def _retarget_positive(self, n_times=10):
        self.say_text("Please speak the new command {} times.".format(n_times))
        self.goose_window.draw_goose("inactive")
        n_said = 0
        while n_said < n_times:
            self.goose_window.draw_goose("inactive")
            snippet = AudioSnippet(self.stream.read(self.chunk_size))
            tot_snippet = AudioSnippet()

            while snippet.amplitude_rms() > 0.01:
                if not tot_snippet.byte_data:
                    self.goose_window.draw_goose("awake")
                    if n_said == n_times // 2 and n_said >= 5:
                        self.say_text("Only {} times left.".format(n_times - n_said))
                    elif n_said == n_times - 5:
                        self.say_text("Only 5 more times.")
                    n_said += 1
                tot_snippet.append(snippet)
                tot_snippet.append(AudioSnippet(self.stream.read(self.chunk_size)))
                snippet = AudioSnippet(self.stream.read(self.chunk_size))
            if tot_snippet.byte_data:
                tot_snippet.trim_window(16000 * 2)
                self.send_retarget_data(tot_snippet.byte_data)

    def _do_retarget(self):
        requests.post("{}/train".format(self.server_endpoint))
        self.say_text("Started training your custom keyword")
        while True:
            time.sleep(5)
            response = requests.get("{}/train".format(self.server_endpoint)).content
            if not json.loads(response.decode())["in_progress"]:
                self.say_text("Completed keyword retargeting!")
                break

    def start_retarget(self):
        print("Follow the goose!")
        self._start_listening()
        requests.delete("{}/data".format(self.server_endpoint))
        self._retarget_positive()
        #self._retarget_negative()
        self._do_retarget()

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
                if answer:
                    print("Answer: {}".format(answer))
                    self.say_text(answer)
                else:
                    print("No answer available!")

                buf = [self.stream.read(self.chunk_size), self.stream.read(self.chunk_size)]
                self.goose_window.draw_goose("inactive")
                continue
            buf[0] = buf[1]
            buf[1] = self.stream.read(self.chunk_size)

def start_client(flags):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    goose_window = GooseWindow(file_dir)
    watson_api = None
    if flags.watson_username and flags.watson_password:
        watson_api = WatsonApi(flags.watson_username, flags.watson_password)
    client = Client(flags.server_endpoint, flags.qa_endpoint, goose_window, watson_api)
    if flags.mode == "query":
        client.start_live_qa()
    elif flags.mode == "retarget":
        client.start_retarget()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-endpoint",
        type=str,
        default="http://127.0.0.1:16888",
        help="The endpoint to use")
    parser.add_argument(
        "--mode",
        type=str,
        default="query",
        choices=["retarget", "query"],
        help="The mode to run the client in.")
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
    start_client(flags)

if __name__ == "__main__":
    main()
