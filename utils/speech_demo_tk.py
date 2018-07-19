
#!/usr/bin/env python3
import argparse
import base64
import json
import math
import threading
import time
import zlib
import pyaudio
import numpy as np
import requests

import tkinter as tk
from tkinter import BOTH, LEFT, RIGHT, TOP, BOTTOM, RAISED, X, N, END
from tkinter import Text
from tkinter.ttk import Frame, LabelFrame, Button, Style, Label, Entry

labels = ["silence", "unknown", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

class SpeechDemo(Frame):
    def __init__(self, label_client):
        super().__init__()
        self.label_client = label_client
        self.init_ui()

    def init_ui(self):
        """ setup the GUI for the app """
        self.master.title("Speech Demo")
        self.pack(fill=BOTH, expand=True)

        label_frame = LabelFrame(self, text="Words you can speak")
        label_frame.pack(fill=X, padx=10, pady=10)
        self.labels = []
        rows = 4
        cols = 3
        for j in range(rows):
            for i in range(cols):
                k = i  + j * cols
                label = Label(label_frame, text=labels[k])
                label.config(font=("Courier", 36))
                label.grid(row=j,column=i,padx=10, pady=10)
                self.labels += [ label ]
        self.selected = None
        self.after(100, self.ontick)

    def on_play_button_click(self):
        pass

    def ontick(self):
        # check for new labels and display them
        words = self.label_client.get_words()
        if len(words) > 0:
            key = words[-1]
            i = labels.index(key)
            label = self.labels[i]
            if label["text"] != key:
                print("That's weird label {} has text {}".format(i, label["text"]))
            if label != self.selected and self.selected is not None:
                self.selected.configure(background  = '')
            label.configure(background  = "green")
            self.selected = label
        self.after(100, self.ontick)

    
class LabelClient(object):
    def __init__(self, server_endpoint):
        self.endpoint = server_endpoint
        self.chunk_size = 1000
        self._audio = pyaudio.PyAudio()
        self._audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, 
            frames_per_buffer=self.chunk_size, stream_callback=self._on_audio)
        self.last_data = np.zeros(1000)
        self._audio_buf = []
        self.words = []

    def _on_audio(self, in_data, frame_count, time_info, status):
        data_ok = (in_data, pyaudio.paContinue)
        self.last_data = in_data
        self._audio_buf.append(in_data)
        if len(self._audio_buf) != 16:
            return data_ok
        audio_data = base64.b64encode(zlib.compress(b"".join(self._audio_buf)))
        self._audio_buf = []
        response = requests.post("{}/listen".format(self.endpoint), json=dict(wav_data=audio_data.decode(), method="all_label"))
        response = json.loads(response.content.decode())
        if not response:
            return data_ok
        max_key = max(response.items(), key=lambda x: x[1])[0]
        for key in response:
            p = response[key]
            if p < 0.5 and key != "__unknown__":
                continue
            key = key.replace("_", "")
            print(key)
            self.words += [ key ]
        return data_ok

    def get_words(self):
        temp = self.words
        self.words = []
        return temp


def main(server_endpoint):
    """ Main function to create root UI and SpeechDemo object, then run the main UI loop """
    root = tk.Tk()
    root.geometry("800x600")
    app = SpeechDemo(LabelClient(server_endpoint))
    while True:
        try:
            root.mainloop()
            break
        except UnicodeDecodeError:
            pass
    
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-endpoint",
        type=str,
        default="http://127.0.0.1:16888",
        help="The endpoint to use")
    flags = parser.parse_args()
    main(flags.server_endpoint)