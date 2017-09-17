import argparse
import base64
import json
import os
import pyaudio
import requests
import zlib

class Client(object):
    def __init__(self, endpoint):
        self.endpoint = endpoint

    def contains_command(self, data):
        data = base64.b64encode(zlib.compress(data))
        response = requests.post("{}/listen".format(self.endpoint), json=dict(wav_data=data.decode())).content.decode()
        return json.loads(response)["contains_command"]

    def begin_live_labelling(self):
        p = pyaudio.PyAudio()
        chunk_size = 16000
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=chunk_size)
        buf = [stream.read(chunk_size), stream.read(chunk_size)]
        while True:
            print(self.contains_command(b''.join(buf)))
            buf[0] = buf[1]
            buf[1] = stream.read(chunk_size)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://127.0.0.1:16888",
        help="The endpoint to use")
    flags, _ = parser.parse_known_args()
    client = Client(flags.endpoint)
    client.begin_live_labelling()

if __name__ == "__main__":
    main()