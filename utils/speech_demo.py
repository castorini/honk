import argparse
import base64
import json
import math
import threading
import time
import zlib

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image
import librosa
import pyaudio
import numpy as np
import requests

textures = {}
labels = ["unknown", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
def load_texture(filename, width, height):
    im = Image.open(filename)
    im.convert("RGBA")
    data = im.getdata()
    pixels = []
    for pixel in reversed(data):
        pixels.extend(pixel)
    pixels = np.array(pixels, np.uint8)
    tex_id = glGenTextures(1)

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    return tex_id

def draw_spectrogram(audio_data):
    audio_data = np.frombuffer(audio_data, np.int16).astype(np.float32) / 32768
    spectrogram = np.absolute(np.fft.fft(audio_data)[:len(audio_data) // 2])
    spectrogram = np.power(np.clip(spectrogram, 0, 200) / 200., 0.7)
    glColor3f(0.3, 0.3, 0.3)
    h = 7
    s = 4
    for i, energy in enumerate(spectrogram.tolist()):
        glBegin(GL_QUADS)
        glVertex2f(0, i * (h + s))
        glVertex2f(int(energy * 150), i * (h + s))
        glVertex2f(int(energy * 150), i * (h + s) + h)
        glVertex2f(0, i * (h + s) + h)
        glEnd()
        glBegin(GL_QUADS)
        glVertex2f(800, i * (h + s))
        glVertex2f(800 - int(energy * 150), i * (h + s))
        glVertex2f(800 - int(energy * 150), i * (h + s) + h)
        glVertex2f(800, i * (h + s) + h)
        glEnd()

def draw_text(text, x, y):
    m = 0.0385
    glColor3f(0.9, 0.9, 0.9)
    for i, c in enumerate(text):
        idx = ord(c) - ord("a")
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBindTexture(GL_TEXTURE_2D, textures["font"])
        glBegin(GL_QUADS)
        glTexCoord2f(m * idx, 0)
        glVertex2f(x, y)
        glTexCoord2f(m * idx, 1)
        glVertex2f(x, y + 15)
        glTexCoord2f(m * (idx + 1), 1)
        glVertex2f(x + 10, y + 15)
        glTexCoord2f(m * (idx + 1), 0)
        glVertex2f(x + 10, y)
        x += 10
        glEnd()
        glDisable(GL_TEXTURE_2D)

def create_rot_matrix(rads):
    return np.array([[math.cos(rads), -math.sin(rads)],
        [math.sin(rads), math.cos(rads)]])

def draw_vertices(vertices):
    for vertex in vertices:
        glVertex2f(*vertex)

class LerpStepper(object):
    def __init__(self, a, b, speed):
        self.last_time = time.time()
        self.val = a
        self.b = b
        self.speed = speed
        self.running = False

    def reset(self, val, b=None):
        self.val = val
        if b:
            self.b = b

    def step(self):
        if self.val < self.b:
            self.val += self.speed

class Indicator(object):
    indicators = []
    def __init__(self, text, midpoint, index, radius=250, n_slices=len(labels)):
        self.text = text
        self._state = 0.
        self.midpoint = midpoint
        self.radius = radius
        self.index = index
        self.color = (0.4, 0.4, 0.4)
        self._color_lerp = LerpStepper(1.0, 1.0, 0.01)
        Indicator.indicators.append(self)

        slice_rads = (2 * math.pi) / n_slices
        rads = index * slice_rads
        self._rotmat1 = create_rot_matrix(rads)
        rads = (index + 0.5) * slice_rads
        self._rotmat15 = create_rot_matrix(rads)
        rads = (index + 1) * slice_rads
        self._rotmat2 = create_rot_matrix(rads)
        self._init_shape()

    def highlight(self, intensity):
        self._color_lerp.reset(1 - intensity)

    def _init_shape(self):
        bp = np.array([0, self.radius])
        bp2 = np.array([0, self.radius * 0.8])
        m = np.array(self.midpoint)
        p1 = np.matmul(self._rotmat1, bp) + m
        p2 = np.matmul(self._rotmat2, bp) + m
        self.text_pos = np.matmul(self._rotmat15, bp2) + m
        self.text_pos[0] -= len(self.text) * 5
        p1 = p1.tolist()
        p2 = p2.tolist()
        self.vertices = [self.midpoint, p1, p2, self.midpoint]

    def tick(self):
        self._color_lerp.step()

    def draw(self):
        draw_text(self.text, self.text_pos[0], self.text_pos[1])
        glEnable(GL_LINE_SMOOTH)
        glLineWidth(2)
        glBegin(GL_LINE_STRIP)
        color = [0.4 + min(1 - self._color_lerp.val, 0.6)] * 3
        glColor3f(*color)
        draw_vertices(self.vertices)
        glEnd()
        glDisable(GL_LINE_SMOOTH)

    def highlight(self, intensity):
        self._color_lerp.reset(1 - intensity)

class LabelClient(object):
    def __init__(self, server_endpoint):
        self.endpoint = server_endpoint
        self.chunk_size = 1000
        self._audio = pyaudio.PyAudio()
        self._audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, 
            frames_per_buffer=self.chunk_size, stream_callback=self._on_audio)
        self.last_data = np.zeros(1000)
        self._audio_buf = []

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
            try:
                Indicator.indicators[labels.index(key)].highlight(1.)
            except ValueError:
                continue
        return data_ok

class DemoApplication(object):
    def __init__(self, label_client):
        glutInit()
        self.width, self.height = 800, 600
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)
        glutInitWindowPosition(100, 100)
        self.window = glutCreateWindow(b"Google Speech Dataset Demo")
        self.label_client = label_client

        glutDisplayFunc(self.draw)
        glutIdleFunc(self.draw)
        glutReshapeFunc(self._on_resize)
        glClearColor(0.12, 0.12, 0.15, 1.)
        font_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fonts.png")
        textures["font"] = load_texture(font_path, 208, 15)
        self.children = [Indicator(labels[i], [400, 300], i) for i in range(len(labels))]

    def _refresh(self):
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0.0, self.width, 0.0, self.height, 0.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def _on_resize(self, width, height):
        glutReshapeWindow(self.width, self.height);

    def draw(self):
        self.children = sorted(self.children, key=lambda x: -x._color_lerp.val)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        self._refresh()
        self._draw()
        for obj in self.children:
            obj.draw()
        glutSwapBuffers()

    def _tick(self):
        pass

    def tick(self):
        self._tick()
        for obj in self.children:
            obj.tick()

    def _do_tick(self, hz=60):
        delay = 1 / hz
        while True:
            a = time.time()
            self.tick()
            dt = time.time() - a
            time.sleep(max(0, delay - dt))

    def _draw(self):
        draw_spectrogram(self.label_client.last_data)

    def run(self):
        threading.Thread(target=self._do_tick).start()
        glutMainLoop()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-endpoint",
        type=str,
        default="http://127.0.0.1:16888",
        help="The endpoint to use")
    flags = parser.parse_args()
    app = DemoApplication(LabelClient(flags.server_endpoint))
    app.run()

if __name__ == "__main__":
    main()
