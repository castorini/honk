import os
import subprocess
import librosa

from pytube import YouTube as PyTube

import utils

FFMPEG_TEMPLATE = "ffmpeg -i {0}.mp4 -codec:a pcm_s16le -ac 1 {0}.wav"

class YoutubeCrawler():
    def __init__(self, url):
        self.url = url
        self.video = PyTube(utils.get_youtube_url(url))

    def get_audio(self):
        temp_file_name = "temp_" + self.url.replace('_', '-')
        self.video.streams.first().download(filename=temp_file_name)

        if not os.path.isfile(temp_file_name + ".mp4"):
            raise Exception("crawled file is not in format of mp4")

        cmd = FFMPEG_TEMPLATE.format(temp_file_name).split()
        subprocess.check_output(cmd)

        audio_data = librosa.core.load(temp_file_name+".wav", 16000)[0]

        os.remove(temp_file_name + ".mp4")
        os.remove(temp_file_name + ".wav")

        return audio_data
