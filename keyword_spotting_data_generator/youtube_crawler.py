import os
import librosa
import subprocess
import utils
import color_print as cp
from pytube import YouTube as PyTube

FFMPEG_TEMPLATE = "ffmpeg -i {0}.mp4 -codec:a pcm_s16le -ac 1 {0}.wav"

class YoutubeCrawler():
    def __init__(self, url):
        self.url = url
        self.video = PyTube(utils.get_youtube_url(url))

    def get_audio(self):
        file_name = self.url.replace('_', '-')
        self.video.streams.first().download(filename=file_name)

        temp_file_name = "temp_" + file_name
        if not os.path.isfile(temp_file_name + ".mp4"):
            cp.print_warning("crawled file is not in format of mp4")
            return

        cmd = FFMPEG_TEMPLATE.format(temp_file_name).split()
        subprocess.check_output(cmd)

        os.remove(temp_file_name + ".mp4")
        os.remove(temp_file_name + ".wav")

        return librosa.core.load(temp_file_name+".wav", 16000)[0]