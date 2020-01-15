import librosa
import os
import re
import subprocess
from pydub import AudioSegment
from pytube import YouTube as PyTube

URL_TEMPLATE = "http://youtube.com/watch?v={}"

def get_youtube_url(vid):
    return URL_TEMPLATE.format(vid)

SRT_TIME_PARSER = re.compile(r"(\d+):(\d+):(\d+),(\d+)\s-->\s(\d+):(\d+):(\d+),(\d+)")

def srt_time_to_ms(hour, minute, second, msecond):
    converted = int(msecond)
    converted += (1000 * int(second))
    converted += (1000 * 60 * int(minute))
    converted += (1000 * 60 * 60 * int(hour))
    return converted

def parse_srt_time(time):
    match_result = SRT_TIME_PARSER.match(time)

    start_pos = None
    stop_pos = None

    if match_result:
        start_time_ms = srt_time_to_ms(
            match_result.group(1),
            match_result.group(2),
            match_result.group(3),
            match_result.group(4))
        stop_time_ms = srt_time_to_ms(
            match_result.group(5),
            match_result.group(6),
            match_result.group(7),
            match_result.group(8))

        # * 16 because the audio has sample rate of 16000
        start_pos = start_time_ms * 16
        stop_pos = stop_time_ms * 16

    else:
        raise Exception("failed to parse srt time - " + time)

    return start_pos, stop_pos


class AudioSegmentor():
    def __init__(self, keyword, output_dir, audio_length=1000):
        self.keyword = keyword
        self.audio_length = audio_length
        self.output_dir = os.path.join(output_dir, keyword)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def segment_audio(self, file_name, segments):
        audio_data = AudioSegment.from_wav(file_name)

        for segment in segments:
            assert segment[0] < segment[1]
            center = round((segment[0] + segment[1]) / 2)

            padding = round(self.audio_length / 2)
            if center < padding:
                start_time = 0
            else:
                start_time = center - padding

            end_time = start_time + self.audio_length

            audio_segment = audio_data[start_time:end_time]

            file_prefix = os.path.basename(file_name).split('.')[0]

            file_name = os.path.join(self.output_dir, file_prefix + "_" + str(start_time) + "~" + str(end_time) + ".wav")
            print(file_name)

            audio_segment.export(file_name, format="wav")


class YoutubeCrawler():
    def __init__(self, url):
        self.url = url
        self.video = PyTube(get_youtube_url(url))

    def get_audio(self):
        temp_file_name = self.url.replace('_', '-')
        self.video.streams.first().download(filename=temp_file_name)

        if not os.path.isfile(temp_file_name + ".mp4"):
            raise Exception("crawled file is not in format of mp4")

        ffmpeg_template = "ffmpeg -i {0}.mp4 -codec:a pcm_s16le -ac 1 {0}.wav"
        cmd = ffmpeg_template.format(temp_file_name).split()
        subprocess.check_output(cmd)

        audio_data = librosa.core.load(temp_file_name+".wav", 16000)[0]

        os.remove(temp_file_name + ".mp4")
        os.remove(temp_file_name + ".wav")

        return audio_data
