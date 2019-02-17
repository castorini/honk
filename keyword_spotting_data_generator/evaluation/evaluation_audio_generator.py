"""
For each audio entry in the evaluation data, generate .wav file
store data under data/<keyword>, with following naming convention:
<video id>~<start time (ms)>~<end time (ms)>.wav
"""
import argparse
import csv
import os
import librosa

from utils import color_print as cp
from utils import YoutubeCrawler

SAMPLE_RATE = 16000

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-k",
        "--keyword",
        type=str,
        required=True,
        help="keyword for the given evaluation data list")

    parser.add_argument(
        "-f",
        "--summary_file",
        type=str,
        help="file containing summary of audio blocks")

    parser.add_argument(
        "-a",
        "--api_key",
        type=str,
        required=True,
        help="API key for youtube data v3 API")

    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default="audio_data",
        help="folder to store the audio data")

    args = parser.parse_args()
    keyword = args.keyword.lower()

    directory = os.path.join(args.output_folder, keyword)

    cp.print_progress("location of audio data - ", directory)

    if not os.path.exists(directory):
        os.makedirs(directory)

    total = sum([1 for i in open(args.summary_file, "r").readlines() if i.strip()])

    cp.print_progress("evaluation data file - ", args.summary_file)

    with open(args.summary_file, "r") as file:
        reader = csv.reader(file, delimiter=",")

        prev_vid = None
        for i, line in enumerate(reader):
            curr_vid = line[0]
            start_time = line[1]
            end_time = line[2]
            wav_file = os.path.join(directory, curr_vid + "~" + start_time + "~" + end_time + ".wav")

            start_time = int(start_time)
            end_time = int(end_time)

            cp.print_progress(i + 1, " / ", total, " - ", wav_file)

            if os.path.exists(wav_file):
                cp.print_warning(wav_file, "already exist")
                continue

            if prev_vid != curr_vid:
                try:
                    crawler = YoutubeCrawler(curr_vid)
                    audio_data = crawler.get_audio()
                except Exception as exception:
                    cp.print_error("failed to download audio file for video ", curr_vid)
                    cp.print_warning(exception)
                    continue

            librosa.output.write_wav(wav_file, audio_data[start_time:end_time], SAMPLE_RATE)

            prev_vid = curr_vid

    cp.print_progress("audio file generation is completed for ", keyword, " - ", total)

if __name__ == "__main__":
    main()
