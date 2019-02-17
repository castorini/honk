"""
For each audio entry in the evaluation data,
find where the target target keyword would likely occur by computing measure of similarity
"""
import argparse
import csv
import os
import librosa

from utils import color_print as cp

from extractor import EditDistanceExtractor

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
        help="file containing list of evaluation data to be generated")

    parser.add_argument(
        "-d",
        "--audio_data_directory",
        type=str,
        default="audio_data",
        help="path to audio files")

    parser.add_argument(
        "-e",
        "--extractor",
        type=str,
        default="edit_distance_extractor",
        help="type of extraction algorithm to use")

    parser.add_argument(
        "-th",
        "--threshold",
        type=float,
        default=0.95,
        help="threshold for retrieving a window")

    parser.add_argument(
        "-t",
        "--target",
        type=str,
        help="path to target audio file")

    args = parser.parse_args()
    keyword = args.keyword.lower()

    directory = os.path.join(args.audio_data_directory, keyword)

    if not os.path.exists(directory):
        cp.print_progress("audio data is missing - ", directory)

    total = sum([1 for i in open(args.summary_file, "r").readlines() if i.strip()])

    cp.print_progress("evaluation data file - ", args.summary_file)

    # TODO :: load recorded target keyword audio
    recording = [1, 2, 3, 4] # placeholder

    extractor = None
    if args.extractor == "edit_distance_extractor":
        cp.print_progress("extractor type :", args.extractor, "with threshold :", args.threshold)
        extractor = EditDistanceExtractor(recording, args.threshold)

    with open(args.summary_file, "r") as file:
        reader = csv.reader(file, delimiter=",")

        for i, line in enumerate(reader):
            vid = line[0]
            start_time = line[1]
            end_time = line[2]
            wav_file = os.path.join(directory, vid + "~" + start_time + "~" + end_time + ".wav")

            start_time = int(start_time)
            end_time = int(end_time)

            cp.print_progress(i + 1, " / ", total, " - ", wav_file)

            if not os.path.exists(wav_file):
                cp.print_warning("audio file is missing - ", wav_file)
                continue

            data, _ = librosa.core.load(wav_file, SAMPLE_RATE)

            extracted_audio_times = extractor.extract_keywords(data)

            # TODO :: count how many window has been extracted and compare it against true count

            # TODO :: might be good idea to update threshold if the accuracy is way too low

    cp.print_progress("evaluation is completed for ", keyword, " - ", total)

    # TODO :: calculate accuracy and report metrics

    # TODO :: if we update threshold, report threshold as well

if __name__ == "__main__":
    main()
