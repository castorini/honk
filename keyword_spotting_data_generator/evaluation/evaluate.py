"""
For each audio entry in the evaluation data,
find where the target target keyword would likely occur by computing measure of similarity
"""
import argparse
import csv
import os
import librosa

import numpy as np
import color_print as cp

import similarity_metric.cosine_similarity as cs

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
        "--evaluation_data_file",
        type=str,
        help="file containing list of evaluation data to be generated")

    parser.add_argument(
        "-a",
        "--api_key",
        type=str,
        required=True,
        help="API key for youtube data v3 API")

    parser.add_argument(
        "-d",
        "--audio_data_directory",
        type=str,
        default="audio_data",
        help="audio data folder")

    parser.add_argument(
        "-s",
        "--similarity_metric",
        type=str,
        default="cosine",
        help="similarity metric to use")

    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.95,
        help="threshold for simliarity between audio")

    parser.add_argument(
        "-r",
        "--recording",
        type=str,
        help="recording of target keyword")

    args = parser.parse_args()
    keyword = args.keyword.lower()

    directory = os.path.join(args.audio_data_directory, keyword)

    if not os.path.exists(directory):
        cp.print_progress("audio data is missing - ", directory)

    total = sum([1 for i in open(args.evaluation_data_file, "r").readlines() if i.strip()])

    cp.print_progress("evaluation data file - ", args.evaluation_data_file)

    # TODO :: load recorded target keyword audio
    recording = [1, 2, 3, 4] # placeholder

    # TODO :: instantiate similarity metric object

    similarity = None
    if args.similarity_metric == "cosine":
        similarity = cs.CosineSimilarity(recording)

    with open(args.evaluation_data_file, "r") as file:
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
                cp.print_progress("audio file is missing - ", wav_file)
                break

            data, _ = librosa.core.load(wav_file, SAMPLE_RATE)

            # TODO :: for each window, call similarity metrics using window and recording
            # similarity.compute_similarity(data)

            # TODO :: if similarity is greater than threshold, remember window

            # TODO :: generate a wav file and updaate counts

            # TODO :: might be good idea to update threshold if the accuracy is way too low

    cp.print_progress("evaluation is completed for ", keyword, " - ", total)

    # TODO :: calculate accuracy and report metrics

    # TODO :: if we update threshold, report threshold as well

if __name__ == "__main__":
    main()
