import argparse
import inflect
import re
import string
import time
import sounddevice as sd

from pytube import YouTube as PyTube

import color_print as cp
import utils as utils

from evaluation_data_csv_writer import CsvWriter
from url_file_reader import FileReader
from youtube_crawler import YoutubeCrawler
from youtube_searcher import YoutubeSearcher

SAMPLE_RATE = 16000

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-k",
        "--keyword",
        type=str,
        required=True,
        help="target keyword to generate data for")

    parser.add_argument(
        "-f",
        "--url_file",
        type=str,
        help="file containing urls of the video")

    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=100,
        help="number of videos to consider")

    parser.add_argument(
        "-l",
        "--video_length",
        type=int,
        default=3600,
        help="length of maximum length for a video (s)")

    parser.add_argument(
        "-a",
        "--api_key",
        type=str,
        required=True,
        help="API key for youtube data v3 API")

    parser.add_argument(
        "-c",
        "--continue_from",
        type=str,
        help="url to start from in the given url file")

    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="csv file to append output to")

    args = parser.parse_args()
    keyword = args.keyword.lower()
    sd.default.samplerate = SAMPLE_RATE
    cp.print_progress("keyword is ", keyword)

    plural = inflect.engine()

    if args.url_file:
        # read in from the file
        print('fetching urls from the given file : ', args.url_file)
        url_fetcher = FileReader(args.url_file)
    else:
        # fetch using keywords
        print('fetching urls by searching youtube with keywords : ', keyword)
        url_fetcher = YoutubeSearcher(args.api_key, keyword)

    csv_writer = CsvWriter(keyword, args.output_file)

    total_cc_count = 0
    total_audio_count = 0

    continuing = args.continue_from != None

    for i in range(args.size):
        url = url_fetcher.next()[0]

        if continuing:
            if url != args.continue_from:
                continue
            else:
                continuing = False

        if not url:
            cp.print_warning("there are no more urls to process")

        cp.print_progress(i + 1, " / ", args.size, " - ", url)

        try:
            video = PyTube(utils.get_youtube_url(url))
        except Exception as exception:
            cp.print_error("failed to generate PyTube representation for vidoe ", url)
            continue

        if int(video.length) > args.video_length:
            continue

        caption = video.captions.get_by_language_code('en')
        if not caption:
            cp.print_warning("no caption available for video - ", url)
            continue

        try:
            srt_captions = caption.generate_srt_captions().split('\n\n')
        except Exception as exception:
            cp.print_error("failed to retrieve for vidoe - ", url)
            continue

        translator = str.maketrans('', '', string.punctuation) # to remove punctuation
        srt_tag_re = re.compile(r"<.*?>|\(.*?\)|\[.*?\]")

        keyword_exist = False
        for captions in srt_captions:
            if keyword in captions or plural.plural(keyword) in captions:
                keyword_exist = True
                break

        if not keyword_exist:
            cp.print_warning("keywords never appear in the video - ", url)
            continue

        try:
            crawler = YoutubeCrawler(url)
            audio_data = crawler.get_audio()
        except Exception as exception:
            cp.print_warning(exception)
            continue

        collected_data = []
        video_cc_count = 0
        video_audio_count = 0

        for captions in srt_captions:
            cc_split = captions.split('\n')
            if len(cc_split) == 4 and cc_split[0] == '':
                cc_split = (cc_split[1], cc_split[2], cc_split[3])
            elif len(cc_split) != 3:
                cp.print_warning("srt format is not interpretable for video - ", cc_split)
                continue

            _, cc_time, cc_text = cc_split
            cc_text = srt_tag_re.sub('', cc_text)

            # clean up punctuation
            cc_text = cc_text.translate(translator)
            cc_text = cc_text.lower().strip().replace(',', '')
            words = cc_text.strip().split()

            # skip videos without target keyword audio
            if keyword not in words and plural.plural(keyword) not in words:
                continue

            # occurance in audio
            start_ms, end_ms = utils.parse_srt_time(cc_time)
            cp.print_instruction("How many time was the keyword spoken? (\"r\" to replay audio)\n", "[ " + cc_text + " ]")

            while True:
                try:
                    time.sleep(0.5)
                    sd.play(audio_data[start_ms:end_ms], blocking=True)
                    sd.stop()
                    user_input = input()
                    audio_count = int(user_input)
                except ValueError:
                    if user_input != "r":
                        cp.print_error("Invalid Input. Expect Integer")
                    continue
                else:
                    break

            # occurance in captions
            cc_count = 0
            for word in words:
                if keyword == word or keyword + "s" == word or keyword + "es" == word:
                    cc_count += 1

            collected_data.append([url, start_ms, end_ms, cc_text, cc_count, audio_count])

            video_cc_count += cc_count
            video_audio_count += audio_count

        print(url, "- cc_count : ", video_cc_count, ", audio_count : ", video_audio_count)

        total_cc_count += video_cc_count
        total_audio_count += video_audio_count

        csv_writer.write(collected_data)

    print("total cc_count : ", total_cc_count, ", total audio_count : ", total_audio_count)
    cp.print_progress("collected data sotred in ", keyword + ".csv")

if __name__ == "__main__":
    main()
