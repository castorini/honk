import argparse
import color_print as cp
import re
import string
import utils
from pytube import YouTube as PyTube
from youtube_searcher import YoutubeSearcher
from url_file_reader import FileReader
from youtube_crawler import YoutubeCrawler
from evaluation_data_csv_writer import CsvWriter

API_KEY = "AIzaSyDyZMEDTMIb_RmdPjN8wpkXXuBCnHGFBXA"

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--keyword",
        type=str,
        required=True,
        help="target keyword to generate data for")

    parser.add_argument(
        "--url_file",
        type=str,
        help="file containing urls of the video")

    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="number of videos to consider")

    parser.add_argument(
        "--video_length",
        type=int,
        default=3600,
        help="length of maximum length for a video (s)")

    args = parser.parse_args()
    keyword = args.keyword
    cp.print_progress("keyword is ", keyword)

    if args.url_file:
        # read in from the file
        print('fetching urls from the given file : ', args.url_file)
        url_fetcher = FileReader(args.url_file)
    else:
        # fetch using keywords
        print('fetching urls by searching youtube with keywords : ', keyword)
        url_fetcher = YoutubeSearcher(API_KEY, keyword)

    csv_writer = CsvWriter(keyword)

    for i in range(args.size):
        url = url_fetcher.next()[0]

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

        srt_captions = caption.generate_srt_captions().split('\n\n')

        translator = str.maketrans('', '', string.punctuation) # to remove punctuation
        srt_tag_re = re.compile(r"<.*?>|\(.*?\)|\[.*?\]")
        srt_time_re = re.compile(r"(\d+):(\d+):(\d+),(\d+)\s-->\s(\d+):(\d+):(\d+),(\d+)")

        aligner_config = "task_language=eng|is_text_type=plain|os_task_file_format=json"

        keyword_exist = False
        for captions in srt_captions:
            if keyword not in captions and keyword + "s" not in captions and keyword + "es" not in captions:
                keyword_exist = True
                break

        if not keyword_exist:
            cp.print_warning("keywords never appear in the video - ", url)
            continue

        audio_data = YoutubeCrawler(url)
        collected_data = []

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
            cc_text = cc_text.lower()
            words = cc_text.strip().split()

            # filter srt that contains target keyword
            if keyword not in words and keyword + "s" not in words and keyword + "es" not in words:
                continue

            srt_count = 0
            for word in words:
                if keyword == word or keyword + "s" == word or keyword + "es" == word:
                    srt_count += 1

            # TODO::play audio

            audio_count = int(input("How many time was target keyword appeared in the audio?\n"))

            start_ms, end_ms = utils.parse_srt_time(cc_time)
            collected_data.append([url, start_ms, end_ms, srt_count, audio_count])

        csv_writer.write(collected_data)

    cp.print_progress("collected data sotred in ", keyword + ".csv")

if __name__ == "__main__":
    main()
