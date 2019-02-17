import argparse
from pytube import YouTube as PyTube

import inflect
from utils import color_print as cp
from utils import util
from url_fetcher import YoutubeSearcher

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-k",
        "--keyword",
        type=str,
        required=True,
        help="target keyword to generate data for")

    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=100,
        help="number of url to collect")

    parser.add_argument(
        "-a",
        "--api_key",
        type=str,
        required=True,
        help="API key for youtube data v3 API")

    args = parser.parse_args()
    keyword = args.keyword.lower()
    cp.print_progress("keyword is ", keyword)

    url_fetcher = YoutubeSearcher(args.api_key, keyword)
    urls = []

    plural = inflect.engine()

    while len(urls) < args.size:
        url = url_fetcher.next()[0]

        if not url:
            cp.print_warning("there are no more urls to process")

        if url in urls:
            cp.print_warning("video is already added", url)
            continue

        try:
            video = PyTube(util.get_youtube_url(url))
        except Exception as exception:
            cp.print_error("failed to generate PyTube representation for vidoe - ", url)
            continue

        caption = video.captions.get_by_language_code('en')
        if not caption:
            cp.print_warning("no caption available for video - ", url)
            continue

        try:
            srt_captions = caption.generate_srt_captions().lower().split('\n\n')
        except Exception as exception:
            cp.print_error("failed to retrieve for vidoe - ", url)
            continue

        keyword_exist = False
        for captions in srt_captions:
            if keyword in captions or plural.plural(keyword) in captions:
                keyword_exist = True
                break

        if not keyword_exist:
            cp.print_warning("keywords never appear in the video - ", url)
            continue

        urls.append(url)
        cp.print_progress(len(urls), " / ", args.size, " - ", url)

    cp.print_warning(len(urls), "urls are collected for ", keyword)

    with open(keyword + "_url_" + args.size +".txt", 'w') as output_file:
        for url in urls:
            output_file.write(url+"\n")

if __name__ == "__main__":
    main()
