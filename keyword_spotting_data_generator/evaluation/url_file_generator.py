import argparse
from pytube import YouTube as PyTube
import color_print as cp
from youtube_searcher import YoutubeSearcher
import utils as utils

API_KEY = "< API_KEY >"

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--keyword",
        type=str,
        required=True,
        help="target keyword to generate data for")

    parser.add_argument(
        "--file_name",
        type=str,
        default="url_file.txt",
        help="name of url file to be generated")

    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="number of url to collect")

    args = parser.parse_args()
    keyword = args.keyword
    cp.print_progress("keyword is ", keyword)

    url_fetcher = YoutubeSearcher(API_KEY, keyword)
    urls = []

    while len(urls) < args.size:
        url = url_fetcher.next()[0]

        if not url:
            cp.print_warning("there are no more urls to process")

        try:
            video = PyTube(utils.get_youtube_url(url))
        except Exception as exception:
            cp.print_error("failed to generate PyTube representation for vidoe ", url)
            continue

        caption = video.captions.get_by_language_code('en')
        if not caption:
            cp.print_warning("no caption available for video - ", url)
            continue

        srt_captions = caption.generate_srt_captions().split('\n\n')

        keyword_exist = False
        for captions in srt_captions:
            if keyword not in captions and keyword + "s" not in captions and keyword + "es" not in captions:
                keyword_exist = True
                break

        if not keyword_exist:
            cp.print_warning("keywords never appear in the video - ", url)
            continue

        urls.append(url)
        cp.print_progress(len(urls), " / ", args.size, " - ", url)

    with open(args.file_name, 'w') as output_file:
        for url in urls:
            output_file.write(url+"\n")

if __name__ == "__main__":
    main()
