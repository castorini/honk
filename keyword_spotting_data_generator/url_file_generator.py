import argparse
import color_print as cp
from youtube_searcher import YoutubeSearcher

API_KEY = "AIzaSyDyZMEDTMIb_RmdPjN8wpkXXuBCnHGFBXA"

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
    cp.print_progress("keyword is ", args.keyword)

    url_fetcher = YoutubeSearcher(API_KEY, args.keyword)
    with open(args.file_name, 'w') as output_file:
        for _ in range(args.size):
            output_file.write(url_fetcher.next()[0]+"\n")

if __name__ == "__main__":
    main()
