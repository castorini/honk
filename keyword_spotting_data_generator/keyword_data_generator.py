import inflect
import librosa
import os
import re
import subprocess
import string
import wordset
import youtube_processor as yp
from argparse import ArgumentParser
from extractor import SphinxSTTExtractor
from pytube import YouTube as PyTube
from utils import color_print as cp
from utils import file_utils
from youtube_searcher import YoutubeSearcher


TEMP_DIR = "/tmp/keyword_generator"

def count_keyword(keyword, words):
    keyword_count = 0

    for word in words:
        if word == keyword:
            keyword_count += 1
            continue

    return keyword_count

def contain_keyword(keyword, caption):
    keyword_exist = False

    if keyword in caption:
        keyword_exist = True

    return keyword_exist


def process_cation(keyword, caption, punctutation_translator, srt_tag_re):
    '''
    process caption to get caption time and text
    if the target keyword is missing or srt format is incorrect, return None
    '''

    cc_split = caption.split('\n')
    if len(cc_split) == 4 and cc_split[0] == '':
        cc_split = (cc_split[1], cc_split[2], cc_split[3])
    elif len(cc_split) != 3:
        # cp.print_color(cp.ColorEnum.YELLOW, "srt format is not interpretable for the video")
        return None, None, None

    _, cc_time, cc_text = cc_split
    cc_text = srt_tag_re.sub('', cc_text)
    cc_text = cc_text.encode('ascii', errors='ignore').decode()

    # clean up punctuation
    cc_text = cc_text.translate(punctutation_translator)
    cc_text = cc_text.lower().strip().replace(',', '')
    words = cc_text.strip().split()

    # check if the caption contain the keyword
    keyword_exist = contain_keyword(keyword, words)

    if not keyword_exist:
        # cp.print_color(cp.ColorEnum.YELLOW, "srt format is not interpretable for the video")
        return None, None, None

    try:
        start_time, end_time = yp.parse_srt_time(cc_time)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exception:
        # cp.print_color(cp.ColorEnum.YELLOW, exception)
        return None, None, None

    return start_time, end_time, words


def retrieve_captions(url, keyword):
    '''
    return captions if the video is in right format and contains target keyword
    else, return None
    '''

    try:
        video = PyTube(yp.get_youtube_url(url))
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        cp.print_color(cp.ColorEnum.YELLOW, "failed to generate PyTube representation for the video")
        return None

    caption = video.captions.get_by_language_code('en')
    if not caption:
        cp.print_color(cp.ColorEnum.YELLOW, "no caption available for the video")
        return None

    try:
        srt_captions = caption.generate_srt_captions().lower().split('\n\n')
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        cp.print_color(cp.ColorEnum.YELLOW, "failed to retrieve for the video")
        return None

    # make sure the keyword appear in captions before crawling
    keyword_exist = False
    for caption in srt_captions:
        keyword_exist = contain_keyword(keyword, caption)

        if keyword_exist:
            return srt_captions

    cp.print_color(cp.ColorEnum.YELLOW, "captions do not contain the keyword")
    return None


def extract_keyword(url, extractor, audio_segmentor, audio_data, start_time, end_time, cc_count):
    '''
    process audio and extract one second long audio using Sphinx keyword spotting
    '''
    temp_audio_file = os.path.join(TEMP_DIR, "temp.wav")
    audio_file = os.path.join(TEMP_DIR, f"{url.replace('_', '-')}_{start_time}.wav")

    # generate an audip file for the current block
    librosa.output.write_wav(temp_audio_file, audio_data[start_time:end_time], 16000)

    # librosa stores floating-point data but we need signed-integer for speech-to-text
    # https://github.com/wblgers/py_speech_seg/issues/2
    sox_command_template = "sox {0} -b 16 -e signed-integer {1}"
    cmd = sox_command_template.format(temp_audio_file, audio_file).split()
    subprocess.check_output(cmd)

    os.remove(temp_audio_file)
    extracted_audio_times = extractor.extract_keywords(audio_file, 16000)

    extracted_audio_count = 0
    # to increase the quality of the audio generated, only extract if counts from caption is equal to the kws count
    if len(extracted_audio_times) == cc_count:
        audio_segmentor.segment_audio(audio_file, extracted_audio_times)
        extracted_audio_count = len(extracted_audio_times)

    os.remove(audio_file)

    return extracted_audio_count


def generate_dataset(youtube_api_key, words_api_key, keyword, data_size, output_dir):
    '''
    search keyword on youtube and extract keyword audio
    '''
    plural_engine = inflect.engine()

    # valid form of keyword
    keyword = keyword.lower()

    # list of keywords to search youtube about
    synonyms = wordset.get_relevant_words(keyword, words_api_key)
    synonyms = [keyword] + synonyms

    search_terms = []
    for term in synonyms:
        if term not in search_terms:
            search_terms.append(term)

        plural = plural_engine.plural(term)
        if plural not in search_terms:
            search_terms.append(plural)

    print(f"searching synonyms : {search_terms}")

    # search youtube about the term one by one
    audio_counter = 0

    # to clean up the captions
    punctutation_translator = str.maketrans('', '', string.punctuation)
    srt_tag_re = re.compile(r"<.*?>|\(.*?\)|\[.*?\]")

    # for audio processing
    extractor = SphinxSTTExtractor(keyword)
    audio_segmentor = yp.AudioSegmentor(keyword, output_dir)

    urls = []
    for search_term in search_terms:

        url_fetcher = YoutubeSearcher(youtube_api_key, search_term)
        cp.print_color(cp.ColorEnum.BOLD, f"search term : {search_term}")

        while True:
            url = url_fetcher.next()

            if len(url) == 0:
                cp.print_color(cp.ColorEnum.YELLOW, "there are no more urls to process")
                break

            url = url[0]

            print(f"keyword: {keyword}")
            print(f"searched term: {search_term}")
            print(f"url: {url}")

            if url in urls:
                cp.print_color(cp.ColorEnum.YELLOW, "the video is already added")
                continue

            # check for valid format and retreive captions
            srt_captions = retrieve_captions(url, keyword)
            if srt_captions is None:
                continue

            # prevent crawler from repeating same url
            urls.append(url)

            # crawl the video
            try:
                crawler = yp.YoutubeCrawler(url)
                audio_data = crawler.get_audio()
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exception:
                cp.print_color(cp.ColorEnum.YELLOW, "failed to download audio file for the video")
                cp.print_color(cp.ColorEnum.YELLOW, exception)
                continue

            # locate and segment keyword audio
            for caption in srt_captions:
                start_time, end_time, words = process_cation(keyword, caption, punctutation_translator, srt_tag_re)

                if words is None:
                    continue

                # occurance in captions
                cc_count = count_keyword(keyword, words)

                if cc_count == 0:
                    continue

                extracted_audio_counts = extract_keyword(url, extractor, audio_segmentor, audio_data, start_time, end_time, cc_count)

                if extracted_audio_counts > 0:
                    audio_counter += extracted_audio_counts
                    cp.print_color(cp.ColorEnum.GREEN, f"{keyword} - {audio_counter}/{data_size}")

            if audio_counter > data_size:
                break

        if audio_counter > data_size:
            cp.print_color(cp.ColorEnum.BOLD, f"successfully extracted {audio_counter} audios of {keyword}")
            break

    return audio_counter

def main():
    parser = ArgumentParser()

    parser.add_argument(
        "-y",
        "--youtube_api_key",
        type=str,
        required=True,
        help="API key for youtube data v3 API")

    parser.add_argument(
        "-w",
        "--words_api_key",
        type=str,
        required=True,
        help="API key for words API")

    parser.add_argument(
        "-k",
        "--keyword_list",
        nargs='+',
        type=str,
        required=True,
        help="list of keywords to collect")

    parser.add_argument(
        "-s",
        "--samples_per_keyword",
        type=int,
        default=10,
        help="number of samples to collect per keyword")

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./generated_keyword_audios",
        help="path to the output dir")

    args = parser.parse_args()
    file_utils.ensure_dir(TEMP_DIR)

    file_utils.ensure_dir(args.output_dir)

    collection_result = {}

    for index, keyword in enumerate(args.keyword_list):
        cp.print_color(cp.ColorEnum.BOLD, f"collecting {args.samples_per_keyword} audio samples of keyword : {keyword}")

        count = generate_dataset(args.youtube_api_key, args.words_api_key, keyword, args.samples_per_keyword, args.output_dir)

        collection_result[keyword] = count

    for keyword, count in collection_result.items():
        cp.print_color(cp.ColorEnum.BOLD, f"collected {count} keywords of {keyword}")

    file_utils.remove_dir(TEMP_DIR)

if __name__ == "__main__":
    main()
