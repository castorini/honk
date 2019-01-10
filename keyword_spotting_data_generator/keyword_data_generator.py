import datetime
import math
import os
import re
import string
import subprocess
import sys
import time
from argparse import ArgumentParser
import numpy as np
import librosa
from aeneas.executetask import ExecuteTask
from aeneas.task import Task
from pytube import YouTube
import wordset
import search

TEXT_COLOUR = {
    'HEADER' : '\033[95m',
    'OKBLUE' : '\033[94m',
    'OKGREEN' : '\033[92m',
    'WARNING' : '\033[93m',
    'FAIL' : '\033[91m',
    'ENDC' : '\033[0m',
    'BOLD' : '\033[1m',
    'UNDERLINE' : '\033[4m'
}

DATA_DIR = "data"
TEMP_DIR = DATA_DIR + "/temp/"

def grab_videos(keyword, token=None):
    res = search.youtube_search(keyword, max_results=50, token=token)
    token = res[0]
    videos = res[1]
    video_dict = {}

    for video_data in videos:
        # print(json.dumps(video_data, indent=4, sort_keys=True))
        video_id = video_data['id']['videoId']
        video_title = video_data['snippet']['title']
        video_dict[video_id] = video_title
    print("search returned " + str(len(videos)) + " videos")
    return token, video_dict

def srt_time_to_ms(hour, minute, second, msecond):
    converted = int(msecond)
    converted += (1000 * int(second))
    converted += (1000 * 60 * int(minute))
    converted += (1000 * 60 * 60 * int(hour))
    return converted

def pad_and_center_align(arr, size):
    pad_size = size - len(arr)
    left_pad = math.floor(pad_size/2)
    right_pad = pad_size - left_pad
    return np.pad(arr, (left_pad, right_pad), 'constant')

URL_TEMPLATE = "http://youtube.com/watch?v={}"
FFMPEG_TEMPLATE = "ffmpeg -i {0}.mp4 -codec:a pcm_s16le -ac 1 {0}.wav"

TAG_CLEANER = re.compile(r"<.*?>|\(.*?\)|\[.*?\]")
SRT_TIME_PARSER = re.compile(r"(\d+):(\d+):(\d+),(\d+)\s-->\s(\d+):(\d+):(\d+),(\d+)")
TRANSPLATOR = str.maketrans('', '', string.punctuation)

ALIGNER_CONFIG_STRING = "task_language=eng|is_text_type=plain|os_task_file_format=json"

def clean_up_temp_files():
    for filename in os.listdir(TEMP_DIR):
        os.remove(TEMP_DIR + "/" + filename)

def retrieve_keyword_audio(vid, keyword):
    audio_index = 0
    v_url = URL_TEMPLATE.format(vid)
    youtube = YouTube(v_url)

    if int(youtube.length) > 2700:
        # only consider video < 45 mins
        return audio_index

    caption = youtube.captions.get_by_language_code('en')
    if caption:
        # retrieve audio from video
        youtube.streams.first().download(output_path=TEMP_DIR, filename=vid)

        temp_file_name = TEMP_DIR+vid
        if not os.path.isfile(temp_file_name + ".mp4"):
            return audio_index

        time.sleep(1) # need to wait before ffmpeg takes in as input file
        cmd = FFMPEG_TEMPLATE.format(temp_file_name).split()
        subprocess.check_output(cmd)

        audio = librosa.core.load(temp_file_name+".wav", 16000)[0]

        os.remove(temp_file_name + ".mp4")
        os.remove(temp_file_name + ".wav")

        formatted_vid = vid.replace('_', '-')

        cc_arr = caption.generate_srt_captions().split('\n\n')
        for captions in cc_arr:
            cc_split = captions.split('\n')
            if len(cc_split) == 4 and cc_split[0] == '':
                cc_split = (cc_split[1], cc_split[2], cc_split[3])
            elif len(cc_split) != 3:
                continue

            _, cc_time, cc_text = cc_split
            cc_text = TAG_CLEANER.sub('', cc_text)

            # clean up punctuation
            cc_text = cc_text.translate(TRANSPLATOR)

            cc_text = cc_text.lower()
            words = cc_text.strip().split()

            # steming words
            if keyword not in words and keyword + "s" not in words and keyword + "es" not in words:
                continue

            aligner_task = Task(config_string=ALIGNER_CONFIG_STRING)

            # prepare label file for forced aligner

            label_file = temp_file_name + ".txt"
            with open(label_file, "w+") as file:
                for word in words:
                    file.write(word+"\n")

            # prepare audio file for forced aligner

            match_result = SRT_TIME_PARSER.match(cc_time)
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

                start_pos = start_time_ms * 16
                stop_pos = stop_time_ms * 16

                block = audio[start_pos:stop_pos] # *16 since 16 samples are captured per each ms

                # temporary audio file for forced aligner
                audio_file = temp_file_name + ".wav"
                librosa.output.write_wav(audio_file, block, 16000)
                time.sleep(1) # buffer for writing wav file

            else:
                print(TEXT_COLOUR['FAIL'] + "failed pasing srt time : "
                      + cc_time + TEXT_COLOUR['ENDC'])
                raise Exception('srt time fail error')

            aligner_task.text_file_path_absolute = label_file
            aligner_task.audio_file_path_absolute = audio_file

            # process aligning task
            ExecuteTask(aligner_task).execute()

            for fragment in aligner_task.sync_map_leaves():
                if fragment.is_regular and keyword in fragment.text and fragment.length < 0.9:
                    begin = int(fragment.begin * 16000)
                    end = int(fragment.end * 16000)
                    keyword_audio = pad_and_center_align(block[begin:end], 16000)

                    file_name = formatted_vid+"_"+str(audio_index)+".wav"
                    librosa.output.write_wav(
                        DATA_DIR + "/" + keyword + "/" + file_name, keyword_audio, 16000)
                    audio_index += 1

    return audio_index

def generate_dataset(keyword, data_size):
    audio_counter = 0

    youtube_search_terms = wordset.get_relevant_words(keyword)
    print("\n" + TEXT_COLOUR['WARNING'] + str(datetime.datetime.now())
          + " - search terms for keyword " + str(keyword) + " : " + str(youtube_search_terms)
          + " (" + str(len(youtube_search_terms)) + ")" + TEXT_COLOUR['ENDC'])
    term_index = 0

    token = None
    while audio_counter < data_size and term_index < len(youtube_search_terms):
        term = youtube_search_terms[term_index]
        token, video_dict = grab_videos(term, token=token)
        if not video_dict or token == "last_page":
            token = None
            term_index += 1

        for vid, title in video_dict.items():
            print(vid + " - " + title)
            prev_counter = audio_counter
            try:
                audio_counter += retrieve_keyword_audio(vid, keyword)
            except KeyboardInterrupt:
                print(TEXT_COLOUR['FAIL'] + "keyboard interruption. terminating ..."
                      + TEXT_COLOUR['ENDC'])
                sys.exit()
            except Exception as exception:
                print(TEXT_COLOUR['FAIL'])
                print("an error has occured while processing video")
                print(exception)
                print(TEXT_COLOUR['ENDC'])
            finally:
                time.sleep(1)
                clean_up_temp_files()
            if prev_counter < audio_counter:
                print("\n" + TEXT_COLOUR['WARNING'] + str(datetime.datetime.now())
                      + " - collected " + str(audio_counter) + " / " + str(data_size)
                      + " " + keyword + TEXT_COLOUR['ENDC'])
    return audio_counter

def main():
    parser = ArgumentParser()
    parser.add_argument("-s", "--size_per_keyword", dest="data_size", type=int, default=100)
    parser.add_argument('-k', '--keyword_list', nargs='+', dest="keyword_list",
                        type=str, required=True)

    args = parser.parse_args()
    print(TEXT_COLOUR['WARNING'] + str(datetime.datetime.now()) + " - collecting "
          + str(args.data_size) + " audios of keywords : " + str(args.keyword_list)
          + TEXT_COLOUR['ENDC'])

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    collection_result = {}

    for index, keyword in enumerate(args.keyword_list):
        start_time = datetime.datetime.now()
        output_dir = DATA_DIR + "/" + keyword + "/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        count = generate_dataset(keyword.lower(), args.data_size)
        finish_time = datetime.datetime.now()

        minute_elasped = math.floor((finish_time - start_time).total_seconds() / 60.0)
        print("\n" + TEXT_COLOUR['WARNING'] + str(datetime.datetime.now())
              + " - completed collecting " + str(index) + " th keyword ("
              + str(len(args.keyword_list)) + ") : " + keyword + " - " + str(count)
              + TEXT_COLOUR['ENDC'])
        print("\n" + TEXT_COLOUR['WARNING'] + "\t took " + str(minute_elasped) + " minutes"
              + TEXT_COLOUR['ENDC'])

        collection_result[keyword] = count

    for keyword, count in collection_result.items():
        print(TEXT_COLOUR['OKGREEN'] + keyword + " - " + str(count) + TEXT_COLOUR['ENDC'])

    os.rmdir(TEMP_DIR)

if __name__ == "__main__":
    main()
