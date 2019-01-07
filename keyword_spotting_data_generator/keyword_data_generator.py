import datetime
import json
import librosa
import math
import numpy as np
import os
import re
import search
import string
import subprocess
import sys
import time
import wordset
from argparse import ArgumentParser
from aeneas.executetask import ExecuteTask
from aeneas.syncmap.fragment import SyncMapFragment
from aeneas.task import Task
from pytube import YouTube

class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

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

def srt_time_to_ms(h, m, s, ms):
	converted = int(ms)
	converted += (1000 * int(s))
	converted += (1000 * 60 * int(m))
	converted += (1000 * 60 * 60 * int(h))
	return converted

def pad_and_center_align(arr, size):
	pad_size = size - len(arr)
	left_pad = math.floor(pad_size/2)
	right_pad = pad_size - left_pad
	return np.pad(arr, (left_pad, right_pad), 'constant')

url_template = "http://youtube.com/watch?v={}"
ffmpeg_template = "ffmpeg -i {0}.mp4 -codec:a pcm_s16le -ac 1 {0}.wav"

tag_cleanr = re.compile('<.*?>|\(.*?\)|\[.*?\]')
srt_time_parser = re.compile("(\d+):(\d+):(\d+),(\d+)\s-->\s(\d+):(\d+):(\d+),(\d+)")
translator = str.maketrans('', '', string.punctuation)

aligner_config_string = "task_language=eng|is_text_type=plain|os_task_file_format=json"

def clean_up_temp_files():
	for filename in os.listdir(TEMP_DIR):
		os.remove(TEMP_DIR + "/" + filename)

def retrieve_keyword_audio(vid, title, keyword):
	audio_index = 0
	v_url = url_template.format(vid)
	yt = YouTube(v_url)

	if int(yt.length) > 2700:
		# only consider video < 45 mins
		return audio_index

	caption = yt.captions.get_by_language_code('en')
	if caption:
		# retrieve audio from video
		yt.streams.first().download(output_path=TEMP_DIR, filename=vid)

		temp_file_name = TEMP_DIR+vid
		if not os.path.isfile(temp_file_name + ".mp4"):
			return audio_index

		time.sleep(1) # need to wait before ffmpeg takes in as input file
		cmd = ffmpeg_template.format(temp_file_name).split()
		subprocess.check_output(cmd)

		audio = librosa.core.load(temp_file_name+".wav", 16000)[0]

		os.remove(temp_file_name + ".mp4")
		os.remove(temp_file_name + ".wav")
		
		formatted_vid = vid.replace('_', '-')

		cc_arr = caption.generate_srt_captions().split('\n\n')
		for cc in cc_arr:
			cc_split = cc.split('\n')
			if len(cc_split) == 4 and cc_split[0] == '':
				cc_split = (cc_split[1], cc_split[2], cc_split[3])
			elif len(cc_split) != 3:
				continue

			cc_index, cc_time, cc_text = cc_split
			cc_text = tag_cleanr.sub('', cc_text)
			
			# clean up punctuation
			cc_text = cc_text.translate(translator)

			cc_text = cc_text.lower()
			words = cc_text.strip().split()

			# steming words
			if keyword not in words and keyword + "s" not in words and keyword + "es" not in words:
				continue

			aligner_task = Task(config_string=aligner_config_string)

			# prepare label file for forced aligner

			label_file = temp_file_name + ".txt"
			with open(label_file, "w+") as f:
				for word in words:
					f.write(word+"\n")
			f.close()

			# prepare audio file for forced aligner

			match_result = srt_time_parser.match(cc_time)
			if match_result:
				start_time_ms = srt_time_to_ms(match_result.group(1), match_result.group(2), match_result.group(3), match_result.group(4))
				stop_time_ms = srt_time_to_ms(match_result.group(5), match_result.group(6), match_result.group(7), match_result.group(8))

				start_pos = start_time_ms * 16
				stop_pos = stop_time_ms * 16

				block = audio[start_pos:stop_pos] # *16 since 16 samples are captured per each ms

				# temporary audio file for forced aligner
				audio_file = temp_file_name + ".wav"
				librosa.output.write_wav(audio_file, block, 16000)
				time.sleep(1) # buffer for writing wav file

			else:
				print(bcolors.FAIL + "failed pasing srt time : " + cc_time + bcolors.ENDC)
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
					librosa.output.write_wav(DATA_DIR + "/" + keyword + "/" + file_name, keyword_audio, 16000)
					audio_index += 1

	return audio_index

def generate_dataset(keyword, data_size):
	audio_counter = 0

	youtube_search_terms = wordset.get_relevant_words(keyword)
	print("\n" + bcolors.WARNING + str(datetime.datetime.now()) + " - search terms for keyword " + str(keyword) + " : " + str(youtube_search_terms) + " (" + str(len(youtube_search_terms)) + ")" + bcolors.ENDC)
	term_index = 0

	token = None
	while audio_counter < data_size and term_index < len(youtube_search_terms):
		term = youtube_search_terms[term_index]
		token, video_dict = grab_videos(term, token=token)
		if len(video_dict) == 0 or token == "last_page":
			token = None
			term_index += 1

		for vid, title in video_dict.items():
			print(vid + " - " + title)
			prev_counter = audio_counter
			try:
				audio_counter += retrieve_keyword_audio(vid, title, keyword)
			except KeyboardInterrupt:
				print(bcolors.FAIL + "keyboard interruption. terminating ..." + bcolors.ENDC)
				sys.exit()
			except Exception as e:
				print(bcolors.FAIL)
				print("an error has occured while processing video")
				print(e)
				print(bcolors.ENDC)
			finally:
				time.sleep(1)
				clean_up_temp_files()
			if prev_counter < audio_counter:
				print("\n" + bcolors.WARNING + str(datetime.datetime.now()) + " - collected " + str(audio_counter) + " / " + str(data_size) + " " + keyword + bcolors.ENDC)
	return audio_counter

def main():
	parser = ArgumentParser()
	parser.add_argument("-s", "--size_per_keyword", dest="data_size", type=int, default=100)
	parser.add_argument('-k', '--keyword_list', nargs='+', dest="keyword_list", type=str, required=True)

	args = parser.parse_args()
	print(bcolors.WARNING + str(datetime.datetime.now()) + " - collecting " + str(args.data_size) + " audios of keywords : " + str(args.keyword_list) + bcolors.ENDC)

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
		print("\n" + bcolors.WARNING + str(datetime.datetime.now()) + " - completed collecting " + str(index) + " th keyword (" + str(len(args.keyword_list)) + ") : " + keyword + " - " + str(count) + bcolors.ENDC)
		print("\n" + bcolors.WARNING + "\t took " + str(minute_elasped) + " minutes" + bcolors.ENDC)

		collection_result[keyword] = count

	for keyword, count in collection_result.items():
		print(bcolors.OKGREEN + keyword + " - " + str(count) + bcolors.ENDC)

	os.rmdir(TEMP_DIR)

if __name__ == "__main__":
	main()
