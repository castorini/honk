# Keyword Spotting Data Generator
---
In order to add flexibility of keyword spotting, we are working on dataset generator using youtube videos. Key idea is to decrease the search space by utilizing subtitles.

This is still in development but it's possible to generate some dataset.
Note that current version has precision of ~ 0.5.

## < Preparation >
___
1. Current version is implemented with technique called [forced alignment](https://github.com/pettarin/forced-alignment-tools#definition-of-forced-alignment). Install [Aeneas](https://github.com/readbeyond/aeneas#system-requirements-supported-platforms-and-installation). If it complains about `/usr/bin/ld: cannot find -lespeak`, this [page](https://github.com/readbeyond/aeneas/issues/189) may help.
2. Instal necessary packages by running `pip install -r requirements.txt`
3. [Obtain a Google API key](https://support.google.com/googleapi/answer/6158862?hl=en), and set `API_KEY = google_api_key` in `search.py`

## < Usage >
___
### Generating Dataset

```
python keyword_data_generator.py -k < keywords to search > -s < size of keyword >
```

### Filtering Correct Audios
by running `drop_audio.py` script, user can manually drop false positive audios. This script plays the audio in the folder and asks whether the audio file contains target keyword.

```
python3 drop_audio.py < folder_name >
```

## < Improvements >
___
- filtering non-english videos
- ffmpeg handling more dynamic vidoe types : mov,mp4,m4a,3gp,3g2,mj2
- if video contains any of target words, generate a block
- dynamic handling of long videos (currently simple filter)
- increase the number of youtube videos retrieved from search (ex. searching similar words)
- increase rate of finding target term by stemming words

## Evaluation of Improvements
In order to quantify the improvements, we are working on evaluation framework which measures the quality of selected audio. We are hoping that this helps us to develop robust keyword spotting data generator.

Evaluation process involves following steps:

1. `python url_file_generator.py` : collect urls which contains target keyword in the audio and store it in a single .txt file (url file)
2. `evaluation_data_generator.py` : for each audio block containing target keyword, record how many times the target keyword actually appear; csv file is generated summarizing details of each audio block (summary file)
3. `evaluation_audio_generator.py` : generate audio dataset from summary file
4. `evaluate.py` : measure the quality of the specified similar audio extraction algorithm on given summary file

##### Setting up Experiment
After cloning this repo, run following command to clone submodule [kws-gen-data](https://github.com/castorini/kws-gen-data)
`git submodule update --init --recursive`

##### `url_file_generator.py`
Collect urls of videos which subtitle contains target keywords

```
python url_file_generator.py
	-a < youtube data v3 API key >
	-k < keywords to search >
	-s < number of urls >
```

##### `evaluation_data_generator.py`
For each audio block with keyword, allow users to record how many times the target keyword actually appear. This is the ground truth for measuring quality.
A csv file generated is called a summary file where each column represents `url`, `start_ms`, `end_ms`, `cc_count`, `audio_count`
- url - unique id of youtube video
- start_ms - start time of the given subtitle section
- end_ms - end time of the given subtitle section
- cc_count - how many keyword appeared in subtitle
- audio_count - how many time keyword appeared in the audio (user input)

```
python evaluation_data_generator.py
	-a < youtube data v3 API key >
	-k < keywords to search >
	-s < number of urls >
	-f < url file name (when unspecified, directly search youtube) >
	-c < url in url file to start from >
	-l < length of maximum length for a video (s) >
	-o < output csv file to append output to >
```

##### `evaluation_data_generator.py`
Generate set of `.wav` files from the provided summary file

```
python evaluation_audio_generator.py
	-a < youtube data v3 API key >
	-k < keywords to search >
	-f < summary file >
```

##### `evaluate.py`
Measure the quality of the specified similar audio retrieval process on given summary file

```
python evaluation_audio_generator.py
	-k < keywords to search >
	-f < summary file >
	-r < type of extraction algorithm to use >
	-th < threshold for retrieving a window >
```
