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
In order to quantize the improvements, we are working on evaluation framework. We are hoping that this helps us to develop robust keyword spotting data generator.

##### Evaluation Data Generator
Dataset which we run our evaluation on must be correctly labeled,
for given keyword or list of url, `evaluation_data_generator.py` generates csv with each column representing `url`, `start_ms`, `end_ms`, `cc_count`, `audio_count`
- url - unique id of youtube video
- start_ms - start time of the given subtitle section
- end_ms - end time of the given subtitle section
- cc_count - how many keyword appeared in subtitle
- audio_count - how many time keyword appeared in the audio (user input)

```
python url_file_generator.py -a < youtube data v3 API key > -k < keywords to search > -s < number of urls >
```

##### URL file
In order to ease distribution of video list which we use, `url_file_generator.py` script can be used to generate .txt file with urls.

```
python evaluation_data_generator.py -a < youtube data v3 API key > -k < keywords to search > -s < number of urls > -l < length of maximum length for a video (s) >
```

if url_file is not specified, it will search youtube on the fly.
