# Keyword Spotting Data Generator
---
In order to improve the flexibility of [Honk](https://github.com/castorini/honk) and [Honkling](https://github.com/castorini/honkling), we provide a program that constructs a dataset from youtube videos.
Key idea is to decrease the search space by utilizing subtitles and extract target audio using [PocketSphinx](https://github.com/cmusphinx/pocketsphinx).

## < Preparation >
- Necessary python packages can be downloaded with `pip -r install requirements.txt`
- [ffmpeg](https://www.ffmpeg.org/) and [SoX](http://sox.sourceforge.net/) must be available as well.
- YouTube Data API - follow [this instruction](https://developers.google.com/youtube/v3/getting-started) to obtain a new API key

## < Usage >
```
python keyword_data_generator.py
	-a < youtube data v3 API key >
	-k < list of keywords to search >
	-s < number of samples to collect per keyword (default: 10) >
	-o < output path (default: "./generated_keyword_audios") >
```

example:
```
python keyword_data_generator.py -a $YOUTUBE_API_KEY -k google slack -s 20 -o ./generated
```

## < Improvements >
___
- filtering non-english videos
- adjust ffmpeg command to handle different types of video : mov,mp4,m4a,3gp,3g2,mj2
- dynamic handling of long videos (currently simple filter)
- improve throughput by parallelizing the process
