# keyword_spotting_data_generator

## Preparation:

1. Install [Aeneas](https://github.com/readbeyond/aeneas#system-requirements-supported-platforms-and-installation)
   
   (if `/usr/bin/ld: cannot find -lespeak` error occured, this [page](https://github.com/readbeyond/aeneas/issues/189) may help )
2. [Obtain a Google API key](https://support.google.com/googleapi/answer/6158862?hl=en), and set `API_KEY = google_api_key` in `search.py`


## Usage :

### Crawling YouTube

```
python keyword_data_generator.py -k < keywords to search > -s < size of keyword >
```

### Filtering True Positive Audios

```
python3 drop_audio.py <folder_name>
```


## Improvements :

- ffmpeg handling more dynamic vidoe types : mov,mp4,m4a,3gp,3g2,mj2
- if video contains any of target words, generate a block
- dynamic handling of long videos (currently simple filter)
- increases rate of finding target term by stemming words
