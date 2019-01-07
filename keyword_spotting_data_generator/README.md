# keyword_spotting_data_generator

https://github.com/readbeyond/aeneas

https://github.com/readbeyond/aeneas/issues/189


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
