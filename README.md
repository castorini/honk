# Honk: Speech Command Recognition
## Speech commands demo
Currently, PyTorch has official support for only Linux and OS X. Thus, Windows users will not be able to run this demo easily.

To deploy the demo, run the following commands:
- If you do not have PyTorch, please see [the website](http://pytorch.org).
- Install Python dependencies: `pip install -r requirements.txt`
- Install GLUT through your package manager (e.g. `apt-get install freeglut3-dev`)
- Start the PyTorch server: `python .`
- Run the demo: `python utils/speech_demo.py`

Please ensure that you have a working microphone. If you cannot get the demo working but would still like to see it in action, please see [the video](https://www.youtube.com/watch?v=31J4CD6VhX4).

## Server
### Setup and deployment
`python .` deploys the web service for identifying if audio contain the command word. By default, `config.json` is used for configuration, but that can be changed with `--config=<file_name>`.

If the server is behind a firewall, one workflow is to create an SSH tunnel and use port forwarding with the port specified in config (default 16888).

### Endpoint specifications
```
POST /listen
```
Args (JSON):
* `wav_data`: 16kHz sampling rate, 16-bit PCM mono-channel raw audio data (with no WAVE header), gzipped and base64-encoded.

Returns (JSON):
* `contains_command`: `true` if `wav_data` contains the command word, `false` otherwise.

For a real-time example, please see `utils/client.py`.

## Utilities
### QA client
Unfortunately, the QA client has no support for the general public yet, since it requires a custom QA service. However, it can still be used to retarget the command keyword.

`python client.py` runs the QA client. You may retarget a keyword by doing `python client.py --mode=retarget`. Please note that text-to-speech may not work well on Linux distros; in this case, please supply IBM Watson credentials via `--watson-username` and `--watson--password`. You can view all the options by doing `python client.py -h`.

### Training and evaluating the model
`python model.py --mode [train|eval]` trains or evaluates the model. It expects all training examples to follow the same format as that of [Speech Commands Dataset](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz). The recommended workflow is to download the dataset and add custom keywords, since the dataset already contains many useful audio samples and background noise.

There are command options available:

| option         | input format | default | description |
|----------------|--------------|---------|-------------|
| `--batch_size`   | [1, n)       | 100     | the mini-batch size to use            |
| `--cache_size`   | [0, inf)       | 32768     | number of items in audio cache, consumes around 32 KB * n   |
| `--conv1_pool`   | [1, inf) [1, inf) | 2 2     | the width and height of the pool filter       |
| `--conv1_size`     | [1, inf) [1, inf) | 10 4  | the width and height of the conv filter            |
| `--conv1_stride`     | [1, inf) [1, inf) | 1 1  | the width and length of the stride            |
| `--conv2_pool`   | [1, inf) [1, inf) |  1 1    | the width and height of the pool filter            |
| `--conv2_size`     | [1, inf) [1, inf) | 10 4  | the width and height of the conv filter            |
| `--conv2_stride`     | [1, inf) [1, inf) | 1 1  | the width and length of the stride            |
| `--data_folder`   | string       | /data/speech_dataset     | path to data       |
| `--dev_every`    | [1, inf)     |  10     | dev interval in terms of epochs            |
| `--dev_pct`   | [0, 100]       | 10     | percentage of total set to use for dev        |
| `--dropout_prob` | [0.0, 1.0)   | 0.5     | the dropout rate to use            |
| `--gpu_no`     | [-1, n] | 1  | the gpu to use            |
| `--input_file`   | string       |      | the path to the model to load   |
| `--input_length`   | [1, inf)       | 16000     | the length of the audio   |
| `--lr`           | (0.0, inf)   | 0.001   | the learning rate to use            |
| `--mode`         | {train, eval}| train   | the mode to use            |
| `--n_dct_filters`| [1, inf)     | 40      | the number of DCT bases to use  |
| `--n_epochs`     | [0, inf) | 500  | number of epochs            |
| `--n_feature_maps1` | [1, inf)             | 64        | the number of feature maps for conv net 1            |
| `--n_feature_maps2`   | [1, inf)       | 64     | the number of feature maps for conv net 2        |
| `--n_labels`   | [1, n)       | 4     | the number of labels to use            |
| `--n_mels`       | [1, inf)     |   40    | the number of Mel filters to use            |
| `--no_cuda`      | switch     | false   | whether to use CUDA            |
| `--noise_prob`     | [0.0, 1.0] | 0.8  | the probability of mixing with noise    |
| `--output_file`   | string    | model/model.pt     | the file to save the model to        |
| `--seed`   | (inf, inf)       | 0     | the seed to use        |
| `--silence_prob`     | [0.0, 1.0] | 0.1  | the probability of picking silence    |
| `--test_pct`   | [0, 100]       | 10     | percentage of total set to use for testing       |
| `--timeshift_ms`| [0, inf)       | 100    | time in milliseconds to shift the audio randomly |
| `--train_pct`   | [0, 100]       | 80     | percentage of total set to use for training       |
| `--unknown_prob`     | [0.0, 1.0] | 0.01  | the probability of picking an unknown word    |
| `--wanted_words` | string1 string2 ... stringn  | command random  | the desired target words            |

### Recording audio

You may do the following to record sequential audio and save to the same format as that of speech command dataset:
```bash
python manage_audio.py record
```
Input any key (return is fastest) to open the microphone. After one second of silence, recording automatically halts.

Several options are available:
```
--output-prefix: Prefix of the output audio sequence
--min-sound-lvl: Minimum sound level at which audio is not considered silent
--timeout-seconds: Duration of silence after which recording halts
--output-begin-index: Starting sequence number
```

### Listening to sound level

```bash
python manage_audio.py listen
```

This assists in setting sane values for `--min-sound-lvl` for recording.

### Trimming audio
Speech command dataset contains one-second-long snippets of audio.

`python manage_audio.py trim [directory]` trims to the loudest one-second for all .wav files in `[directory]`. The careful user should manually check all audio samples using an audio editor like Audacity.
