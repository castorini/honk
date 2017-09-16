# Speech Command Recognition
## Server
### Setup and deployment
`python speech-command-recognition` deploys the web service for identifying if audio contain "anserini." By default, `config.json` is used for configuration, but that can be changed with `--config=<file_name>`.

If the server fails to start with Tensorflow errors, then you don't have the Tensorflow nightly version. You may find that information [here](https://hub.docker.com/r/tensorflow/tensorflow/tags/). Tensorflow 1.4 will have the missing modules.

Since the server is behind a firewall, one workflow is to create an SSH tunnel and use port forwarding with the port specified in config (default 16888).

### Endpoint specifications
```
POST /listen
```
Args (JSON):
* `wav_data`: 16kHz sampling rate, 16-bit PCM mono-channel raw audio data (with no WAVE header), gzipped and base64-encoded.

Returns (JSON):
* `contains_command`: `true` if `wav_data` contains "anserini," `false` otherwise.

For a real-time example, please see `utils/client.py`.

## Utilities
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

```bash
python manage_audio.py trim [directory]
```
trims (from the right) all .wav files in `[directory]`. The user should manually check all audio files beforehand using an audio editor.
