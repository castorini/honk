# Speech Command Recognition

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
