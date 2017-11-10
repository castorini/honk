import argparse
import enum
import json
import wave

from .manage_audio import AudioSnippetGenerator
from server import load_service

class KeyInput(enum.Enum):
    QUIT = b"q"
    REDO = b"\x1b[A"

def record_speech_sequentially(min_sound_lvl=0.01, speech_timeout_secs=1.):
    """Records audio in sequential audio files.

    Args:
        min_sound_lvl: The minimum sound level as measured by root mean square
        speech_timeout_secs: Timeout of audio after that duration of silence as measured by min_sound_lvl

    Returns:
        The recorded audio samples.
    """
    samples = []
    i = 0
    while True:
        cmd = input("> ").encode()
        if cmd == KeyInput.QUIT.value:
            return samples
        elif cmd == KeyInput.REDO.value:
            print("Index now at {}.".format(i))
            i = max(i - 1, 0)
            try:
                samples.pop()
            except IndexError:
                pass
            continue
        with AudioSnippetGenerator() as generator:
            timeout_len = int(speech_timeout_secs * generator.sr / generator.chunk_size)
            active_count = timeout_len
            curr_snippet = None
            for audio in generator:
                if curr_snippet:
                    curr_snippet.append(audio)
                else:
                    curr_snippet = audio
                if audio.amplitude_rms() < min_sound_lvl:
                    active_count -= 1
                else:
                    active_count = timeout_len
                print("Time left: {:<10}".format(active_count), end="\r")
                if active_count == 0:
                    i += 1
                    samples.append(curr_snippet)
                    print("Recorded #{:<10}".format(i))
                    break

def trim_sequence(samples, cutoff_ms):
    for sample in samples:
        n_samples = int((cutoff_ms / 1000) * 16000)
        sample.trim_window(n_samples * 2)
    return samples

def do_record_sequence():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-sound-lvl", type=float, default=0.01, 
        help="Minimum sound level at which audio is not considered silent")
    parser.add_argument(
        "--timeout-seconds", type=float, default=1.,
        help="Duration of silence after which recording halts")
    flags, _ = parser.parse_known_args()
    return record_speech_sequentially(min_sound_lvl=flags.min_sound_lvl, speech_timeout_secs=flags.timeout_seconds)

def do_trim(audio_samples):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff-ms", type=int, default=1000)
    flags, _ = parser.parse_known_args()
    trim_sequence(audio_samples, flags.cutoff_ms)

def do_discard_true(audio_samples):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--correct-label", type=str)
    flags, _ = parser.parse_known_args()
    with open(flags.config) as f:
        config = json.loads(f.read())
    lbl_service = load_service(config)
    false_samples = []
    for i, snippet in enumerate(audio_samples):
        label, _ = lbl_service.label(snippet.byte_data)
        if label != flags.correct_label:
            action = "Keep"
            false_samples.append(snippet)
        else:
            action = "Discard"
        print("#{:<5} Action: {:<8} Label: {}".format(i, action, label))
    return false_samples

def main():
    parser = argparse.ArgumentParser()
    record_choices = ["sequence"]
    process_choices = ["discard_true", "trim"]
    parser.add_argument("--mode", type=str, default="sequence", choices=record_choices)
    parser.add_argument("--output-begin-index", type=int, default=0)
    parser.add_argument("--output-prefix", type=str, default="output")
    parser.add_argument("--post-process", nargs="+", type=str, choices=process_choices, default=[])
    args, _ = parser.parse_known_args()

    if args.mode == "sequence":
        audio_samples = do_record_sequence()
    for choice in args.post_process:
        if choice == "discard_true":
            audio_samples = do_discard_true(audio_samples)
        elif choice == "trim":
            do_trim(audio_samples)

    for i, snippet in enumerate(audio_samples):
        fullpath = "{}.{}.wav".format(args.output_prefix, i)
        with wave.open(fullpath, "w") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(snippet.byte_data)
        print("Saved {}.".format(fullpath))

if __name__ == "__main__":
    main()