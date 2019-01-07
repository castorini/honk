import librosa
import os
import pyaudio
import subprocess
import sys
import wave
from tqdm import tqdm

if len(sys.argv) < 2:
    print("usage: python3 drop_audio.py <folder_name>")
    sys.exit()

DIR_NAME = sys.argv[1]

def play_audio(file_name) :
    subprocess.check_output(["ffplay", "-nodisp", "-autoexit", file_name])

file_list = os.listdir(DIR_NAME)
total_count = len(file_list)

delete_count = 0
for file_name in tqdm(file_list):
    keep = ''
    path = os.path.join(DIR_NAME, file_name)
    print(path)
    if not file_name.endswith("wav"):
        os.remove(path)
        continue

    while keep != "s" and keep != "d":
        play_audio(path)
        keep = input("\n\n> keep? (yes = s / no = d)\n")

    if keep == "d":
        print("deleting audio ...")
        delete_count += 1;
        os.remove(path)

remaining_count = total_count - delete_count

print("deleted : " + str(delete_count))
print("false positive : " + str(round(100*delete_count/total_count)) + " %")
print("\nremaining : " + str(remaining_count))
print("true positive : " + str(round(100*remaining_count/total_count)) + " %")
print("\ntotal : " + str(total_count))
