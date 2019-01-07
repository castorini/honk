import os
import subprocess
import sys
from tqdm import tqdm

if len(sys.argv) < 2:
    print("usage: python3 drop_audio.py <folder_name>")
    sys.exit()

DIR_NAME = sys.argv[1]

def play_audio(file):
    subprocess.check_output(["ffplay", "-nodisp", "-autoexit", file])

FILE_LIST = os.listdir(DIR_NAME)
TOTAL_COUNT = len(FILE_LIST)

DELETE_COUNT = 0
for file_name in tqdm(FILE_LIST):
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
        DELETE_COUNT += 1
        os.remove(path)

REMAINING_COUNT = TOTAL_COUNT - DELETE_COUNT

print("deleted : " + str(DELETE_COUNT))
print("false positive : " + str(round(100*DELETE_COUNT/TOTAL_COUNT)) + " %")
print("\nremaining : " + str(REMAINING_COUNT))
print("true positive : " + str(round(100*REMAINING_COUNT/TOTAL_COUNT)) + " %")
print("\ntotal : " + str(TOTAL_COUNT))
