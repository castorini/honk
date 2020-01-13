import shutil
from pathlib import Path

def ensure_dir(dir_path):
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True, exist_ok=False)

def remove_dir(dir_path):
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        shutil.rmtree(dir_path)
