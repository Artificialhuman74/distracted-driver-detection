"""Runs ON the Colab VM (sent by scripts/colab_run.sh via `colab exec -f`).
Downloads the REAL State Farm data with the Kaggle API. The token was uploaded as a file
(/content/kaggle_access_token) and is moved to ~/.kaggle/access_token here; it is never printed.
"""
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

home_kaggle = Path.home() / ".kaggle"
home_kaggle.mkdir(exist_ok=True)
src = Path("/content/kaggle_access_token")
if src.exists():
    shutil.move(str(src), home_kaggle / "access_token")
os.chmod(home_kaggle / "access_token", 0o600)

subprocess.run("pip install -q -U kaggle", shell=True, check=True)
data = Path("/content/data")
data.mkdir(exist_ok=True)
if not (data / "driver_imgs_list.csv").exists():
    subprocess.run("kaggle competitions download -c state-farm-distracted-driver-detection -p /content -q",
                   shell=True, check=True)
    with zipfile.ZipFile("/content/state-farm-distracted-driver-detection.zip") as z:
        members = [m for m in z.namelist() if m == "driver_imgs_list.csv" or m.startswith("imgs/train/")]
        z.extractall(data, members)   # labelled images only; skips the 79k unlabelled test images
    os.remove("/content/state-farm-distracted-driver-detection.zip")
n = sum(1 for _ in (data / "imgs" / "train").rglob("*.jpg"))
print(f"DATA READY: {n} labelled training images in {data}")
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; free -g | head -2", shell=True)
