"""Turn scripts/train_notebook_src.py (cells separated by '# %%') into kaggle/train.ipynb.

Usage:
    python scripts/build_notebook.py          # smoke run (SMOKE = True)
    python scripts/build_notebook.py --full   # full run  (SMOKE = False)
    python scripts/build_notebook.py --full --colab   # Google Colab version -> kaggle/train_colab.ipynb
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "scripts" / "train_notebook_src.py"
full = "--full" in sys.argv
colab = "--colab" in sys.argv
DST = ROOT / "kaggle" / ("train_colab.ipynb" if colab else "train.ipynb")
text = SRC.read_text()
text, n = re.subn(r"^SMOKE = (True|False)", f"SMOKE = {not full}", text, count=1, flags=re.M)
assert n == 1, "could not find the SMOKE line"


COLAB_SETUP = """# ---- Google Colab only: download the REAL State Farm data with the Kaggle API ----
# 1. Runtime > Change runtime type > T4 GPU.
# 2. Click the key icon (Secrets) on the left, add a secret named KAGGLE_API_TOKEN with your Kaggle token, enable notebook access.
#    The token is never printed or saved in this notebook.
# 3. You must have accepted the competition rules on kaggle.com.
import os, subprocess, zipfile
from pathlib import Path
from google.colab import userdata
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
tok_path = os.path.expanduser("~/.kaggle/access_token")
with open(tok_path, "w") as f:
    f.write(userdata.get("KAGGLE_API_TOKEN").strip())
os.chmod(tok_path, 0o600)
subprocess.run("pip install -q -U kaggle", shell=True, check=True)
data = Path("/content/data"); data.mkdir(exist_ok=True)
if not (data / "driver_imgs_list.csv").exists():
    subprocess.run("kaggle competitions download -c state-farm-distracted-driver-detection -p /content", shell=True, check=True)
    with zipfile.ZipFile("/content/state-farm-distracted-driver-detection.zip") as z:
        members = [m for m in z.namelist() if m == "driver_imgs_list.csv" or m.startswith("imgs/train/")]
        z.extractall(data, members)   # labelled images only; skips the 79k unlabelled test images
    os.remove("/content/state-farm-distracted-driver-detection.zip")
print("train images:", sum(1 for _ in (data / "imgs" / "train").rglob("*.jpg")))"""

COLAB_DOWNLOAD = """# ---- Google Colab only: zip the results and download them to your laptop ----
import shutil
from google.colab import files
shutil.make_archive("/content/outputs", "zip", "/content/outputs")
files.download("/content/outputs.zip")"""

cells = []
for chunk in re.split(r"^# %%", text, flags=re.M)[1:]:
    header, _, body = chunk.partition("\n")
    if header.strip() == "[markdown]":
        lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip().splitlines()]
        cells.append({"cell_type": "markdown", "id": f"cell{len(cells)}", "metadata": {}, "source": "\n".join(lines)})
    else:
        cells.append({"cell_type": "code", "id": f"cell{len(cells)}", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": body.strip("\n")})

if colab:
    mk = lambda src: {"cell_type": "code", "id": f"colab{len(cells)}", "metadata": {}, "execution_count": None, "outputs": [], "source": src}
    cells.insert(1, mk(COLAB_SETUP))
    cells.append(mk(COLAB_DOWNLOAD))
    for i, c in enumerate(cells):
        c["id"] = f"cell{i}"

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"},
                 **({"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []}} if colab else {})},
    "nbformat": 4, "nbformat_minor": 5,
}
DST.write_text(json.dumps(nb, indent=1))
print(f"wrote {DST.relative_to(ROOT)} with {len(cells)} cells (SMOKE = {not full}, colab = {colab})")
