#!/usr/bin/env bash
# Run the FULL training on Google Colab through the official Colab CLI, then download the results.
#   One-time login first (you do this):  .venv/bin/colab sessions
#   Then:  bash scripts/colab_run.sh [GPU]      GPU = T4 (free) | L4 | A100 (paid compute units)
# Results -> runs/colab/outputs/   (copy to outputs/ once checked)
set -euo pipefail
cd "$(dirname "$0")/.."
COLAB=".venv/bin/colab"
GPU="${1:-T4}"
S="dd-f28"
mkdir -p runs/colab

# The training code is the same source as the Kaggle notebook, with SMOKE switched off.
sed 's/^SMOKE = True/SMOKE = False/' scripts/train_notebook_src.py > runs/colab/train_full.py
grep -q '^SMOKE = False' runs/colab/train_full.py

echo "== creating Colab session $S with GPU $GPU"
$COLAB new -s "$S" --gpu "$GPU"

cleanup() { echo "== stopping session $S"; $COLAB stop -s "$S" || true; }
trap cleanup EXIT

echo "== uploading Kaggle token (file only, never printed)"
TOKEN_FILE="$HOME/.kaggle/access_token"; [ -f "$TOKEN_FILE" ] || TOKEN_FILE="$HOME/.kaggle/kaggle.json"
$COLAB upload -s "$S" "$TOKEN_FILE" /content/kaggle_access_token

echo "== downloading the real State Farm data on the VM"
$COLAB exec -s "$S" -f scripts/colab_setup.py

echo "== training (full run)"
$COLAB exec -s "$S" -f runs/colab/train_full.py

echo "== zipping and downloading results"
echo 'import shutil; shutil.make_archive("/content/outputs", "zip", "/content/outputs"); print("zipped")' | $COLAB exec -s "$S"
$COLAB download -s "$S" /content/outputs.zip runs/colab/outputs.zip
rm -rf runs/colab/outputs && mkdir -p runs/colab/outputs && unzip -q runs/colab/outputs.zip -d runs/colab/outputs
ls runs/colab/outputs
echo "COLAB RUN COMPLETE"
