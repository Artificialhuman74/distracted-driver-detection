#!/usr/bin/env bash
# Run the debiasing experiment (scripts/debias_experiment.py) on Google Colab via the Colab CLI.
#   bash scripts/colab_debias.sh [GPU]      -> runs/debias/   (needs outputs/ from the main run)
set -euo pipefail
cd "$(dirname "$0")/.."
COLAB=".venv/bin/colab"; GPU="${1:-T4}"; S="dd-f28-debias"
mkdir -p runs/debias
for f in outputs/splits/driver_split.csv outputs/model_driver.pt outputs/model_random.pt outputs/metrics.json; do [ -f "$f" ]; done

echo "== creating Colab session $S with GPU $GPU"
$COLAB new -s "$S" --gpu "$GPU"
cleanup() { echo "== stopping session $S"; $COLAB stop -s "$S" || true; }
trap cleanup EXIT

echo "== uploading Kaggle token (file only, never printed), split file and main-run models"
TOKEN_FILE="$HOME/.kaggle/access_token"; [ -f "$TOKEN_FILE" ] || TOKEN_FILE="$HOME/.kaggle/kaggle.json"
$COLAB upload -s "$S" "$TOKEN_FILE" /content/kaggle_access_token
$COLAB upload -s "$S" outputs/splits/driver_split.csv /content/driver_split.csv
$COLAB upload -s "$S" outputs/model_driver.pt /content/model_driver.pt
$COLAB upload -s "$S" outputs/model_random.pt /content/model_random.pt
$COLAB upload -s "$S" outputs/metrics.json /content/metrics.json

echo "== downloading the real State Farm data on the VM"
$COLAB exec -s "$S" -f scripts/colab_setup.py

echo "== debiasing experiment"
$COLAB exec -s "$S" -f scripts/debias_experiment.py

echo "== zipping and downloading results"
echo 'import shutil; shutil.make_archive("/content/debias", "zip", "/content/debias"); print("zipped")' | $COLAB exec -s "$S"
$COLAB download -s "$S" /content/debias.zip runs/debias/debias.zip
rm -rf runs/debias/out && mkdir -p runs/debias/out && unzip -q runs/debias/debias.zip -d runs/debias/out
ls -R runs/debias/out | head -20
echo "DEBIAS RUN COMPLETE"
