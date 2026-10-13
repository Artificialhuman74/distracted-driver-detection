# Machine Learning Techniques for Distracted Driver Detection

**Team F-28 · Project 75 · Section F**: Chiranth R (PES1UG24AM354), Vihaan Wadhwa (PES1UG24AM319)

Reference: Stanford CS229 (Spring 2019), project 24: [report](https://cs229.stanford.edu/proj2019spr/report/24.pdf) · [poster](https://cs229.stanford.edu/proj2019spr/poster/24.pdf)

Dataset: Kaggle competition [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection). It has 22,424 labelled images in 10 classes, taken of 26 drivers.

**Main finding:** a random train/test split puts the *same drivers* in both train and test, which makes accuracy look very high. When we test on drivers the model has never seen (a driver-grouped split), accuracy drops. Part of what the model learns is *who* the driver is, not only *what* they are doing.

> **Data note.** This repository does **not** include any State Farm images. Kaggle's competition rules do not allow redistributing them. To reproduce the results, accept the competition rules on Kaggle and run the training (step 1 below). That run also creates `outputs/sample_images/`, the 30 unseen test photos the demo's photo rack uses. Without that folder, the demo still works with uploaded or webcam photos.

## Folder layout

| Folder | What's inside |
|---|---|
| `kaggle/` | `train.ipynb` (built from `scripts/train_notebook_src.py`) + `kernel-metadata.json` |
| `scripts/` | notebook source/builder, Kaggle run helper, slide and report generators |
| `outputs/` | real results downloaded from the full Kaggle run: `metrics.json`, `figures/`, `splits/`, models, `sample_images/` |
| `runs/` | raw downloads + logs from each Kaggle run (`smoke/`, `full/`) |
| `demo/` | live demo app (FastAPI + plain HTML/CSS/JS, works offline) |
| `deliverables/` | `slides.pptx`, `report.docx`, `report.pdf` |

## Exact steps

### 0. One-time setup (laptop)
```bash
python3 -m venv .venv
.venv/bin/pip install kaggle
# Kaggle token: put it in ~/.kaggle/access_token (or ~/.kaggle/kaggle.json), chmod 600. Never commit it.
.venv/bin/kaggle competitions files state-farm-distracted-driver-detection   # should list files
```
You must have accepted the competition rules on Kaggle.

### 1. Train on Kaggle's GPU (real data)
```bash
bash scripts/kaggle_run.sh smoke   # ~1,000 real images, 1 epoch: bug check, a few minutes -> runs/smoke/
bash scripts/kaggle_run.sh full    # all 22,424 images             -> runs/full/ and copied to outputs/
```
The script builds `kaggle/train.ipynb`, pushes it with `kaggle kernels push -p kaggle`, polls `kaggle kernels status` every 2 minutes, and then downloads the results with `kaggle kernels output`.
The notebook writes everything to `/kaggle/working/outputs/`. Downloads go to `runs/<stage>/` instead of `outputs/` directly, which keeps them from ending up nested inside `outputs/outputs/`. The full run is then copied to `outputs/`.

#### Alternative: Google Colab through the Colab CLI (this is what produced the final `outputs/`)
```bash
.venv/bin/pip install google-colab-cli
.venv/bin/colab sessions            # one-time Google sign-in (open the link, paste the code back)
bash scripts/colab_run.sh T4        # T4 = free; L4/A100 need Colab compute units
```
The script creates a GPU session and uploads the Kaggle token as a file, without printing it. It then downloads the real data on the VM (`scripts/colab_setup.py`) and runs the same training code with `SMOKE = False`. Finally it downloads the results to `runs/colab/outputs/` and stops the session. The final results were then copied to `outputs/`, and the full log is in `outputs/run_log.txt`. On a free T4 the full run took 14.4 minutes. Image augmentation runs on the GPU.

#### Extra experiments (Colab CLI, free T4)
```bash
bash scripts/colab_debias.sh T4          # single driver split: baseline vs A (scramble) vs B/C (+ adversarial driver head) -> outputs/debias/
bash scripts/colab_cv.sh T4 0 1 2        # 5-fold driver-grouped CV x 3 seeds (45 runs, ~2.5 h); resumes after disconnects
.venv/bin/python scripts/analyze_cv.py   # mean ± std, paired tests, identity probes -> outputs/cv/
```

### 2. Demo
```bash
.venv/bin/pip install -r demo/requirements.txt
.venv/bin/python demo/run.py        # loads both models from outputs/ on the CPU and opens http://127.0.0.1:8000
```
Works offline: fonts and code are stored in `demo/static/`, with no CDN.

- **Inside the network** shows the photo flowing left to right through MobileNetV3. Each stack holds the 8 most active real feature maps of that layer, drawn pixel for pixel. Then come the 960 averaged numbers and the 10 outputs. Controls: **Space** replays, and **← / →** step through the 9 stages. Click a stack to see its maps up close. Tick "Where it looked" to show Grad-CAM with an opacity slider. The switch at the top picks which model to look inside.
- **Two models** runs the random-split model and the driver-split model on the same photo, showing predictions, Grad-CAM and CPU time side by side.
- **Results** shows the charts, the table and the confusion matrices, all read from `outputs/metrics.json`.
- **Inputs:** real test photos in the rack (from the 4 driver-split test drivers, never used for training), upload or drag-and-drop, and an *experimental* webcam snapshot.
- Jump straight to a tab with `http://127.0.0.1:8000/#compare` or `#results`.
- To try the demo on the smoke-run models: `DEMO_OUTPUTS=runs/smoke/outputs .venv/bin/python demo/run.py`.

### 3. Slides and report (they read only `outputs/metrics.json` + `outputs/figures/`)
```bash
.venv/bin/python scripts/make_slides.py
.venv/bin/python scripts/make_report.py
```

## Rules we followed
- Only real Kaggle data is used; no synthetic data.
- No test-set information is used for training, normalization or model selection. The scaler is fitted on train only, and the best epoch is picked on validation.
- Every number in the slides and report is read from `outputs/metrics.json`.
