# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Brief-pinned: frontend is a single page in plain HTML/CSS/JS with every asset stored locally (no CDN, must work offline). Backend is FastAPI + PyTorch on the laptop CPU (`demo/server.py`), started with `python demo/run.py`, which opens the browser.

## Users

Team F-28 (Chiranth R, Vihaan Wadhwa) presenting their ML course mini project, "Machine Learning Techniques for Distracted Driver Detection", in a classroom. The audience is **faculty evaluators and classmates equally**: lead with the wow, then show the evidence. **Evaluators may take the mouse and try it themselves**, so the page must make sense to someone who has never seen it, with no narration.

## Product Purpose

A live demo that classifies a dashboard-camera photo of a driver into one of 10 activities (c0 safe driving … c9 talking to passenger). It makes the neural network's inner workings visible as the image flows through MobileNetV3. It also makes the project's main finding tangible: a model trained on a random split, where the same drivers appear in train and test, scores much higher than one trained and tested on separate drivers. Part of what it learned is the people, not only the actions.

Success: after a few minutes, an evaluator can explain (1) what the classifier does, (2) roughly what the layers detect, and (3) why random-split accuracy is inflated.

## Positioning

It runs two models side by side on the *same unseen image*: one trained on a random split and one trained on a driver split. A generic classifier demo can't show this. The honest comparison is the point.

## Operating Context

- Shown on a **classroom projector**, so it must read from the back of the room: large type, high contrast, nothing important in thin or low-contrast detail.
- Runs on an Apple M1 laptop (8 GB), CPU only, offline. One prediction for both models takes well under a second.
- Used live during a short presentation and possibly handed to evaluators afterwards.

## Capabilities and Constraints

- Input: a gallery of ~30 real test images from the driver-split **test drivers**, chosen so neither model trained on them. Also drag-and-drop upload, and an *experimental* webcam snapshot (the webcam angle differs from the dataset's camera, so it must be labelled experimental).
- "Inside the network": input → 6 stages of MobileNetV3 (top 8 most-active feature maps each) → global-pooled 960-number vector → 10 class probability bars. Stages light up in sequence, each with a plain-English caption.
- Grad-CAM heatmap over the image with an opacity slider.
- Side-by-side mode: random-split model vs driver-split model on the same image.
- CPU prediction time in ms is shown.
- A Results tab reads `outputs/metrics.json` and the figures.
- Terminology: "random split" (same drivers in train and test) and "driver split" (test drivers never seen). Classes use their dataset codes c0–c9 plus short names.

## Evidence on Hand

- `outputs/metrics.json`: every number (accuracy, macro-F1, per-class, confusion matrices, per-driver accuracy, timings), produced by the Kaggle full run.
- `outputs/figures/*.png`: confusion matrices, split comparison chart, per-driver accuracy, Grad-CAM grid, training curves, class examples.
- `outputs/sample_images/`: real State Farm images plus `manifest.json`.
- No number may be invented or estimated. Every figure shown must come from `metrics.json` or a live prediction.

## Product Principles

1. Honesty over spectacle: the dramatic part is real. Never hide the driver-split model's lower accuracy.
2. Self-explanatory: every view says in plain words what it shows.
3. Legible at projector scale before anything else.
4. The same image is always the anchor for comparisons.

## Accessibility & Inclusion

Projector legibility: high contrast and large type. Correct/incorrect and random/driver must never be told apart by red/green colour alone; pair colour with labels or icons.
