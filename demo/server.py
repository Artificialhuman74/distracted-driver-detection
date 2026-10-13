"""FastAPI backend for the live demo. Runs both MobileNetV3 models on the laptop CPU.

It returns, for one image:
  - class probabilities and CPU prediction time
  - the most active feature maps from several stages of the network (captured with forward hooks)
  - the pooled 960-number feature vector
  - a Grad-CAM heatmap
"""
import base64
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import models

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = Path(os.environ.get("DEMO_OUTPUTS", ROOT / "outputs"))
STATIC = Path(__file__).resolve().parent / "static"
torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

# Stages we show in the "inside the network" view: (layer index in model.features, title, caption).
STAGES = [
    (1, "Layer 2", "Early layers respond to edges and colour blobs"),
    (3, "Layer 4", "Simple textures: lines, corners, gradients"),
    (6, "Layer 7", "Shapes and parts start to appear"),
    (12, "Layer 13", "Responds to object parts: hands, wheel, face"),
    (15, "Layer 16", "High-level cues such as phone, cup, arm pose"),
    (16, "Layer 17", "960 detectors: the richest features"),
]
MAPS_PER_STAGE = 8
MAX_MAP_SIZE = 112


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m = models.mobilenet_v3_large(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(ckpt["classes"]))
    m.load_state_dict(ckpt["state_dict"])
    m.train(False)  # evaluation mode
    return m, ckpt


MODELS, CKPT = {}, {}
for split in ["random", "driver"]:
    p = OUTPUTS / f"model_{split}.pt"
    if p.exists():
        MODELS[split], CKPT[split] = load_model(p)
if not MODELS:
    raise SystemExit(f"No models found in {OUTPUTS}. Download the Kaggle results first (see README).")

any_ckpt = next(iter(CKPT.values()))
CLASSES, CLASS_NAMES = any_ckpt["classes"], any_ckpt["class_names"]
IMG = any_ckpt["img_size"]
MEAN = torch.tensor(any_ckpt["mean"]).view(1, 3, 1, 1)
STD = torch.tensor(any_ckpt["std"]).view(1, 3, 1, 1)


def preprocess(pil):
    """Same preprocessing as training: RGB, bilinear resize to 224x224, ImageNet normalisation."""
    pil = pil.convert("RGB").resize((IMG, IMG), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.uint8)
    x = torch.from_numpy(arr).permute(2, 0, 1)[None].float() / 255.0
    return (x - MEAN) / STD, arr


def b64_u8(a):
    """uint8 numpy array -> base64 string (the browser draws it on a canvas)."""
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


def to_u8(t):
    t = t - t.min()
    return (255 * t / (t.max() + 1e-8)).byte().numpy()


def run_model(model, x):
    acts = {}
    hooks = [model.features[i].register_forward_hook(lambda m, inp, out, i=i: acts.__setitem__(i, out.detach()))
             for i, _, _ in STAGES]
    with torch.no_grad():
        t = time.perf_counter()
        logits = model(x)
        ms = (time.perf_counter() - t) * 1000
    for h in hooks:
        h.remove()
    probs = F.softmax(logits, 1)[0]

    stages = []
    for i, title, caption in STAGES:
        a = acts[i][0]                                   # C x H x W
        strength = a.mean(dim=(1, 2))
        top = torch.argsort(strength, descending=True)[:MAPS_PER_STAGE]
        maps = a[top]
        if maps.shape[-1] > MAX_MAP_SIZE:
            maps = F.adaptive_avg_pool2d(maps[None], MAX_MAP_SIZE)[0]
        stages.append({
            "title": title, "caption": caption,
            "channels": int(a.shape[0]), "size": int(a.shape[-1]), "map_size": int(maps.shape[-1]),
            "maps": [b64_u8(to_u8(m)) for m in maps],
            "map_ids": top.tolist(),
        })
    pooled = acts[16][0].mean(dim=(1, 2))                 # the 960-number vector after global pooling
    return probs, ms, stages, pooled


def grad_cam(model, x, class_idx):
    store = {}

    def fwd(m, inp, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    h = model.features[-1].register_forward_hook(fwd)
    with torch.enable_grad():
        logits = model(x)
        model.zero_grad()
        logits[0, class_idx].backward()
    h.remove()
    w = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * store["act"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG, IMG), mode="bilinear", align_corners=False)[0, 0].detach()
    return to_u8(cam)


def analyse(pil, which):
    x, arr = preprocess(pil)
    out = {"input": b64_u8(arr), "size": IMG, "models": {}}
    for split in which:
        model = MODELS[split]
        probs, ms, stages, pooled = run_model(model, x)
        pred = int(probs.argmax())
        out["models"][split] = {
            "probs": [round(float(p), 5) for p in probs],
            "pred": CLASSES[pred], "pred_name": CLASS_NAMES[CLASSES[pred]],
            "confidence": float(probs[pred]),
            "ms": round(ms, 1),
            "stages": stages,
            "pooled": [round(float(v), 4) for v in pooled],
            "gradcam": b64_u8(grad_cam(model, x, pred)),
            "best_val_acc": CKPT[split].get("best_val_acc"),
        }
    return out


app = FastAPI(title="Distracted Driver Detection demo")


@app.get("/api/info")
def info():
    manifest_path = OUTPUTS / "sample_images" / "manifest.json"
    samples = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    return {"classes": CLASSES, "class_names": CLASS_NAMES, "models": list(MODELS),
            "samples": samples, "smoke": bool(any_ckpt.get("smoke")),
            "threads": torch.get_num_threads()}


@app.get("/api/metrics")
def metrics():
    p = OUTPUTS / "metrics.json"
    if not p.exists():
        raise HTTPException(404, "metrics.json not found")
    return JSONResponse(json.loads(p.read_text()))


@app.post("/api/predict")
async def predict(file: UploadFile | None = File(None), sample: str | None = Form(None),
                  models_: str = Form("random,driver", alias="models")):
    if sample:
        p = (OUTPUTS / "sample_images" / Path(sample).name)
        if not p.exists():
            raise HTTPException(404, "unknown sample")
        pil = Image.open(p)
    elif file is not None:
        try:
            pil = Image.open(io.BytesIO(await file.read()))
        except Exception:
            raise HTTPException(400, "Could not read that image file")
    else:
        raise HTTPException(400, "Send an image file or a sample name")
    which = [m for m in models_.split(",") if m in MODELS] or list(MODELS)
    return analyse(pil, which)


@app.get("/samples/{name}")
def sample_file(name: str):
    p = OUTPUTS / "sample_images" / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/figures/{name}")
def figure_file(name: str):
    p = OUTPUTS / "figures" / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
