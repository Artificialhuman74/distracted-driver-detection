"""Debiasing experiment: can MobileNetV3 do better on UNSEEN drivers without learning what people look like?

Runs ON a Colab GPU (sent by scripts/colab_debias.sh). Uses the REAL State Farm data and the SAME
driver split as the main run (driver_split.csv uploaded from outputs/splits/).

Variants (all MobileNetV3-Large, ImageNet-pretrained, 8 epochs, same optimiser as the main run):
  baseline   : the main run's driver-split model (re-evaluated here as a sanity check, not retrained)
  aug        : Option 1, stronger appearance-scrambling augmentation (random erasing, random grayscale,
               stronger colour jitter)
  aug+adv0.3 : Option 1 + Option 2, a driver-ID head behind a gradient-reversal layer (lambda up to 0.3)
  aug+adv1.0 : same, lambda up to 1.0
Model selection: each variant keeps its best epoch by VALIDATION-driver accuracy, and the winning variant is
chosen by validation accuracy only. Test numbers are reported for all variants as an ablation.
Identity probe: logistic regression predicting WHICH of the 4 test drivers is in a photo, from each model's
frozen 960-d pooled features (50/50 split of those photos). Analysis only; never used for selection.
"""
import json, math, os, random, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED, EPOCHS, BS, LR, IMG = 42, 8, 64, 1e-3, 224
T0 = time.time(); TIMINGS = {}
def stamp(name, t):
    TIMINGS[name] = round(time.time() - t, 1)
    print(f"----- {name} took {(time.time()-t)/60:.1f} min (total {(time.time()-T0)/60:.1f} min)", flush=True)
def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

DEVICE = "cuda"; assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
print("GPU:", torch.cuda.get_device_name(0), flush=True)
OUT = Path("/content/debias"); (OUT / "figures").mkdir(parents=True, exist_ok=True)
CLASSES = [f"c{i}" for i in range(10)]

# ---------------- data: same images, same driver split as the main run ----------------
t = time.time()
DATA = Path("/content/data")
df = pd.read_csv(DATA / "driver_imgs_list.csv")
assert df["img"].is_unique
split = pd.read_csv("/content/driver_split.csv")[["img", "split"]]
df = df.merge(split, on="img", how="inner").reset_index(drop=True)
assert len(df) == 22424, len(df)
df["label"] = df["classname"].str[1].astype(int)
y_all = df["label"].values
idx = {p: np.where(df["split"].values == p)[0] for p in ["train", "val", "test"]}
train_drivers = sorted(df.loc[idx["train"], "subject"].unique())
test_drivers = sorted(df.loc[idx["test"], "subject"].unique())
drv_id = {d: i for i, d in enumerate(train_drivers)}
d_all = np.array([drv_id.get(s, -1) for s in df["subject"]])
print({p: len(v) for p, v in idx.items()}, "train drivers", len(train_drivers), "test drivers", test_drivers, flush=True)

def load_one(p):
    with Image.open(p) as im:   # same preprocessing as the main run: RGB, bilinear resize to 224
        return np.asarray(im.convert("RGB").resize((IMG, IMG), Image.BILINEAR), dtype=np.uint8)
paths = [str(DATA / "imgs" / "train" / c / f) for c, f in zip(df["classname"], df["img"])]
X = np.empty((len(df), IMG, IMG, 3), dtype=np.uint8)
with ThreadPoolExecutor(16) as ex:
    for i, a in enumerate(ex.map(load_one, paths)):
        X[i] = a
stamp("load images", t)

MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

def gray(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

def augment(x, strong):
    """Batch augmentation on the GPU. Base = the main run's light augmentation (no horizontal flip).
    strong=True adds appearance scrambling: wider colour jitter, random grayscale (p=0.2) and random erasing (p=0.5)."""
    B, dev = x.size(0), x.device
    u = lambda lo, hi, *shape: lo + (hi - lo) * torch.rand(*(shape or (B,)), device=dev)
    ang, scale = u(-8, 8) * math.pi / 180, u(0.9, 1.1)
    tx, ty = u(-0.1, 0.1), u(-0.1, 0.1)
    c, s = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([c, -s, tx], 1), torch.stack([s, c, ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=False), align_corners=False)
    lo_b, hi_b, lo_s, hi_s = (0.6, 1.4, 0.5, 1.5) if strong else (0.7, 1.3, 0.8, 1.2)
    x = x * u(lo_b, hi_b, B, 1, 1, 1)
    m = gray(x).mean(dim=(2, 3), keepdim=True)
    x = (x - m) * u(lo_b, hi_b, B, 1, 1, 1) + m
    g = gray(x); x = (x - g) * u(lo_s, hi_s, B, 1, 1, 1) + g
    if strong:
        sel = (torch.rand(B, 1, 1, 1, device=dev) < 0.2)
        x = torch.where(sel, gray(x).expand_as(x), x)
        H, W = x.shape[2:]
        area = u(0.02, 0.2) * H * W
        ratio = torch.exp(u(math.log(0.3), math.log(3.3)))
        h = torch.sqrt(area * ratio).clamp(max=H); w = torch.sqrt(area / ratio).clamp(max=W)
        y0 = torch.rand(B, device=dev) * (H - h); x0 = torch.rand(B, device=dev) * (W - w)
        ys = torch.arange(H, device=dev).view(1, H, 1); xs = torch.arange(W, device=dev).view(1, 1, W)
        box = (ys >= y0.view(B, 1, 1)) & (ys < (y0 + h).view(B, 1, 1)) & (xs >= x0.view(B, 1, 1)) & (xs < (x0 + w).view(B, 1, 1))
        box = box & (torch.rand(B, device=dev) < 0.5).view(B, 1, 1)
        x = torch.where(box[:, None], torch.rand_like(x), x)
    return x.clamp(0, 1)

def to_input(xu8, aug=None):
    x = xu8.float() / 255.0
    if aug is not None:
        x = augment(x, strong=aug)
    return ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)

def gpu_batch(b):
    return torch.from_numpy(X[b]).pin_memory().to(DEVICE, non_blocking=True).permute(0, 3, 1, 2)

# ---------------- model with optional adversarial driver head ----------------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None

class Net(nn.Module):
    """MobileNetV3-Large; the driver head sees the 960-d pooled features through a gradient-reversal layer,
    so the backbone is pushed to make those features useless for telling drivers apart."""
    def __init__(self, n_drivers=0):
        super().__init__()
        self.m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        self.m.classifier[3] = nn.Linear(1280, 10)
        self.drv = nn.Sequential(nn.Linear(960, 256), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(256, n_drivers)) if n_drivers else None
    def feats(self, x):
        return self.m.avgpool(self.m.features(x)).flatten(1)
    def forward(self, x, lam=0.0):
        f = self.feats(x)
        return self.m.classifier(f), (self.drv(GradReverse.apply(f, lam)) if self.drv is not None else None)

def load_torchvision(path):
    m = models.mobilenet_v3_large(weights=None); m.classifier[3] = nn.Linear(1280, 10)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
    return m.to(DEVICE).to(memory_format=torch.channels_last)

@torch.no_grad()
def predict(m, ids):
    m.train(False); out = np.empty((len(ids), 10), dtype=np.float32)
    for i in range(0, len(ids), 256):
        b = ids[i:i + 256]
        with torch.autocast("cuda", dtype=torch.float16):
            out[i:i + len(b)] = F.softmax(m(to_input(gpu_batch(b))).float(), 1).cpu().numpy()
    return out

@torch.no_grad()
def features(backbone_fn, ids):
    feats = []
    for i in range(0, len(ids), 256):
        with torch.autocast("cuda", dtype=torch.float16):
            feats.append(backbone_fn(to_input(gpu_batch(ids[i:i + 256]))).float().cpu().numpy())
    return np.concatenate(feats)

def metrics(pred, ids):
    yt = y_all[ids]; cm = confusion_matrix(yt, pred, labels=list(range(10)))
    pd_acc = {d: float((pred[df.loc[ids, "subject"].values == d] == yt[df.loc[ids, "subject"].values == d]).mean()) for d in test_drivers} \
        if set(df.loc[ids, "subject"]) <= set(test_drivers) else None
    return {"accuracy": float((pred == yt).mean()), "macro_f1": float(f1_score(yt, pred, average="macro")),
            "per_class_accuracy": {CLASSES[i]: float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(10)},
            "confusion_matrix": cm.tolist(), "per_test_driver_accuracy": pd_acc, "n": int(len(ids))}

def train_variant(name, strong, lam_max):
    seed_all(); t = time.time()
    net = Net(len(train_drivers) if lam_max > 0 else 0).to(DEVICE).to(memory_format=torch.channels_last)
    steps = EPOCHS * (len(idx["train"]) // BS)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda"); ce = nn.CrossEntropyLoss(label_smoothing=0.1); ce_d = nn.CrossEntropyLoss()
    hist, best, best_state, best_ep, step = [], -1, None, -1, 0
    for ep in range(1, EPOCHS + 1):
        net.train(); te = time.time(); n = 0; d_ok = torch.zeros((), device=DEVICE); tot = torch.zeros((), device=DEVICE)
        order = idx["train"][np.random.permutation(len(idx["train"]))]
        for i in range(0, len(order) - BS + 1, BS):
            b = np.sort(order[i:i + BS])
            x, y = gpu_batch(b), torch.from_numpy(y_all[b]).to(DEVICE)
            p = step / steps; lam = lam_max * (2 / (1 + math.exp(-10 * p)) - 1)   # standard DANN ramp-up
            with torch.autocast("cuda", dtype=torch.float16):
                logits, dlog = net(to_input(x, aug=strong), lam)
                loss = ce(logits, y)
                if dlog is not None:
                    dy = torch.from_numpy(d_all[b]).to(DEVICE)
                    loss = loss + ce_d(dlog, dy)
                    d_ok += (dlog.argmax(1) == dy).sum()
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.detach() * len(b); n += len(b); step += 1
        torch.cuda.synchronize(); ips = n / (time.time() - te)
        va = float((predict(net.m, idx["val"]).argmax(1) == y_all[idx["val"]]).mean())
        rec = {"epoch": ep, "loss": float(tot) / n, "val_acc": va, "img_per_s": round(ips)}
        if net.drv is not None:
            rec["driver_head_train_acc"] = float(d_ok) / n
        hist.append(rec)
        flag = ""
        if va > best:
            best, best_ep, flag = va, ep, "  <- best"
            best_state = {k: v.detach().cpu().clone() for k, v in net.m.state_dict().items()}
        extra = f"  driver-head acc {rec['driver_head_train_acc']:.3f}" if net.drv is not None else ""
        print(f"  [{name}] epoch {ep}/{EPOCHS} loss {rec['loss']:.4f} val {va:.4f}{extra} ({ips:.0f} img/s){flag}", flush=True)
    net.m.load_state_dict(best_state)
    stamp(f"train {name}", t)
    return net.m, hist, best_ep, best

# ---------------- run ----------------
results = {}
t = time.time()
base = load_torchvision("/content/model_driver.pt")
results["baseline"] = {"desc": "main run driver-split model (light augmentation), re-evaluated",
                       "val_acc": float((predict(base, idx["val"]).argmax(1) == y_all[idx["val"]]).mean()),
                       "test": metrics(predict(base, idx["test"]).argmax(1), idx["test"]), "history": None, "best_epoch": None}
print(f"SANITY baseline re-evaluated: val {results['baseline']['val_acc']:.4f} test {results['baseline']['test']['accuracy']:.4f} (main run: 0.8748)", flush=True)
stamp("baseline re-eval", t)

trained = {"baseline": base}
for name, strong, lam in [("aug", True, 0.0), ("aug+adv0.3", True, 0.3), ("aug+adv1.0", True, 1.0)]:
    m, hist, best_ep, best_val = train_variant(name, strong, lam)
    trained[name] = m
    results[name] = {"desc": {"aug": "Option 1: appearance-scrambling augmentation",
                              "aug+adv0.3": "Option 1 + 2: + adversarial driver head (gradient reversal, lambda<=0.3)",
                              "aug+adv1.0": "Option 1 + 2: + adversarial driver head (gradient reversal, lambda<=1.0)"}[name],
                     "val_acc": best_val, "best_epoch": best_ep, "history": hist,
                     "test": metrics(predict(m, idx["test"]).argmax(1), idx["test"])}
    print(f"  [{name}] best epoch {best_ep}: val {best_val:.4f} -> test {results[name]['test']['accuracy']:.4f} "
          f"macro-F1 {results[name]['test']['macro_f1']:.4f}", flush=True)

# selection by VALIDATION only, among the new variants and the baseline
selected = max(results, key=lambda k: results[k]["val_acc"])
print("SELECTED BY VALIDATION:", selected, flush=True)

# ---------------- identity probe on the 4 test drivers ----------------
t = time.time()
ids = idx["test"]; subj = df.loc[ids, "subject"].values
tr_i, te_i = train_test_split(np.arange(len(ids)), test_size=0.5, stratify=subj, random_state=SEED)
probe_models = {"imagenet (no fine-tuning)": None, "random-split model": load_torchvision("/content/model_random.pt"), **trained}
probe = {}
for name, m in probe_models.items():
    if m is None:
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2).to(DEVICE).to(memory_format=torch.channels_last)
    m.train(False)
    Fz = features(lambda x: m.avgpool(m.features(x)).flatten(1), ids)
    sc = StandardScaler().fit(Fz[tr_i])
    clf = LogisticRegression(max_iter=3000, C=0.5).fit(sc.transform(Fz[tr_i]), subj[tr_i])
    probe[name] = float(clf.score(sc.transform(Fz[te_i]), subj[te_i]))
    print(f"  identity probe [{name}]: {probe[name]:.4f}  (chance 0.25)", flush=True)
stamp("identity probe", t)

# ---------------- Grad-CAM: baseline vs selected, same 12 photos as the main run ----------------
def grad_cam(m, i):
    m.train(False); st = {}
    def hook(mod, inp, out):
        st["a"] = out; out.register_hook(lambda g: st.__setitem__("g", g))
    h = m.features[-1].register_forward_hook(hook)
    x = to_input(torch.from_numpy(X[i]).permute(2, 0, 1)[None].to(DEVICE)).float()
    with torch.enable_grad():
        lo = m(x); p = int(lo.argmax(1)); m.zero_grad(); lo[0, p].backward()
    h.remove()
    cam = F.relu((st["g"].mean((2, 3), keepdim=True) * st["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG, IMG), mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy(), p, float(F.softmax(lo, 1)[0, p])

main_metrics = json.load(open("/content/metrics.json"))
gc_imgs = [r["img"] for r in main_metrics["gradcam"]]
gc_idx = [int(np.where(df["img"].values == im)[0][0]) for im in gc_imgs]
if selected != "baseline":
    fig, axes = plt.subplots(6, 6, figsize=(14.4, 15)); [a.axis("off") for a in axes.ravel()]
    for k, i in enumerate(gc_idx):
        r, c0 = divmod(k, 2); c0 *= 3
        axes[r, c0].imshow(X[i]); axes[r, c0].set_title(f"true {CLASSES[y_all[i]]}", fontsize=8)
        for off, name in [(1, "baseline"), (2, selected)]:
            cam, p, pr = grad_cam(trained[name], i)
            axes[r, c0 + off].imshow(X[i]); axes[r, c0 + off].imshow(cam, cmap="jet", alpha=0.45)
            axes[r, c0 + off].set_title(f"{'driver-split' if name == 'baseline' else 'debiased'}: {CLASSES[p]} {pr:.0%}",
                                        fontsize=8, color="green" if p == y_all[i] else "red")
    fig.suptitle(f"Grad-CAM: driver-split baseline vs debiased ({selected}), same unseen photos", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "figures" / "gradcam_debiased.png", dpi=140); plt.close(fig)
    torch.save({"state_dict": trained[selected].state_dict(), "arch": "mobilenet_v3_large", "classes": CLASSES,
                "class_names": main_metrics["class_names"], "img_size": IMG,
                "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "split": "driver",
                "variant": selected, "best_epoch": results[selected]["best_epoch"], "best_val_acc": results[selected]["val_acc"],
                "smoke": False}, OUT / "model_debiased.pt")

# ---------------- figures + json ----------------
order = ["baseline", "aug", "aug+adv0.3", "aug+adv1.0"]
fig, ax = plt.subplots(figsize=(8, 4.2)); xs = np.arange(len(order)); w = 0.38
for off, key, col, lab in [(-w / 2, "val", "#8592a3", "validation drivers (used to choose)"), (w / 2, "test", "#3d8fe0", "test drivers (reported)")]:
    vals = [results[k]["val_acc"] if key == "val" else results[k]["test"]["accuracy"] for k in order]
    bars = ax.bar(xs + off, vals, w, color=col, label=lab)
    for b in bars: ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, f"{b.get_height():.1%}", ha="center", fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels(order); ax.set_ylim(0.5, 1.0); ax.set_ylabel("Accuracy on unseen drivers")
ax.set_title("Debiasing variants (driver split)"); ax.legend(loc="lower right", fontsize=8); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "figures" / "debias_accuracy.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.2)); names = list(probe)
bars = ax.barh(names, [probe[n] for n in names], color="#7fb8a8"); ax.axvline(0.25, color="k", ls="--", lw=1, label="chance (4 drivers)")
for b in bars: ax.text(b.get_width() + 0.005, b.get_y() + b.get_height() / 2, f"{b.get_width():.1%}", va="center", fontsize=8)
ax.set_xlim(0, 1.05); ax.set_xlabel("Probe accuracy: which of the 4 test drivers is this?"); ax.invert_yaxis()
ax.set_title("Identity probe on frozen features (lower = less 'who is this' information)"); ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(OUT / "figures" / "identity_probe.png", dpi=150); plt.close(fig)

TIMINGS["total"] = round(time.time() - T0, 1)
json.dump({"variants": results, "selected_by_validation": selected, "identity_probe": probe,
           "probe_setup": "logistic regression (C=0.5) on standardised 960-d pooled features; 4 driver-split test drivers; 50/50 stratified split, seed 42; chance 0.25",
           "augmentation_strong": "base light aug + brightness/contrast 0.6-1.4, saturation 0.5-1.5, random grayscale p=0.2, random erasing p=0.5 (2-20% area, noise fill); no flip",
           "adversary": "960-d pooled features -> gradient reversal -> Linear(960,256)-Hardswish-Dropout(0.2)-Linear(256,18 training drivers); lambda ramp 2/(1+exp(-10p))-1 times lambda_max",
           "epochs": EPOCHS, "seed": SEED, "gpu": torch.cuda.get_device_name(0), "timings_seconds": TIMINGS},
          open(OUT / "metrics_debias.json", "w"), indent=1)
print("DEBIAS SUMMARY")
for k in order:
    print(f"  {k:12s} val {results[k]['val_acc']:.4f}  test {results[k]['test']['accuracy']:.4f}  F1 {results[k]['test']['macro_f1']:.4f}")
print("  selected (by validation):", selected, "| total", TIMINGS["total"], "s", flush=True)
